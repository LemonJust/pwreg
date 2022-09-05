import json
import numpy as np
import pandas as pd
import random
import tifffile as tif
from tqdm import tqdm
import warnings
import shutil
import scipy.ndimage as nd
import os

from .utils import *
from .viewers import BlockView, BlockPairView

try:
    import ants
except:
    print("Exception occured when trying to import ants !\n "
          "BlockPair.register method won't work and will cause an error if you use it. ")


class Image:
    def __init__(self, resolution, filename=None, img=None, info=None, mask=None):
        """
        mask : dict with xmin, xmax, ymin, ymax, zmin, zmax optional
        ( fields can be empty if don't need to crop the corresponding dimention).
        """

        assert filename is not None or img is not None, "Provide filename or img."
        assert (filename is None) is not (img is None), "Provide exactly one of the following: filename or img."

        self.filename = filename
        self.resolution = np.array(resolution)
        self.info = info

        if img is None:
            self.img = self.read_image()
        else:
            self.img = img

        self.shape = self.img.shape

        self.mask = mask
        if self.mask is not None:
            self.mask = self.crop(mask)

        self.shape = self.img.shape

    def read_image(self):
        """
        Reads image in ZYX order.
        """""
        img = tif.imread(self.filename)
        return img

    def crop(self, mask):
        """
        Crops an image: drops everything outside a rectangle mask (in pixels) and remembers the parameters of the crop.
        mask : dict with xmin, xmax, ymin, ymax, zmin, zmax optional ( fields can be empty if don't need to crop there).
        """

        for key in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
            if key not in mask:
                mask[key] = None

        self.img = self.img[
                   mask['zmin']:mask['zmax'],
                   mask['ymin']:mask['ymax'],
                   mask['xmin']:mask['xmax']
                   ]

        self.shape = self.img.shape

        return mask

    def blur(self, sigma):
        """
        sigma : gaussian filter parameters, ZYX order , in pixels
        (from scipy : scalar or sequence of scalars Standard deviation for Gaussian kernel. The
        standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.)
        """
        img = nd.gaussian_filter(self.img, sigma)
        return Image(self.resolution, img=img, info=f"Blurred {self.filename}")

    def dog(self, sigma1, sigma2):
        """
        sigma1, sigma2 : gaussian filter parameters, ZYX order , in pixels
        (from scipy : scalar or sequence of scalars Standard deviation for Gaussian kernel. The
        standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.)
        """
        img = nd.gaussian_filter(self.img, sigma2) - nd.gaussian_filter(self.img, sigma1)
        return Image(self.resolution, img=img, info=f"DoG {self.filename}")

    def normalize(self, min_value=None, max_value=None):
        """
        min_value, max_value if you want to provide the range to which to normalize
        will set everything below min_value to zero
        """
        SCALE = 1000
        if min_value is None:
            min_value = np.min(self.img)
        if max_value is None:
            max_value = np.max(self.img)

        img = self.img
        img[img < min_value] = min_value
        img = (img - min_value) / (max_value - min_value)
        img = img * SCALE

        return Image(self.resolution, img=img.astype(np.uint16), info=f"Normalised {self.filename}")

    def threshold(self, thr, binary=True):
        """
        Image threshold. Returns an image.
        binary : wether to return a binary mask or the original pixel values ( only the ones above the threshold ).
        """
        img = (self.img > thr)

        if not binary:
            img = self.img * img

        return Image(self.resolution, img=img.astype(np.uint16), info=f"Threshold {self.filename}, thr = {thr}")

    def local_max(self, order):
        """
        Finds local maxima in Image.
        Returns points.
        """
        # local_maxima_3D from utils
        coords, values = local_maxima_3D(self.img, order=order)
        return Points(coords, units='pix', resolution=self.resolution, info={'values': values})

    def split(self, blc_size, overlap):
        """
        Prepares a list of voxels for a specific padding and overlap.
        For now the voxels should split the image perfectly.
        :param blc_size: can be int or 3x1 numpy array representing 3 voxel sides
        :param overlap: can be int or 3x1 numpy array for overlap along 3 voxel sides
        :return: list of voxels of class Voxel
        """

        def input_check(some_input):
            if isinstance(some_input, np.ndarray):
                some_input = some_input
            elif isinstance(some_input, int):
                some_input = np.array([1, 1, 1]) * some_input
            elif isinstance(some_input, float):
                assert some_input.is_integer(), "Voxel size must be integer"
                some_input = np.array([1, 1, 1]) * int(some_input)
            elif isinstance(some_input, list):
                some_input = np.array(some_input)
            else:
                raise TypeError("Only integers, whole floats, lists or numpy arrays are allowed")
            return some_input

        blc_size = input_check(blc_size)
        overlap = input_check(overlap)
        img_size = np.array(self.shape)
        # get number of blocks along each dimension
        num_blc = (img_size - overlap) / (blc_size - overlap)
        if not np.all([n.is_integer() for n in num_blc]):
            warnings.warn(f"Specified voxel size + overlap don't cover the whole image."
                          f"Image size is {img_size}, block size {blc_size},"
                          f" overlap {overlap} results in {num_blc} blocks.\nLeaving some image out. ")
        num_blc = num_blc.astype(int)

        blocks = []
        for nz in np.arange(num_blc[0]):
            # block start z pixel
            tlz = int((blc_size[0] - overlap[0]) * nz)
            for ny in np.arange(num_blc[1]):
                # block start y pixel
                tly = int((blc_size[1] - overlap[1]) * ny)
                for nx in np.arange(num_blc[2]):
                    # block start x pixel
                    tlx = int((blc_size[2] - overlap[2]) * nx)
                    blocks.append(Block(self, [tlz, tly, tlx], blc_size, overlap, [nz, ny, nx], num_blc))
        return blocks

    @staticmethod
    def get_image_shape(img_file):
        # get file info
        stack = TiffFile(img_file, _multifile=False)
        z_size = len(stack.pages)
        page = stack.pages.get(0)
        y_size, x_size = page.shape
        stack.close()
        return (z_size, y_size, x_size)

    def to_dict(self):
        """
        Creates a disctionary representation of the Image object.
        Can only be used with the Images , that have the corresponding image file saved on drive.
        """
        # assert self.filename is not None, "Can not save an image that has no filename. Save image as tif
        # separately, " \ "remember to remove crop (if any) " TODO : figure out what to do in this case

        d = {'resolution': self.resolution.tolist(),
             'filename': self.filename,
             'info': self.info,
             'mask': self.mask,
             'img': None}

        if self.filename is None:
            d['img'] = "Sorry I haven't figured out how this needs to work yet ... " \
                       "this image hasn't been saved"
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d['resolution'], filename=d['filename'], img=d['img'], info=d['info'], mask=d['mask'])

    def imwrite(self, filename):
        """
        Saves image to disc as tif.
        """
        # ImageJ hyperstack axes must be in TZCYXS order...
        # it ignores my 'axis' metadata (Is it called something else?).. so just expand to ZCYX
        tif.imwrite(filename, np.expand_dims(self.img, axis=1).astype(np.uint16), imagej=True)

    def warp(self, matrix, interpolator='nearestNeighbor', fixed_resolution=None, fixed_file=None, fixed=None):
        """
        Returns warped image: using matrix to transform from self to fixed.
        matrix : affine matrix with the transform, 4x4
        interpolator : any interpolator, excepted by ANTs: 'linear', 'nearestNeighbor' .. etc
        fixed_resolution : resolution of the fixed image
        fixed_file : fixed image file
        fixed : fixed image, Image
        """
        if fixed_resolution is None:
            fixed_resolution = self.resolution.tolist()
        if fixed_file is not None and fixed is None:
            fixed = Image(fixed_resolution, filename=fixed_file)

        # keep in mind, ants want the resolution in XYZ order
        fixed = ants.from_numpy(fixed.img.astype(float), spacing=fixed.resolution.tolist())
        moving = ants.from_numpy(self.img.astype(float), spacing=self.resolution.tolist())

        warpedimg = ants.apply_transforms(fixed=fixed, moving=moving,
                                          transformlist=[matrix], interpolator=interpolator)

        return Image(fixed.resolution.tolist(), img=warpedimg.numpy().astype(np.uint16))


class ImagePair:

    def __init__(self, fixed, moving):
        self.fixed = fixed
        self.moving = moving
        self.shape = {'fixed': self.fixed.img.shape, 'moving': self.moving.img.shape}
        self.alignment = None
        self.block_info = None

    @classmethod
    def from_dict(cls, spec):
        fixed = Image(spec['fixed']['resolution'],
                      filename=spec['fixed']['filename'],
                      img=spec['fixed']['img'],
                      info=spec['fixed']['info'],
                      mask=spec['fixed']['mask'])

        moving = Image(spec['moving']['resolution'],
                       filename=spec['moving']['filename'],
                       img=spec['moving']['img'],
                       info=spec['moving']['info'],
                       mask=spec['moving']['mask'])
        return cls(fixed, moving)


    def register(self, transform_folder, fixed_size=None, moving_size=None, fixed_overlap=0, moving_overlap=0,
                 verbose=False):
        """
        Register the image pair : moving to fixed.
        fixed_size, moving_size : sized of 3D Blocks used for regitration, in [Z, Y, X] order, in pixels.
        fixed_overlap, moving_overlap : overlap of 3D Blocks used for regitration, in [Z, Y, X] order, in pixels,
         or 0 for no overlap.
        """
        #TODO : change voxels to blocks
        voxels_f = self.fixed.split(fixed_size, fixed_overlap)
        voxels_m = self.moving.split(moving_size, moving_overlap)

        pairs = [BlockPair(voxf, voxm) for voxf, voxm in zip(voxels_f, voxels_m)]
        for pair in tqdm(pairs):
            pair.register(transform_folder, verbose=verbose)

        self.alignment = pairs
        self.block_info = {"fixed_size": fixed_size, "moving_size": moving_size,
                           "fixed_overlap": fixed_overlap, "moving_overlap": moving_overlap}

    def save(self, folder, padding=0, info=None):
        """
        Saves the fixed, moving and warped images.
        """

        os.makedirs(folder)

        # save info
        info = {'mask': {'fixed': self.fixed.mask, 'moving': self.moving.mask},
                'padding': padding, 'info': info}
        j = json.dumps(info)
        with open(f"{folder}/info.json", 'w') as json_file:
            json_file.write(j)

        # fixed image
        blc1 = [pair.blc1 for pair in self.alignment]
        bv1 = BlockView(blc1, padding)
        bv1.write_volume(f"{folder}/fixed.tif")

        # moving image
        blc2 = [pair.blc2 for pair in self.alignment]
        bv2 = BlockView(blc2, padding)
        bv2.write_volume(f"{folder}/moving.tif")

        # warped (registered) image
        bp = BlockPairView(self.alignment, padding)
        bp.write_volume(f"{folder}/warped.tif")

        # alignment for future use with the points
        transforms = [pair.alignment['affine'] for pair in self.alignment]
        to_json(transforms, f"{folder}/transform_list.json")

    # def to_dict(self):
    #     alignment_d = {'ants_file': self.alignment['ants_file'],
    #                    'affine': self.alignment['affine'].to_dict()}
    #     d = {'blc1': self.blc1.to_dict(),
    #          'blc2': self.blc2.to_dict(),
    #          'alignment': alignment_d}
    #     return d
    #
    # @classmethod
    # def from_dict(cls, d):
    #     blc1 = Block.from_dict(d['blc1'])
    #     blc2 = Block.from_dict(d['blc2'])
    #     alignment = {'ants_file': d['alignment']['ants_file'],
    #                  'affine': AffineTransform.from_dict(d['alignment']['affine'])}
    #     return cls(blc1, blc2, alignment=alignment)


class Block:
    """
    Individual block information. Defined by the top left corner and the size.
    """

    def __init__(self, img, start, size, overlap, idx, num_blc):
        # measurements in pixels
        # in ZYX order, in pixels
        self.start = np.array(start)
        # in ZYX order
        self.size = np.array(size)
        self.idx = np.array(idx)
        self.img = img

        # TODO : don't need the rest?
        self.num_blc = num_blc
        self.overlap = overlap

    def __str__(self):
        return f"start {self.start}\nsize {self.size}\nidx {self.idx}\noverlap {self.overlap}"

    def __repr__(self):
        return self.__str__()

    def crop(self):
        """
        Crop the area corresponding to block from the image.
        """
        z0, y0, x0 = self.start
        z1, y1, x1 = self.start + self.size
        volume = self.img.img[z0:z1, y0:y1, x0:x1]
        return volume

    def to_dict(self):
        d = {'img': self.img.to_dict(),
             'start': self.start.tolist(),
             'size': self.size.tolist(),
             'overlap': self.overlap.tolist(),
             'num_blc': self.num_blc.tolist(),
             'idx': self.idx.tolist()
             }
        return d

    @classmethod
    def from_dict(cls, d):
        img = Image.from_dict(d['img'])
        return cls(img, d['start'], d['size'], d['overlap'], d['idx'], d['num_blc'])


class BlockPair:
    def __init__(self, blc1, blc2, alignment=None):
        self.blc1 = blc1
        self.blc2 = blc2
        # from vox2 to vox1
        self.alignment = alignment
        self.warped = None
        # generate a random id for the blockpair ( need it for identifying the saved alignmnet later )
        # TODO : there should be a better way to do it...
        self.bp_id = random.randint(0, 10000)

    def register(self, transform_folder, keep_warped=False, verbose=False):
        """
        Registers the blocks ( block 2 (blc2) -to-> block 1 (blc1) ).
        verbose : prints information about the fixed and moving images
        """
        # keep in mind, ants want the resolution in XYZ order
        fixed = ants.from_numpy(self.blc1.crop().astype(float), spacing=self.blc1.img.resolution.tolist())
        moving = ants.from_numpy(self.blc2.crop().astype(float), spacing=self.blc2.img.resolution.tolist())
        if verbose:
            print(f'fixed : {fixed}')
            print(f'moving : {moving}')
        # run ants registration
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine',
                                syn_metric='CC')
        # save ants file just in case ... probably will et rid of it in the future
        copy_of_ants = f'{transform_folder}/{self.bp_id}_affine.mat'
        shutil.copyfile(reg['fwdtransforms'][0], copy_of_ants)
        self.alignment = {'ants_file': copy_of_ants}

        # save affine transformation and center
        # get center and size in physical units
        center = self.blc2.start * self.blc2.img.resolution
        size = self.blc2.size * self.blc2.img.resolution
        self.alignment['affine'] = AffineTransform(ants_to_affine(
            ants.read_transform(reg['fwdtransforms'][0], dimension=3)), center=center, size=size)

        # in case you want to keep the transformed image in memory ( interpolation is set to linear by default )
        if keep_warped:
            self.warped = reg['warpedmovout'].numpy().astype(np.uint16)

    def warp(self, interpolator='nearestNeighbor', img=None):
        """
        Returns warped image: from blc2 to blc1.
        interpolator : any interpolator, excepted by ANTs: 'linear', 'nearestNeighbor' .. etc
        img : can apply transform to image img, if None , applies to the moving image.
                Note that the img should already be properly cropped.
        """
        # keep in mind, ants want the resolution in XYZ order
        fixed = ants.from_numpy(self.blc1.crop().astype(float), spacing=self.blc1.img.resolution.tolist())
        if img is None:
            moving = ants.from_numpy(self.blc2.crop().astype(float), spacing=self.blc2.img.resolution.tolist())
        else:
            moving = ants.from_numpy(img.img.astype(float), spacing=img.resolution.tolist())
        warpedimg = ants.apply_transforms(fixed=fixed, moving=moving,
                                          transformlist=[self.alignment['ants_file']], interpolator=interpolator)

        return Image(self.blc1.img.resolution, img=warpedimg.numpy().astype(np.uint16))

    def to_dict(self):
        alignment_d = {'ants_file': self.alignment['ants_file'],
                       'affine': self.alignment['affine'].to_dict()}
        d = {'blc1': self.blc1.to_dict(),
             'blc2': self.blc2.to_dict(),
             'alignment': alignment_d}
        return d

    @classmethod
    def from_dict(cls, d):
        blc1 = Block.from_dict(d['blc1'])
        blc2 = Block.from_dict(d['blc2'])
        alignment = {'ants_file': d['alignment']['ants_file'],
                     'affine': AffineTransform.from_dict(d['alignment']['affine'])}
        return cls(blc1, blc2, alignment=alignment)


class AffineTransform:
    """
    A 3D affine transform that can be centered at a different coordinate than 0,0,0,
    and carries the information about the area onto which it should be applied (block size, relative to the center).
    """

    def __init__(self, matrix, center=None, size=None):

        self.matrix = np.array(matrix)

        if center is None:
            center = [0, 0, 0]
        self.center = np.array(center)

        # TODO : add fixed_start and fixed_size... maybe remove size ?
        self.size = None
        if size is not None:
            self.size = np.array(size)

    def __str__(self):
        string = f' matrix : \n{self.matrix}\ncenter : {self.center}\nsize : {self.size}'
        return string

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        d = {"matrix": self.matrix.tolist(),
             "center": self.center.tolist(),
             "size": None}

        if self.size is not None:
            d["size"] = self.size.tolist()

        return d

    @classmethod
    def from_dict(cls, d):
        """
        Load BlockTransform object from dict.
        """
        af_transform = cls(d['matrix'], center=d['center'], size=d['size'])
        return af_transform

    @classmethod
    def from_json(cls, filename):
        """
        Load AffineTransform object from json file.
        """
        # create an object for the class to return
        with open(filename) as json_file:
            j = json.load(json_file)

        return cls.from_dict(j)

    def to_json(self, filename):
        """
        Transform AffineTransform object into json format and save as a file.
        """
        j = json.dumps(self.to_dict())

        with open(filename, 'w') as json_file:
            json_file.write(j)


class Points:
    """ Points class represents and manipulates xyz coords. """

    def __init__(self, zyx_arr, units='pix', resolution=None, idx=None, info=None):
        """ Create a new point at the origin
        units : in what units the zyx_arr coordinates are given. Can be 'pix' or 'phs'
        for pixels or physical units respectively.
        info : dictionary with lists or numpy arrays, specifying property per point.
        """

        if resolution is None:
            resolution = [1, 1, 1]
        self.resolution = np.array(resolution)

        self.zyx = {}
        if units == 'pix':
            self.zyx['pix'] = np.array(zyx_arr)
            self.zyx['phs'] = self.zyx['pix'] * self.resolution
        elif units == 'phs':
            self.zyx['phs'] = np.array(zyx_arr)
            self.zyx['pix'] = np.round(self.zyx['phs'] / self.resolution)

            # personal id for each point
        self.num_points = self.zyx['pix'].shape[0]
        if idx is None:
            self.idx = np.arange(self.num_points)
        else:
            self.idx = np.array(idx)

        self.info = info

    def __repr__(self):
        return f'Number of points : {self.num_points}\nResolution : {self.resolution}\nCoordinates' \
               f' :\n- pixels\n{self.zyx["pix"]}\n- physical units\n{self.zyx["phs"]}'

    @classmethod
    def from_dict(cls, d):
        """
        Load Points object from a dictionary.
        TODO : maybe you want to save and load both pix and phs units ... using phs only for now
        """
        # info might be missing
        if 'info' in d:
            info = d['info']
        else:
            info = None

        points = cls(d['zyx'], units='phs', resolution=d['resolution'], idx=d['idx'], info=info)
        return points

    @classmethod
    def from_json(cls, filename):
        """
        Load Points object from json file.
        TODO : maybe you want to save and load both pix and phs units ... using phs only for now
        """
        # create an object for the class to return
        with open(filename) as json_file:
            j = json.load(json_file)
        points = cls.from_dict(j)
        return points

    @classmethod
    def from_predictions(cls, filename, prob_thr=0.5, resolution=[1, 1, 1], units='pix'):
        df = pd.read_csv(filename)
        points = cls(df[['Z', 'Y', 'X']][df["prob"] > prob_thr].to_numpy(),
                     units=units, resolution=resolution)
        return points

    def to_dict(self):
        """
        Transform Points object into json format and save as a file.
        """

        d = {"resolution": self.resolution.tolist(),
             "zyx": self.zyx['phs'].tolist(),
             "idx": self.idx.tolist()}

        if self.info is not None:
            d["info"] = {key: self.info[key].tolist() for key in self.info}

        return d

    def to_json(self, filename):
        """
        Transform Points object into json format and save as a file.
        """
        j = json.dumps(self.to_dict())

        with open(filename, 'w') as json_file:
            json_file.write(j)

    def crop(self, mask, units='pix'):
        """
        Crops a point cloud: drops everything outside a rectangle mask (in pixels or physical units)
        and remembers the parameters of the crop.
        mask : dict with xmin, xmax, ymin, ymax, zmin, zmax optional ( fields can be empty if don't need to crop there).

        """
        # calculate the crop
        is_in = np.ones(self.num_points, dtype=bool)

        for ikey, key in enumerate(['zmin', 'ymin', 'xmin']):
            if key in mask and mask[key] is not None:
                is_in = np.logical_and(is_in,
                                       mask[key] < self.zyx[units][:, ikey])
        for ikey, key in enumerate(['zmax', 'ymax', 'xmax']):
            if key in mask and mask[key] is not None:
                is_in = np.logical_and(is_in,
                                       self.zyx[units][:, ikey] < mask[key])
        # apply crop
        zyx = self.zyx[units][is_in, :]
        idx = self.idx[is_in]
        if self.info is not None:
            info = {key: np.array(self.info[key])[is_in] for key in self.info}

        points = Points(zyx, units=units, resolution=self.resolution, idx=idx, info=info)
        return points

    def recenter(self, center, units='pix'):
        """
        Sets the zero to center ( array of 3 elements in zyx order ).
        Center needs to be in pixels or the same physical units as the pointcloud.
        """
        center = np.array(center)
        zyx = self.zyx[units] - center

        points = Points(zyx, units=units, resolution=self.resolution, idx=self.idx)
        return points

    def transform(self, transform, units='phs'):
        """
        Applies transform to points in given units , default to physical.
        transform : AffineTransform, a matrix and a center representing an affine transform in 3D.
        In such format, that to apply transform matrix to a set of zyx1 points : zyx1@transform.matrix .

        Returns Points with the same type of dta as the original, but coordinates transformed.
        """

        def to_zyx1(zyx_arr):
            n_points = zyx_arr.shape[0]
            ones = np.ones(n_points)
            return np.c_[zyx_arr, ones[:, np.newaxis]]

        zyx = self.zyx[units] - transform.center
        zyx1 = to_zyx1(zyx)
        transformed_zyx1 = zyx1 @ transform.matrix
        transformed_zyx = transformed_zyx1[:, 0:3] + transform.center

        points = Points(transformed_zyx, units=units, resolution=self.resolution, idx=self.idx, info=self.info)
        return points

    def fit_block(self, blc, padding=[0, 0, 0]):
        """
        Takes a ptc and crops it to block.
        padding : in pixels (in the pixel space of the block)
        """
        # get mask in physical units :
        start = (blc.start - padding) * blc.img.resolution
        end = (blc.start + blc.size + padding) * blc.img.resolution
        mask = {'zmin': start[0], 'zmax': end[0],
                'ymin': start[1], 'ymax': end[1],
                'xmin': start[2], 'xmax': end[2]}

        return self.crop(mask, units='phs')

    def fit_transform(self, af, padding=[0, 0, 0]):
        """
        Takes a ptc and crops it to the area on which affine transform was calculated.
        padding : in physical units? in zyx order
        Assumes the transform is in physical units.
        """
        # TODO : maybe make transform carry the UNITS
        # TODO : make transform carry FIXED info ... now works ONLY because fixed and moving are the same

        # get mask in physical units :
        start = af.center - padding
        end = af.center + af.size + padding
        mask = {'zmin': start[0], 'zmax': end[0],
                'ymin': start[1], 'ymax': end[1],
                'xmin': start[2], 'xmax': end[2]}

        return self.crop(mask, units='phs')

    def split(self, blocks, padding=[0, 0, 0]):
        """ Splits points into Blocks
        Creates a points list in the order, that corresponds to the given blocks list.
        """
        points = []
        for block in blocks:
            points.append(self.fit_block(block, padding))
        return points

    @classmethod
    def concat(cls, ptc_list):
        """
        combines point clouds in ptc_list into one, concatenating the coordinates and idx.
        all point clouds need to have the same resolution.
        padding : zyx padding in pixels or phs
        """

        resolution = ptc_list[0].resolution
        zyx = None
        idx = None

        for i_ptc, ptc in enumerate(ptc_list):
            if i_ptc == 0:
                zyx = ptc.zyx['phs']
                idx = ptc.idx
            else:
                assert np.all(resolution == ptc.resolution), "Resolution should be the same for all point clouds"
                zyx = np.r_[zyx, ptc.zyx['phs']]
                idx = np.r_[idx, ptc.idx]

        # TODO : add info as well
        points = cls(zyx, units='phs', resolution=resolution, idx=idx)
        return points

    def pw_transform(self, transfom_list):
        # TODO : remake AffineTransforms to have info about the fixed as well as the moving
        """
        Piece-wise transforms the ptc according to each block alignment.
        Creates a points list in the order, that corresponds to the given transfom list.
        Each ptc in the list contains all the points, but transformed according to the different alignments.
        """
        points = []
        for af in transfom_list:
            # transform already takes the top left corner into account (center)
            ptc = self.transform(af, units='phs')
            points.append(ptc)

        return points

    def filter_by_info(self, feature, filter, units='phs'):
        """
        Applies threshold on the specified info.
        filter: 'max' and 'min' values to keep.
        feature: what info to use
        """
        assert self.info is not None, "Can't filter by info : info is None"

        is_in = np.ones((self.num_points,)).astype(np.bool)
        if 'max' in filter:
            is_in = np.logical_and(is_in, (np.array(self.info[feature]) <= filter['max']))
        if 'min' in filter:
            is_in = np.logical_and(is_in, (np.array(self.info[feature]) >= filter['min']))

        zyx = self.zyx[units][is_in, :]
        idx = self.idx[is_in]
        info = {key: np.array(self.info[key])[is_in] for key in self.info}

        return Points(zyx, units=units, resolution=self.resolution, idx=idx, info=info)

    def reset_idx(self):
        """
        Creates new set of idx.
        """
        self.idx = np.arange(self.num_points)
