import json
import numpy as np
import random
import tifffile as tif
import warnings
import shutil
# on wsl:
import sys

project_path = '/mnt/d/Code/repos/pwreg'
sys.path.insert(1, f'{project_path}/pwreg')
from utils.utils import *

try:
    import ants
except:
    print("Exception occured when trying to import ants !\n "
          "BlockPair.register method won't work and will cause an error if you use it. ")


class Image:
    def __init__(self, resolution, filename=None, img=None, info=None, mask=None):
        """
        mask : dict with xmin, xmax, ymin, ymax, zmin, zmax optional ( fields can be empty if don't need to crop there).
        """

        assert filename is not None or img is not None, "Provide filename or img."

        if filename is not None and img is None:
            self.filename = filename
            self.img = self.read_image()
        elif img is not None and filename is None:
            self.img = img

        self.resolution = np.array(resolution)

        self.mask = None
        if mask is not None:
            self.mask = self.crop(mask)

        self.shape = self.img.shape
        self.info = info

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

        self.img = self.img[mask['zmin']:mask['zmax'],
                   mask['ymin']:mask['ymax'],
                   mask['xmin']:mask['xmax']]
        return mask

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


class Block:
    """
    Individual block information.
    """

    def __init__(self, img, start, size, overlap, idx, num_blc):
        # measurements in pixels
        # in ZYX order, in pixels
        self.start = np.array(start)
        # in ZYX order
        self.size = size
        self.idx = idx
        self.num_blc = num_blc
        self.overlap = overlap
        self.img = img

    def __str__(self):
        return f"start {self.start}\nsize {self.size}\nidx {self.idx}\noverlap {self.overlap}"

    def __repr__(self):
        return self.__str__()

    def crop(self):
        z0, y0, x0 = self.start
        z1, y1, x1 = self.start + self.size
        volume = self.img.img[z0:z1, y0:y1, x0:x1]
        return volume


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

    def register(self, keep_warped=False):
        """
        Registers the voxels, keeps the alignment mat.
        :return:
        """
        fixed = ants.from_numpy(self.blc1.crop().astype(float), spacing=self.blc1.img.resolution.tolist())
        moving = ants.from_numpy(self.blc2.crop().astype(float), spacing=self.blc2.img.resolution.tolist())
        # run ants registration
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine',
                                syn_metric='CC')

        copy_of_ants = f'{project_path}/tmp/{self.bp_id}_affine.mat'
        shutil.copyfile(reg['fwdtransforms'][0], copy_of_ants)
        # TODO : probably don't need to carry 'ants' at all ....
        self.alignment = {'ants_file': copy_of_ants,
                          'ants': ants.read_transform(reg['fwdtransforms'][0], dimension=3)}
        center = self.blc2.start * self.blc2.img.resolution
        self.alignment['affine'] = AffineTransform(ants_to_affine(self.alignment['ants']), center=center)
        if keep_warped:
            self.warped = reg['warpedmovout'].numpy().astype(np.uint16)

    def warp(self, interpolator='nearestNeighbor', keep_warped=False):
        fixed = ants.from_numpy(self.blc1.crop().astype(float), spacing=self.blc1.img.resolution.tolist())
        moving = ants.from_numpy(self.blc2.crop().astype(float), spacing=self.blc2.img.resolution.tolist())
        warpedimg = ants.apply_transforms(fixed=fixed, moving=moving,
                                          transformlist=[self.alignment['ants_file']], interpolator=interpolator)
        warpedimg = warpedimg.numpy().astype(np.uint16)
        if keep_warped:
            self.warped = warpedimg
        return warpedimg


class AffineTransform:
    """
    A 3D affine transform that can be centered at a different coordinate than 0,0,0.
    Needed to keep track of the transforms obtained for individual blocks.
    """

    def __init__(self, matrix, center=None):
        self.matrix = np.array(matrix)
        if center is None:
            self.center = np.array([0, 0, 0])
        else:
            self.center = np.array(center)

    def __str__(self):
        string = f' matrix : \n{self.matrix}\ncenter : {self.center}'
        return string

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_json(cls, filename):
        """
        Load AffineTransform object from json file.
        """
        # create an object for the class to return
        with open(filename) as json_file:
            j = json.loads(json_file)
        af_transform = cls(j['matrix'], center=j['center'])
        return af_transform

    def to_json(self, filename):
        """
        Transform AffineTransform object into json format and save as a file.
        """
        j = json.dumps({"matrix": self.matrix.tolist(),
                        "center": self.center.tolist()})

        with open(filename, 'w') as json_file:
            json_file.write(j)


class Points:
    """ Points class represents and manipulates xyz coords. """

    def __init__(self, xyz_arr, units='pix', resolution=None, idx=None):
        """ Create a new point at the origin
        units : in what units the xyz_arr coordinates are given. Can be 'pix' or 'phs'
        for pixels or physical units respectively.
        """

        if resolution is None:
            resolution = [1, 1, 1]
        self.resolution = np.array(resolution)

        self.xyz = {}
        if units == 'pix':
            self.xyz['pix'] = np.array(xyz_arr)
            self.xyz['phs'] = self.xyz['pix'] * self.resolution
        elif units == 'phs':
            self.xyz['phs'] = np.array(xyz_arr)
            self.xyz['pix'] = np.round(self.xyz['phs'] / self.resolution)

            # personal id for each point
        self.num_points = self.xyz['pix'].shape[0]
        if idx is None:
            self.idx = np.arange(self.num_points)
        else:
            self.idx = idx

    def __repr__(self):
        return f'Number of points : {self.num_points}\nResolution : {self.resolution}\nCoordinates' \
               f' :\n- pixels\n{self.xyz["pix"]}\n- physical units\n{self.xyz["phs"]}'

    @classmethod
    def from_json(cls, filename):
        """
        Load Points object from json file.
        TODO : maybe you want to save and load both pix and phs units ... using phs only for now
        """
        # create an object for the class to return
        with open(filename) as json_file:
            j = json.loads(json_file)
        points = cls(j['xyz'], units='phs', resolution=j['resolution'], idx=j['idx'])
        return points

    def to_json(self, filename):
        """
        Transform Points object into json format and save as a file.
        """
        j = json.dumps({"resolution": self.resolution.tolist(),
                        "xyz": self.xyz['phs'].tolist(),
                        "idx": self.idx.tolist()})

        with open(filename, 'w') as json_file:
            json_file.write(j)

    def split(self, blocks):
        """ Splits points into Blocks
        Creates a points list in the order, that corresponds to the given blocks list.
        """
        points = []
        for block in blocks:
            x, start_x, end_x = self.xyz['pix'][:, 0], block.start[2], block.start[2] + block.size[2]
            y, start_y, end_y = self.xyz['pix'][:, 1], block.start[1], block.start[1] + block.size[1]
            z, start_z, end_z = self.xyz['pix'][:, 2], block.start[0], block.start[0] + block.size[0]

            in_x = start_x <= x <= end_x
            in_y = start_y <= y <= end_y
            in_z = start_z <= z <= end_z

            is_inside = np.logical_and(in_z, np.logical_and(in_x, in_y))
            points.append(Points(self.xyz['phs'][is_inside], units='phs',
                                 resolution=self.resolution, idx=self.idx[is_inside]))

        return points

    def crop(self, mask, units='pix'):
        """
        Crops a pointcloud: drops everything outside a rectangle mask (in pixels or physical units)
        and remembers the parameters of the crop.
        mask : dict with xmin, xmax, ymin, ymax, zmin, zmax optional ( fields can be empty if don't need to crop there).

        """
        # calculate the crop
        is_in = np.ones(self.num_points, dtype=bool)

        for ikey, key in enumerate(['xmin', 'ymin', 'zmin']):
            if key in mask and key is not None:
                is_in = np.logical_and(is_in,
                                       mask[key] < self.xyz[units][:, ikey])
        for ikey, key in enumerate(['xmax', 'ymax', 'zmax']):
            if key in mask and key is not None:
                is_in = np.logical_and(is_in,
                                       self.xyz[units][:, ikey] < mask[key])
        # apply crop
        xyz = self.xyz[units][is_in, :]
        idx = self.idx[is_in]

        points = Points(xyz, units=units, resolution=self.resolution, idx=idx)
        return points

    def recenter(self, center, units='pix'):
        """
        Sets the zero to center ( array of 3 elements in xyz order ).
        Center needs to be in pixels or the same physical units as the pointcloud.
        """
        center - np.array(center)
        xyz = self.xyz[units] - center

        points = Points(xyz, units=units, resolution=self.resolution, idx=self.idx)
        return points

    def transform(self, transform, units='phs'):
        """
        Applies transform to points in given units , default to physical.
        transform : AffineTransform, a matrix and a center representing an affine transform in 3D.
        In such format, that to apply transform matrix to a set of xyz1 points : xyz1@transform.matrix .

        Returns Points with the same type of dta as the original, but coordinates transformed.
        """

        def to_xyz1(xyz_arr):
            n_points = xyz_arr.shape[0]
            ones = np.ones(n_points)
            return np.c_[xyz_arr, ones[:, np.newaxis]]

        xyz = self.xyz[units] - transform.center
        xyz1 = to_xyz1(xyz)
        transformed_xyz1 = xyz1 @ transform.matrix
        transformed_xyz = transformed_xyz1[:, 0:3] + transform.center

        points = Points(transformed_xyz, units=units, resolution=self.resolution, idx=self.idx)
        return points


class ElementaryPointsPair:
    """
    Takes care of aligning pontclouds when there is only one alignmnet.
    """

    def __init__(self, ptc1, ptc2, alignment):
        """
        alignment : AffineTransform to align ptc2 to ptc1
        """
        self.ptc1 = ptc1
        self.ptc2 = ptc2
        self.alignment = alignment
        self.wrapped = None
        # linking tp2 to tp1 (first column tp2, second tp1)
        self.pairs = None

    def align(self):
        """
        Transform the point cloud at tp2 to align with the point cloud at tp1.
        """
        self.wrapped = self.ptc2.transform(self.alignment)

    def pair(self, max_radius, units='phs'):
        """
        Pairs the nearest points from tp2 and tp1 that are closer than given max_radius.
        the result is self.pairs, a dict with key : point ID from ptc2 , value : point ID from ptc1
        """

        if self.wrapped is None:
            self.align()

        _, pt2_to_pt1 = nearest_pairs(self.ptc1.xyz[units], self.wrapped.xyz[units], max_radius)

        idx2 = self.wrapped.idx
        idx1 = [self.ptc1.idx[pt1] if pt1 > -1 else -1 for pt1 in ptc2_to_ptc1]
        self.pairs = np.c_[idx2, idx1]


class PointsPair:
    """
    Takes care of aligning pontclouds when they are registered using splitting into blocks.
    """

    def __init__(self, ptc1, ptc2, blockpairs):
        """
        ptc1 : Points , pointcloud at tp1
        ptc2 : Points , pointcloud at tp2
        blockpairs : block pairs used to split the point clouds
        """

        self.blockpairs = blockpairs
        self.ptc1 = ptc1
        self.ptc2 = ptc2
        # linking tp2 to tp1 (first column tp2 ID, next N columns tp1 ID based on different blocks)
        self.pairs = None

    def zero_pairs(self):
        """
        Initialises a pairing matrix filled with Nans.
        """
        n_idx2 = len(self.ptc2.idx)
        n_blockpairs = len(self.blockpairs)
        matrix = np.empty((n_idx2, n_blockpairs))
        matrix.fill(np.nan)
        return matrix

    def pair(self):
        """
        Registers ptc2 to ptc1 using the blocks.
        """
        # split ptc2 into blocks
        blc2 = [blockpair.blc2 for blockpair in self.blockpairs]
        split_ptc2 = ptc2.split(blc2)
        # construct elementary pairs with ptc1 ( full ) and the split ptc2
        elm_pairs = [ElementaryPointsPair(self.ptc1, ptc, blockpair.alignmnet)
                     for ptc, blockpair in zip(split_ptc2, self.blockpairs)]

        # construct the pair matrix
        # TODO: maybe mak it sparse ???
        self.pairs = zero_pairs()

        # pair the elementary point clouds
        # and fill the pair matrix
        for i_block, elm_pair in enumerate(elm_pairs):
            elm_pair.pair()
            for pair in elm_pair.pairs:
                idx1 = pair[1]
                idx2 = pair[0]
                self.pairs[idx2, i_block] = idx1

    def pairs_summary(self):
        """
        Checks if all the blocks produced the same pairing.
        Also returns info on how ofter each synapse was placed into a block.
        """
        # TODO : unfinished
        pairs, votes = np.unique(self.pairs, return_counts=True, axis=1)
        num_candidates = [len(candidates) for candidates in pairs]
