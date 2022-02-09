import numpy as np
import tifffile as tif
import ants
import warnings
# wsl:
import sys

project_path = '/mnt/d/Code/repos/gad1b-redistribution'
sys.path.insert(1, f'{project_path}/src')
from utils.utils import *


class Image:
    def __init__(self, resolution, filename=None, img=None, info=None):

        assert filename is not None or img is not None, "Provide filename or img."

        if filename is not None and img is None:
            self.filename = filename
            self.img = self.read_image()
        elif img is not None and filename is None:
            self.img = img
        self.resolution = resolution
        self.shape = self.img.shape
        self.info = info

    def read_image(self):
        """
        Reads image in ZYX order.
        """""
        img = tif.imread(self.filename)
        return img

    def split(self, vox_size, overlap):
        """
        Prepares a list of voxels for a specific padding and overlap.
        For now the voxels should split the image perfectly.
        :param vox_size: can be int or 3x1 numpy array representing 3 voxel sides
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

        vox_size = input_check(vox_size)
        overlap = input_check(overlap)
        img_size = np.array(self.shape)
        # get number of voxels along each dimension
        num_vox = (img_size - overlap) / (vox_size - overlap)
        if not np.all([n.is_integer() for n in num_vox]):
            warnings.warn(f"Specified voxel size + overlap don't cover the whole image."
                          f"Image size is {img_size}, voxel size {vox_size},"
                          f" overlap {overlap} results in {num_vox} number of voxels. Leaving some image. ")
        num_vox = num_vox.astype(int)

        voxels = []
        for nz in np.arange(num_vox[0]):
            # voxel start z pixel
            tlz = int((vox_size[0] - overlap[0]) * nz)
            for ny in np.arange(num_vox[1]):
                # voxel start y pixel
                tly = int((vox_size[1] - overlap[1]) * ny)
                for nx in np.arange(num_vox[2]):
                    # voxel start x pixel
                    tlx = int((vox_size[2] - overlap[2]) * nx)
                    voxels.append(Voxel(self, [tlz, tly, tlx], vox_size, overlap, [nz, ny, nx], num_vox))
        return voxels


class Voxel:
    """
    Individual voxel information.
    """

    def __init__(self, img, start, size, overlap, idx, num_vox):
        # measurements in pixels
        # in ZYX order
        self.start = start
        # in ZYX order
        self.size = size
        self.idx = idx
        self.num_vox = num_vox
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


class VoxelPair:
    def __init__(self, vox1, vox2):
        self.vox1 = vox1
        self.vox2 = vox2
        # from vox2 to vox1
        self.alignment = {}
        self.wrapped = None

    def register(self, keep_wrapped=True):
        """
        Registers the voxels, keeps the alignment mat.
        :return:
        """
        fixed = ants.from_numpy(self.vox1.crop().astype(float), spacing=self.vox1.img.resolution)
        moving = ants.from_numpy(self.vox2.crop().astype(float), spacing=self.vox2.img.resolution)
        # run ants registration
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine',
                                syn_metric='CC')
        self.alignment['ants'] = ants.read_transform(reg['fwdtransforms'][0], dimension=3)
        self.alignment['affine'] = ants_to_affine(self.alignment['ants'])
        if keep_wrapped:
            self.wrapped = reg['warpedmovout'].numpy().astype(np.uint16)


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
        if idx is None:
            self.idx = np.arrange(self.xyz['pix'].shape[0])
        else:
            self.idx = idx

    def transform(self, transform, units='phs'):
        """
        Applies transform to points in given units , default to physical.
        transform : a matrix representing an affine transform in 3D.
        In such format, that to apply transform matrix to a set of xyz1 points : xyz1@transform .

        Returns Points with the same type of dta as the original, but coordinates transformed.
        """

        def to_xyz1(xyz_arr):
            n_points = xyz_arr.shape[0]
            ones = np.ones((1, n_points))
            return np.r_[xyz_arr, ones]

        xyz = self.xyz[units]
        xyz1 = to_xyz1(xyz)
        transformed_xyz1 = xyz1 @ transform
        transformed_xyz = transformed_xyz1[:, 0:3]

        transformed_points = Points(transformed_xyz, units=units, resolution=self.resolution, idx=self.idx)
        return transformed_points

    def split(self, voxels):
        """ Splits points into Voxels
        Creates a points list in the order, that corresponds to the given voxel list.
        """
        points = []
        for voxel in voxels:
            x, start_x, end_x = self.xyz['pix'][:, 0], voxel.start[2], voxel.start[2] + voxel.size[2]
            y, start_y, end_y = self.xyz['pix'][:, 1], voxel.start[1], voxel.start[1] + voxel.size[1]
            z, start_z, end_z = self.xyz['pix'][:, 2], voxel.start[0], voxel.start[0] + voxel.size[0]

            in_x = start_x <= x <= end_x
            in_y = start_y <= y <= end_y
            in_z = start_z <= z <= end_z

            is_inside = np.logical_and(in_z, np.logical_and(in_x, in_y))
            points.append(Points(self.xyz['phs'][is_inside], units='phs',
                                 resolution=self.resolution, idx=self.idx[is_inside]))

        return points


class PointsPair:
    def __init__(self, ptc1, ptc2, alignment):
        self.ptc1 = ptc1
        self.ptc2 = ptc2
        self.alignment = alignment
        self.wrapped = None
        self.pairs = None

    def align(self):
        """
        Registers the point clouds.
        """
        self.wrapped = self.ptc2.transform(self.alignment)

    def pair(self):
        """

        """
        pass
