import numpy as np
import tifffile as tif


class VoxelView:
    """
    Visualises voxels:
    creates 3D composite image with individual voxels separated by a 3D padding.
    """

    def __init__(self, voxels, padding):
        self.voxels = voxels
        self.padding = padding

    def zero_volume(self):
        """
        Creates zero volume of size that includes visualisation padding.
        :return: np.ndarray of zeroes
        """
        num_vox = self.voxels[0].num_vox
        view_size = self.voxels[0].size * num_vox + self.padding * (num_vox - 1)
        return np.zeros(view_size)

    def fill_volume(self):
        volume = self.zero_volume()

        for voxel in self.voxels:
            z0, y0, x0 = (voxel.size + self.padding) * voxel.idx
            z1, y1, x1 = (voxel.size + self.padding) * voxel.idx + voxel.size
            volume[z0:z1, y0:y1, x0:x1] = voxel.crop()

        return volume

    def write_volume(self, filename):
        volume = self.fill_volume()
        tif.imsave(filename, volume.astype(np.uint16), imagej=True)


class VoxelPairView:
    """
    Visualises voxel pairs:
    creates 3D composite image with individual voxels separated by a 3D padding.
    """

    def __init__(self, pairs, padding):
        self.pairs = pairs
        self.padding = padding

    def zero_volume(self):
        """
        Creates zero volume of size that includes visualisation padding. To match the fixed image.
        :return: np.ndarray of zeroes
        """
        num_vox = self.pairs[0].vox1.num_vox
        view_size = self.pairs[0].vox1.size * num_vox + self.padding * (num_vox - 1)
        return np.zeros(view_size)

    def fill_volume(self):
        volume = self.zero_volume()

        for pair in self.pairs:
            voxel = pair.vox1
            z0, y0, x0 = (voxel.size + self.padding) * voxel.idx
            z1, y1, x1 = (voxel.size + self.padding) * voxel.idx + voxel.size
            volume[z0:z1, y0:y1, x0:x1] = pair.wrapped

        return volume

    def write_volume(self, filename):
        volume = self.fill_volume()
        tif.imsave(filename, volume.astype(np.uint16), imagej=True)
