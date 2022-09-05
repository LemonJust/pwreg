import numpy as np
import tifffile as tif
from .core import *


class BlockView:
    """
    Visualises blocks:
    creates 3D composite image with individual blocks separated by a 3D padding.
    """

    def __init__(self, blocks, padding):
        self.blocks = blocks
        self.padding = padding

    def zero_volume(self):
        """
        Creates zero volume of size that includes visualisation padding.
        :return: np.ndarray of zeroes
        """
        num_blc = self.blocks[0].num_blc
        view_size = self.blocks[0].size * num_blc + self.padding * (num_blc - 1)
        return np.zeros(view_size)

    def fill_volume(self):
        volume = self.zero_volume()

        for block in self.blocks:
            z0, y0, x0 = (block.size + self.padding) * block.idx
            z1, y1, x1 = (block.size + self.padding) * block.idx + block.size
            volume[z0:z1, y0:y1, x0:x1] = block.crop()

        return volume

    def write_volume(self, filename):
        volume = self.fill_volume()

        # a trick to make it write color channels correctly:
        tif.imsave(filename, np.expand_dims(volume, axis=1).astype(np.uint16), imagej=True)


class BlockPairView:
    """
    Visualises block pairs:
    creates 3D composite image with individual blocks separated by a 3D padding.
    """

    def __init__(self, pairs, padding):
        self.pairs = pairs
        self.padding = padding

    def zero_volume(self):
        """
        Creates zero volume of size that includes visualisation padding. To match the fixed image.
        :return: np.ndarray of zeroes
        """
        num_blc = self.pairs[0].blc1.num_blc
        view_size = self.pairs[0].blc1.size * num_blc + self.padding * (num_blc - 1)
        return np.zeros(view_size)

    def fill_volume(self, interpolator):
        volume = self.zero_volume()

        for pair in self.pairs:
            block = pair.blc1
            z0, y0, x0 = (block.size + self.padding) * block.idx
            z1, y1, x1 = (block.size + self.padding) * block.idx + block.size

            volume[z0:z1, y0:y1, x0:x1] = pair.warp(interpolator=interpolator).img

        return volume

    def write_volume(self, filename, interpolator='nearestNeighbor'):

        volume = self.fill_volume(interpolator)

        # a trick to make it write color channels correctly:
        tif.imsave(filename, np.expand_dims(volume, axis=1).astype(np.uint16), imagej=True)


class PointsBlockView:
    """
    Visualises points in blocks:
    creates 3D composite point cloud with individual ptc blocks separated by a 3D padding.
    """

    def __init__(self, points, blocks, padding):
        self.points = points
        self.blocks = blocks
        self.padding = padding

    def fill_ptc(self):
        def place_at_block(ptc, blc):
            # crop only points inside the block
            ptc = ptc.fit_block(blc)
            # set the block's top left corner as 0
            ptc = ptc.recenter(blc.start, units='pix')
            # get new position of the top left corner, in pixels
            z0, y0, x0 = (blc.size + self.padding) * blc.idx
            center = np.array([z0, y0, x0])
            # create new ptc with zero at the new corner
            return Points(ptc.zyx['pix'] + center,
                          units='pix', resolution=[ptc.resolution], idx=ptc.idx)

        points = []
        for ptc, blc in zip(self.points, self.blocks):
            points.append(place_at_block(ptc, blc))

        return Points.concat(points)

    def to_json(self, filename):
        full_ptc = self.fill_ptc()
        full_ptc.to_json(filename)
