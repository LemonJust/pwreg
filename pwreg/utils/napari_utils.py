import napari.layers as nl
import napari
import skimage.data
import skimage.filters
from napari.types import PointsData
import pandas as pd
import numpy as np
import os
import tifffile as tif
import json
import csv

from magicgui import magicgui

import datetime
# from enum import Enum
from pathlib import Path


def xyzum2napari(centroids, resolution):
    """
  Translates the coordinates in xyz order and in um into zyx order and in pixels, as needed for napari.
  resolution in ZYX order.
  """
    return np.round(centroids[:, [2, 1, 0]] / resolution).astype(int)


# modify Points class
class FixedPoints(nl.Points):
    """
    Modifies napari class Points.
    Points can no longer be moved with a mouse.
    """

    def _move(self):
        """Points are not allowed to move."""
        pass


class ImagePointsView:
    def __init__(self, img_dict, ptc_dict, resolution):
        """
        img_dict is a dictionary with core.Image class.
        ptc_dict is a dictionary with is a core.Points class...
        """
        self.imgs = img_dict
        self.ptcs = ptc_dict
        self.resolution = resolution

    def view_in_napari(self, img_cm, ptc_cm):
        """
        Display image with the corresponding point cloud.
        img_cm, ptc_cm : colormap names (str) and colors (str) for display
        """
        # TODO : generate colors if not provided ? maybe with napari.utils.colormaps.label_colormap()

        with napari.gui_qt():
            viewer = napari.Viewer()

            i_ptc = 0
            for name, ptc in self.ptcs.items():
                viewer.add_layer(FixedPoints(
                    ptc.zyx['pix'],
                    # or this way can use downsampled image if desired ...
                    # np.round(ptc.zyx['phs'] / self.resolution)
                    ndim=3,
                    size=2,
                    edge_width=1,
                    scale=self.resolution,
                    name=name,
                    face_color=ptc_cm[i_ptc]))
                i_ptc = i_ptc + 1

            i_img = 0
            for name, image in self.imgs.items():
                viewer.add_image(image.img,
                                 scale=self.resolution,
                                 name=name,
                                 colormap=img_cm[i_img],
                                 blending='additive')
                i_img = i_img + 1
