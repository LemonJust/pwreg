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
    def __init__(self, img, ptc):
        self.img = img
        self.ptc = ptc

    def view_in_napari(self):
        """
        Display image with the corresponding point cloud.
        """
        resolution = self.img.resolution
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(self.img.img,
                             scale=resolution,
                             name='image',
                             colormap='grey',
                             blending='additive')

            viewer.add_layer(FixedPoints(
                self.ptc.xyz['pix'],
                ndim=3,
                size=2,
                edge_width=1,
                scale=resolution,
                name='points',
                face_color='cyan'))
