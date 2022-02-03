import numpy as np
import tifffile as tif
import ants
import warnings


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
        self.start = start
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
        self.alignment = None
        self.wrapped = None

    def register(self):
        """
        Registers the voxels, keeps the alignment mat.
        :return:
        """
        fixed = ants.from_numpy(self.vox1.crop().astype(float), spacing=self.vox1.img.resolution)
        moving = ants.from_numpy(self.vox2.crop().astype(float), spacing=self.vox2.img.resolution)
        # run ants registration
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine',
                                syn_metric='CC')
        self.alignment = ants.read_transform(reg['fwdtransforms'][0], dimension=3)
        self.wrapped = reg['warpedmovout'].numpy().astype(np.uint16)
