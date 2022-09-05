import numpy as np
import json
import scipy.ndimage as ndi

# import scipy.io as si


# def mat_to_affine(filename):
#     """
#     Takes a path to the affine transform *.mat file written with itk::TransformFileWriter
#     and returns an affine matrix.
#     To apply this matrix to a set of xyz1 points : xyz1@transform .
#     """
#     # load and extract data
#     trans = si.loadmat(filename)
#     names = list(trans.keys())
#     A = trans[names[0]]
#     m_center = trans[names[1]]
#
#     return params_to_affine(A, m_center)


def ants_to_affine(antstransform):
    """
    Takes ANTsTransform and returns an affine matrix.
    To apply this matrix to a set of xyz1 points : xyz1@transform .
    """
    # extract data
    A = antstransform.parameters
    m_center = antstransform.fixed_parameters

    return params_to_affine(A, m_center)


def params_to_affine(A, m_center):
    """
    Takes parameters (A) and fixed_parameters (m_center) of ants transform and returns an affine matrix.
    To apply this matrix to a set of xyz1 points : xyz1@transform .
    """
    matrix = np.reshape(A[0:9], (3, 3))
    m_translation = A[9:12]
    # compute offset
    offset = m_translation + m_center - matrix @ m_center
    # compose matrix
    M = np.eye(4)
    M[0:3, 0:3] = matrix
    M[0:3, 3] = offset.T
    # M[3,3] = 1
    Minv = np.linalg.inv(M)
    transform = Minv.T
    return transform


def nearest_pairs(v1, v2, radius):
    """
    Code modified from https://github.com/informatics-isi-edu/synspy > analyze > pair
    see https://github.com/informatics-isi-edu/synspy/blob/master/LICENSE for more info.

    Find nearest k-dimensional point pairs between v1 and v2 and return via output arrays.

       Inputs:
         v1: array with first pointcloud with shape (n, k)
         kdt1: must be cKDTree(v1) for correct function
         v2: array with second pointcloud with shape (m, k)
         radius: maximum euclidean distance between points in a pair
         out1: output adjacency matrix of shape (n,)
         out2: output adjacency matrix of shape (m,)

       Use greedy algorithm to assign nearest neighbors without
       duplication of any point in more than one pair.

       Outputs:
         out1: for each point in kdt1, gives index of paired point from v2 or -1
         out2: for each point in v2, gives index of paired point from v1 or -1

    """
    kdt1 = cKDTree(v1)
    out1 = np.zeros((ve1.shape[0]), dtype=np.int32)
    out2 = np.zeros((ve2.shape[0]), dtype=np.int32)

    depth = min(max(out1.shape[0], out2.shape[0]), 100)
    out1[:] = -1
    out2[:] = -1
    dx, pairs = kdt1.query(v2, depth, distance_upper_bound=radius)
    for d in range(depth):
        for idx2 in np.argsort(dx[:, d]):
            if dx[idx2, d] < radius:
                if out2[idx2] == -1 and out1[pairs[idx2, d]] == -1:
                    out2[idx2] = pairs[idx2, d]
                    out1[pairs[idx2, d]] = idx2

    return out1, out2


def local_maxima_3D(data, order=1):
    """Detects local maxima in a 3D array
    From : https://stackoverflow.com/questions/55453110/how-to-find-local-maxima-of-3d-array-in-python

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


# SAVING AS JSON :
# TODO : Write custom JSONEncoder to make class JSON serializable (???)

def to_json(objs, filename):
    """
    Writes an object, or list of objects as json file.
    The objects should have method to_dict()
    """
    if isinstance(objs, list):
        j = json.dumps([obj.to_dict() for obj in objs])
    else:
        j = json.dumps(objs.to_dict())

    with open(filename, 'w') as json_file:
        json_file.write(j)


def from_json(cls, filename):
    """
    Loads an object, or list of objects of class cls from json file.
    The objects should have method from_dict()
    """
    with open(filename) as json_file:
        j = json.load(json_file)

    if isinstance(j, list):
        objs = [cls.from_dict(d) for d in j]
    else:
        objs = cls.from_dict(j)

    return objs
