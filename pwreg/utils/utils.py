import numpy as np
import scipy.io as si


def mat_to_affine(filename):
    """
    Takes a path to the affine transform mat file written with itk::TransformFileWriter
    and returns an affine matrix.
    To apply this matrix to a set of xyz1 points : xyz1@transform .
    (This is identical to the ants2affine.m matlab function by Anna N.)
    """
    # load and extract data
    trans = si.loadmat(filename)
    names = list(trans.keys())
    A = trans[names[0]]
    m_center = trans[names[1]]

    return params_to_affine(A, m_center)


def ants_to_affine(antstransform):
    """
    Takes ANTsTransform and returns an affine matrix.
    To apply this matrix to a set of xyz1 points : xyz1@transform .
    (This is identical to the ants2affine.m matlab function by Anna N.)
    """
    # extract data
    A = antstransform.parameters
    m_center = antstransform.fixed_parameters

    return params_to_affine(A, m_center)


def params_to_affine(A, m_center):
    """
    Takes parameters (A) and fixed_parameters (m_center) of ants transform and returns an affine matrix.
    To apply this matrix to a set of xyz1 points : xyz1@transform .
    (This is identical to the ants2affine.m matlab function by Anna N.)
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
