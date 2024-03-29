import numpy as np
from math import acos

def rotation_to_align(a, b):
    """
    Compute the rotation matrix of a into b using
    Rodrigues' formula, i.e., the rotation that will
    align a and b.

    Parameters
    ----------
    a: numpy.array
        A vector.
    b: numpy.array
        Another vector.
    """
    # Define an axis of rotation.
    normal = np.cross(a, b)
    n = normal / np.linalg.norm(normal)

    # Compute the cosine and sine of the rotation angle.
    d = np.linalg.norm(a) * np.linalg.norm(b)
    c = a.dot(b) / d
    s = (1 - c ** 2) ** 0.5

    # The identity.
    I = np.eye(3)

    # The cross-product matrix.
    K = np.array([
            0.0, -n[2], n[1],
            n[2], 0.0, -n[0],
            -n[1], n[0], 0.0]).reshape((3,3))

    if np.abs(1 + c) < 1e-8 or np.abs(1 - c) < 1e-8:
        return I
    
    R = I + s * K + (1 - c) * (K.dot(K))

    return R.T

def angle_between(u, v):
    """
    Compute the angle between two vectors

    Parameters
    ----------
    u: numpy.array
        A vector.
    v: numpy.array
        Another vector.
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    u_dot_v = u.dot(v)
    angle = acos(u_dot_v / (norm_u * norm_v))
    return angle


def segments_intersect(s1, s2):
    """
    Check if two line segments intersect.

    Parameters
    ----------
    s1: numpy.array
        A 2D line segment represented by its two endpoints.
    s2: numpy.array
        Another 2D line segment.
    """

    tol = 1e-8
    d1 = direction(s2[0], s2[1], s1[0])
    d2 = direction(s2[0], s2[1], s1[1])
    d3 = direction(s1[0], s1[1], s2[0])
    d4 = direction(s1[0], s1[1], s2[1])

    segments_intersect = False

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        segments_intersect = True
    elif np.abs(d1) < tol and point_on_segment(s2, s1[0]):
        segments_intersect = True
    elif np.abs(d2) < tol and point_on_segment(s2, s1[1]):
        segments_intersect = True
    elif np.abs(d3) < tol and point_on_segment(s1, s2[0]):
        segments_intersect = True
    elif np.abs(d4) < tol and point_on_segment(s1, s2[1]):
        segments_intersect = True

    return segments_intersect


def direction(pi, pj, pk):
    """
    Find if a point is clokwise or counterclockwise to a line segment.

    Parameters
    ----------
    pi: numpy.array
        An endpoint of the line segment.
    pj: numpy.array
        Another endpoint of the line segment.
    pk: numpy.array
        The point to be checked for direction.
    """
    return np.cross(pk - pi, pj - pi)


def point_on_segment(s, p):
    """
    Check if a point lies on a segment.

    Parameters
    ----------
    s: numpy.array
        A 2D line segment represented by its two endpoints.
    p: numpy.array
        The point that will be checked if in segment.
    """
    is_on_segment = (p[0] >= s[:, 0].min()) & (p[0] <= s[:, 0].max()) & (
        p[1] >= s[:, 1].min()) & (p[1] <= s[:, 1].max())
    return is_on_segment
