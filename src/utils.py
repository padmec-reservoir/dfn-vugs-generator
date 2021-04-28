import numpy as np


def segments_intersect(s1, s2):
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
    return np.cross(pk - pi, pj - pi)


def point_on_segment(s, p):
    is_on_segment = (p[0] >= s[:, 0].min()) & (p[0] <= s[:, 0].max()) & (
        p[1] >= s[:, 1].min()) & (p[1] <= s[:, 1].max())
    return is_on_segment
