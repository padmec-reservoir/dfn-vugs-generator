import numpy as np


# ---------------------------
# DATA STRUCTURES FOR THE LINE SWEEP ALGORITHM
# ---------------------------
class LineSegmentTreeNode(object):
    def __init__(self, line_id, endpoints, parent=None, left_child=None, right_child=None):
        self.line_id = line_id
        self.endpoints = endpoints
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        return "ID: {}\nEndpoints: {}".format(self.line_id, self.endpoints)


class LineSweepStatusTree(object):
    def __init__(self):
        self.root = None

    def insert(self, new_node):
        y = None
        x = self.root
        leftest_endpoint_index = new_node.endpoints[:, 0].argmin()
        leftest_endpoint = new_node.endpoints[leftest_endpoint_index]

        while x is not None:
            y = x
            if direction(x.endpoints[0], x.endpoints[1], leftest_endpoint) > 0:
                x = x.left_child
            else:
                x = x.right_child

        new_node.parent = y
        if y is None:
            self.root = new_node
        elif direction(y.endpoints[0], y.endpoints[1], leftest_endpoint) > 0:
            y.left_child = new_node
        else:
            y.right_child = new_node

    def remove(self, node):
        if node.left_child is None:
            self.transplant(node, node.right_child)
        elif node.right_child is None:
            self.transplant(node, node.left_child)
        else:
            y = self.minimum(node.right_child)
            if y.parent.line_id != node.line_id:
                self.transplant(y, y.right_child)
                y.right_child = node.right_child
                y.right_child.parent = y
            self.transplant(node, y)
            y.left_child = node.left_child
            y.left_child.parent = y

    def transplant(self, u, v):
        # u -> sub-árvore a ser substituída
        # v -> sub-árvore substituta
        if u.parent is None:
            self.root = v
        elif u.line_id == u.parent.left_child:
            u.parent.left_child = v
        else:
            u.parent.right_child = v

        if v is not None:
            v.parent = u.parent

    def minimum(self, node):
        while node.left_child is not None:
            node = node.left_child
        return node

    def maximum(self, node):
        while node.right_child is not None:
            node = node.right_child
        return node

    def find(self, node):
        curr_node = self.root
        leftest_endpoint_index = node.endpoints[:, 0].argmin()
        leftest_endpoint = node.endpoints[leftest_endpoint_index]

        while curr_node is not None and node.line_id != curr_node.line_id:
            if direction(curr_node.endpoints[0], curr_node.endpoints[1], leftest_endpoint) > 0:
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child

        return curr_node

    def successor(self, node):
        if node.right_child is not None:
            return self.minimum(node.right_child)
        
        y = node.parent
        while y is not None and node.line_id == y.right_child.line_id:
            node = y
            y = y.parent
        
        return y
    
# ---------------------------
# GEOMETRY ROUTINES
# ---------------------------


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
