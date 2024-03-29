import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from sortedcontainers import SortedList
from .utils import segments_intersect


class Fracture(object):
    """
    A data structure for representation of a 1D fracture mesh.
    """

    def __init__(self, endpoints, fid):
        """
        Parameters
        ------
        endpoints: numpy array
            An array containing the endpoints of the line segment
            representing the fracture.
        fid: int
            An integer ID for the fracture.
        """
        self.id = fid
        self.endpoints = endpoints
        self.nodes = SortedList(key=lambda arr: tuple(arr))

    def set_mesh_nodes(self, min_dist):
        """
        Set the discrete points of the mesh.

        Parameters
        ----------
        min_dist: float
            Minimal distance between two discrete points.
        """
        N = int(np.linalg.norm(
            self.endpoints[0] - self.endpoints[1]) / min_dist)
        convex_param_range = np.linspace(0, 1, N + 2).reshape((N + 2, 1))
        nodes_coords = (1 - convex_param_range) * \
            self.endpoints[0, :] + convex_param_range*self.endpoints[1, :]

        for node_coord in nodes_coords:
            self.nodes.add(node_coord)


class FractureGenerator2D(object):
    """
    A 2D fractures generator according to a D-1 model. Fractures are
    represented by line segments.
    """

    def __init__(self, num_fractures, bounding_box_dimensions, min_node_dist):
        """
        Parameters
        ------
        num_fractures: int
            Number of fractures to be generated.
        bounding_box_dimensions: tuple
            Dimensions of the bounding box across the x and y axis.
        min_node_dist: float
            The minimal distance between the fractures' nodes.
        """
        self.num_fractures = num_fractures
        self.bbox_dimensions = bounding_box_dimensions
        self.random_rng = np.random.default_rng()
        self.line_segments = np.zeros((self.num_fractures, 2, 2))
        self.fractures = None
        self.intersection_points = []
        self.intersecting_pairs = []
        self.min_node_dist = min_node_dist

    def generate_fractures(self):
        """
        Generates the fractures and sets the fractures attribute.
        """
        self.line_segments[:, :, 0] = self.random_rng.uniform(
            high=self.bbox_dimensions[0], low=0.0, size=(self.num_fractures, 2))
        self.line_segments[:, :, 1] = self.random_rng.uniform(
            high=self.bbox_dimensions[1], low=0.0, size=(self.num_fractures, 2))
        self.fractures = [Fracture(
            endpoints, fracture_id) for endpoints, fracture_id in zip(self.line_segments, range(self.num_fractures))]
        for fracture in self.fractures:
            fracture.set_mesh_nodes(self.min_node_dist)

    def find_intersections(self):
        """
        Check fractures for intersections and add points to the
        fracture mesh.
        """
        for f1 in self.fractures:
            for f2 in self.fractures:
                if f1.id != f2.id and \
                    (f1.id, f2.id) not in self.intersecting_pairs and \
                    (f2.id, f1.id) not in self.intersecting_pairs and \
                        segments_intersect(f1.endpoints, f2.endpoints):
                    # Vector directions for the segments.
                    u = f1.endpoints[1] - f1.endpoints[0]
                    v = f2.endpoints[1] - f2.endpoints[0]
                    w = f2.endpoints[0] - f1.endpoints[0]

                    # Parameters of the intersection.
                    s = np.cross(w, v) / np.cross(v, u)

                    # Intersection point.
                    I = f1.endpoints[0] - s*u

                    # Insert point into fracture's mesh.
                    f1.nodes.add(I)
                    f2.nodes.add(I)

                    self.intersection_points.append(I)
                    self.intersecting_pairs.append((f1.id, f2.id))

    def plot_fractures(self):
        for fracture in self.fractures:
            nodes_coords = np.array(list(fracture.nodes.irange()))
            plt.plot(fracture.endpoints[:, 0],
                     fracture.endpoints[:, 1], "ro-", linewidth=2)
            if len(nodes_coords) > 0:
                plt.plot(nodes_coords[:, 0], nodes_coords[:, 1], "ro")
        plt.show()

    def export_fractures_to_file(self, path):
        """
        Write the fractures matrix to a text file.
        """
        num_intersecting_pairs = len(self.intersecting_pairs)

        fracture_points_array = np.array(
            [np.array(list(f.nodes.irange())) for f in self.fractures], dtype=object)
        intersection_points_array = np.array(
            self.intersection_points).reshape((num_intersecting_pairs, 2))
        intersecting_pairs_array = np.array(
            self.intersecting_pairs).reshape((num_intersecting_pairs, 2)) + 1

        mdict = {"fractures": fracture_points_array, "intersections": intersection_points_array,
                 "intersecting_pairs": intersecting_pairs_array}
        savemat(path, mdict)
