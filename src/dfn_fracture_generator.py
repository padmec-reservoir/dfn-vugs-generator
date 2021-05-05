import numpy as np
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
from .utils import segments_intersect, LineSweepStatusTree, LineSegmentTreeNode


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
        self.nodes = SortedList(key=lambda arr: tuple(arr))

        if (endpoints[0, 1] > endpoints[1, 1]) or \
                (endpoints[0, 1] == endpoints[1, 1] and endpoints[0, 0] <= endpoints[1, 0]):
            endpoints[[0, 1]] = endpoints[[1, 0]]
            self.endpoints = endpoints
        else:
            self.endpoints = endpoints

    def set_mesh_nodes(self, min_dist):
        """
        Set the discrete points of the mesh.

        Parameters
        ----------
        min_dist: float
            Minimal distance between two discrete points.
        """
        N = int(np.linalg.norm(
            self.endpoints[0, :] - self.endpoints[1, :]) / min_dist)
        convex_param_range = np.linspace(0, 1, N + 2).reshape((N + 2, 1))
        nodes_coords = (1 - convex_param_range) * \
            self.endpoints[0, :] + convex_param_range * self.endpoints[1, :]

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
        flattened_endpoints = self.line_segments.reshape(
            (self.num_fractures * 2, 2))
        event_queue = SortedList(key=lambda event_point: tuple(event_point))
        # Segments by event point (L for lower endpoint, U for upper, and C for containing).
        L, U, C = {}, {}, {}
        for fracture in self.fractures:
            upper_endpoint, lower_endpoint = fracture.endpoints[0], fracture.endpoints[1]

            event_queue.add(upper_endpoint)
            if tuple(upper_endpoint) in U:
                U[tuple(upper_endpoint)] = [fracture.id]
            else:
                U[tuple(upper_endpoint)].append(fracture.id)

            event_queue.add(lower_endpoint)
            if tuple(lower_endpoint) in U:
                L[tuple(lower_endpoint)] = [fracture.id]
            else:
                L[tuple(lower_endpoint)].append(fracture.id)

            C[tuple(upper_endpoint)] = []
            C[tuple(lower_endpoint)] = []

        status = LineSweepStatusTree()
        intersections = []

        while len(event_queue) > 0:
            p = event_queue.pop(-1)
            L_p, U_p, C_p = L[tuple(p)], U[tuple(p)], C[tuple(p)]
            all_containing_p = set(L_p) | set(U_p) | set(C_p)

            # If point is in more than one segment, then it is an intersection.
            if len(all_containing_p) > 1:
                intersections.append((p, all_containing_p))

            # We then remove all segments from the status tree that have
            # p as a lower endpoint or inside it.
            segments_to_remove = set(L_p) | set(C_p)
            for line in segments_to_remove:
                node = status.find(LineSegmentTreeNode(
                    line, self.fractures[line].endpoints))
                if node is not None:
                    status.remove(node)

            lines_to_reinsert = set(U_p) | set(C_p)
            for line in lines_to_reinsert:
                status.insert(LineSegmentTreeNode(
                    line, self.fractures[line].endpoints))

            if len(lines_to_reinsert) == 0:
                pass
            else:
                pass

    def _find_new_event_point(self):
        pass

    def plot_fractures(self):
        for fracture in self.fractures:
            nodes_coords = np.array(list(fracture.nodes.irange()))
            if len(nodes_coords) > 0:
                plt.plot(nodes_coords[:, 0],
                         nodes_coords[:, 1], "ro-", linewidth=3)
            else:
                plt.plot(
                    fracture.endpoints[:, 0], fracture.endpoints[:, 1], "ro-", linewidth=3)
        plt.show()

    def export_fractures_to_file(self, path):
        """
        Write the fractures matrix to a text file.
        """
        reshaped_fractures = self.line_segments.reshape(
            (self.num_fractures, 4))
        np.savetxt(path, reshaped_fractures)
