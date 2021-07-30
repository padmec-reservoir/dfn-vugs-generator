from typing import Iterable
import numpy as np
from scipy.special import comb
from pymoab import rng
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
from .utils import rotation_to_align


class DFNMeshGenerator(object):
    """
    Base class for the mesh generator.
    """

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError()

    def compute_vugs(self):
        raise NotImplementedError()

    def compute_fractures(self):
        raise NotImplementedError()


class DFNMeshGenerator2D(DFNMeshGenerator):
    """
    A 2D mesh generator for fracured and vuggy reservoirs.
    """

    def __init__(self, mesh_file, ellipsis_params_range, num_ellipsis, num_fractures):
        """
        Parameters
        ------
        mesh_file : str
            A string containing the path to the input mesh file.
        ellipsis_params_range : iterable
            An iterable containing the maximum and the minimum value for the ellipsis parameters.
        num_ellipsis : int
            The number of ellipsis to create.
        num_fractures : int
            Number of fractures to create.

        Raises
        ------
        ValueError
            If the number of fractures is greater than the number of possible
            pairs of ellipsis.
        """

        self.mesh = FineScaleMesh(mesh_file, dim=2)
        self.ellipsis_params_range = ellipsis_params_range
        self.num_ellipsis = num_ellipsis
        if num_fractures > comb(self.num_ellipsis, 2):
            raise ValueError(
                "The number of fractures must be inferior to the number of possible pairs of ellipsis.")
        self.num_fractures = num_fractures
        self.random_rng = np.random.default_rng()

    def run(self):
        """
        Main method. Generates a mesh containing fractures and vugs.
        """

        centroids = self.mesh.faces.center[:][:, 0:2]
        xs, ys = centroids[:, 0], centroids[:, 1]
        x_range = xs.min(), xs.max()
        y_range = ys.min(), ys.max()
        centers, params, angles = self.get_random_ellipsis(x_range, y_range)

        print('Computing vugs')
        faces_per_ellipsis = self.compute_vugs(
            centers, angles, params, centroids)
        print('Computing fractures')
        self.compute_fractures(faces_per_ellipsis, centers)
        print('Done!')

    def compute_vugs(self, centers, angles, params, centroids):
        """
        Compute the volumes inside the ellipsis given by centers, angles and params.

        Parameters
        ------
        centers : numpy.array
            Array containing the cartesian coordinates of
            each ellipsoid center.

        angles : numpy.array
            Array containing the values (in radians) of the
            three rotation angles with respect to the cartesian axis.

        params : numpy.array
            Array containing the parameters of each ellipsoid, i.e,
            the size of the axis.

        centroids : numpy.array
            The centroids of the volumes compouding the mesh.

        Returns
        ------
        vols_per_ellipsoid : list
            A list of Pymoab's ranges describing the volumes
            inside each ellipsoid.
        """

        faces_per_ellipsis = []
        for center, param, angle in zip(centers, params, angles):
            R = self.get_rotation_matrix(angle)
            X = (centroids - center).dot(R.T)
            faces_in_vug = (X / param)**2
            faces_in_vug = faces_in_vug.sum(axis=1)
            # Recuperar range dos volumes que estão no vug e salvar na lista faces_per_ellipsis
            faces_per_ellipsis.append(
                self.mesh.core.all_faces[faces_in_vug < 1])
            self.mesh.vug[faces_in_vug < 1] = 1

        return faces_per_ellipsis

    def compute_fractures(self, faces_per_ellipsis, centers):
        """
        Generates random fractures, i.e, rectangles connecting two vugs, 
        and computes the volumes inside them. If a volumes is inside a 
        fracture, then the property "fracture" from the mesh data 
        structure is set to 1.

        Parameters
        ----------
        faces_per_ellipsis : list 
            A list of Pymoab's ranges describing the volumes
            inside each ellipsoid.

        centers : numpy.array 
            Array containing the cartesian coordinates of
            each ellipsis' center.

        Returns
        -------
        None

        """

        selected_pairs = []
        num_possible_pairs = comb(self.num_ellipsis, 2)
        found = True
        for i in range(self.num_fractures):
            # Find a pair of ellipsis that are not overlapped and are
            # not already connected by a fracture.
            count = 0
            while True:
                count += 1
                e1, e2 = self.random_rng.choice(
                    np.arange(self.num_ellipsis), size=2, replace=False)
                if (e1, e2) not in selected_pairs and \
                        rng.intersect(faces_per_ellipsis[e1], faces_per_ellipsis[e2]).empty():
                    selected_pairs.extend([(e1, e2), (e2, e1)])
                    break
                if count > num_possible_pairs:
                    found = False
                    break

            if not found:
                break

            # Calculating the rectangle's parameters.
            L = np.linalg.norm(centers[e1] - centers[e2])   # Length
            h = 10 / L  # Height

            print("Creating fracture {} of {}".format(
                i + 1, self.num_fractures))
            self.check_intersections(h, L, centers[e1], centers[e2])

    def get_random_ellipsis(self, x_range, y_range):
        random_centers = np.zeros((self.num_ellipsis, 2))

        random_centers[:, 0] = self.random_rng.uniform(
            low=x_range[0], high=x_range[1], size=self.num_ellipsis)
        random_centers[:, 1] = self.random_rng.uniform(
            low=y_range[0], high=y_range[1], size=self.num_ellipsis)

        random_params = self.random_rng.uniform(low=self.ellipsis_params_range[0],
                                                high=self.ellipsis_params_range[1],
                                                size=(self.num_ellipsis, 2))
        random_angles = self.random_rng.uniform(
            low=0.0, high=2*np.pi, size=self.num_ellipsis)

        return random_centers, random_params, random_angles

    def check_intersections(self, h, L, c1, c2):
        """
        Check which volumes are inside the fracture.

        Parameters
        ----------
        h : float
            Rectangle's height.

        L : float
            Rectangle's length.

        c1 : numpy.array
            Left end of the rectangle's axis.

        c2 : numpy.array
            Right end of the rectangle's axis.

        Returns
        -------
        None

        """

        vertices_coords = self.mesh.nodes.coords[:][:, 0:2]
        edges_endpoints = self.mesh.edges.connectivities[:]
        num_edges_endpoints = edges_endpoints.shape[0]
        edges_endpoints_coords = self.mesh.nodes.coords[edges_endpoints.ravel()][:, 0:2].reshape(
            (num_edges_endpoints, 2, 2))

        # We'll first check whether an edge intersects the line segment between
        # the ellipsis' centers.
        u = edges_endpoints_coords[:, 1, :] - edges_endpoints_coords[:, 0, :]
        v = c2 - c1
        w = edges_endpoints_coords[:, 0, :] - c1

        uv_perp_prod = np.cross(u, v)
        wv_perp_prod = np.cross(w, v)
        uw_perp_prod = np.cross(u, w)

        maybe_intersect = np.where(uv_perp_prod != 0)[0]
        s1 = - wv_perp_prod[maybe_intersect] / uv_perp_prod[maybe_intersect]
        t1 = uw_perp_prod[maybe_intersect] / uv_perp_prod[maybe_intersect]

        intersecting_edges = maybe_intersect[(
            s1 >= 0) & (s1 <= 1) & (t1 >= 0) & (t1 <= 1)]
        faces_in_fracture_from_edges = self.mesh.edges.bridge_adjacencies(
            intersecting_edges, "edges", "faces").ravel()

        # We now check which vertices are inside the fracture's bounding
        # rectangle.
        r = vertices_coords - c1
        norm_v = np.linalg.norm(v)
        d = np.cross(r, v) / norm_v
        l = np.dot(r, v) / norm_v
        vertices_in_fracture = self.mesh.nodes.all[(
            d >= 0) & (d <= h) & (l >= 0) & (l <= L)]
        faces_in_fracture_from_nodes = np.concatenate(self.mesh.nodes.bridge_adjacencies(
            vertices_in_fracture, "edges", "faces")).ravel()

        faces_in_fracture = np.intersect1d(np.unique(faces_in_fracture_from_edges),
                                           np.unique(faces_in_fracture_from_nodes))
        faces_in_fracture_vug_value = self.mesh.vug[faces_in_fracture].flatten(
        )
        filtered_faces_in_fracture = faces_in_fracture[faces_in_fracture_vug_value != 1]
        self.mesh.vug[filtered_faces_in_fracture] = 2

    def get_rotation_matrix(self, angle):
        """
        Calculates the 2D rotation matrix for the given angle.

        Parameters
        ----------
        angle : numpy.array
            The three rotation angles in radians.

        Returns
        -------
        R : numpy.array
            The rotation matrix.

        """

        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        R = np.array((cos_theta, sin_theta,
                     - sin_theta, cos_theta)).reshape(2, 2)

        return R

    def write_file(self, path):
        """
        Writes the resulting mesh to a file.

        Parameters
        ----------
        path : str
            A string containing the file path.

        Returns
        -------
        None

        """

        vugs_meshset = self.mesh.core.mb.create_meshset()
        self.mesh.core.mb.add_entities(
            vugs_meshset, self.mesh.core.all_faces)
        self.mesh.core.mb.write_file(path, [vugs_meshset])


class DFNMeshGenerator3D(DFNMeshGenerator):
    """
    A 3D mesh generator for fracured and vuggy reservoirs.
    """

    def __init__(self, mesh_file: str, num_free_fractures: int,
                 num_connected_fractures: int = 0,
                 num_ellipsoids: int = 0, ellipsoid_params_range: Iterable = None,
                 fracture_shape: str = "cylinder"):
        """
        Constructor method.

        Parameters
        ----------
        mesh_file : str
            A string containing the path to the input mesh file.

        ellipsis_params_range : iterable
            An iterable of size 2 containing the maximum and the
            minimum value for the ellipsoids parameters.

        num_ellipsoids : int
            The number of ellipsoids to create.

        num_fractures : int
            The number of fractures to create.

        fracture_shape: str
            The shape of fractures to be generated.

        Raises
        ------
        ValueError
            If the number of fractures is greater than the number of possible
            pairs of ellipsoids.

        """
        if num_ellipsoids == 0 and num_connected_fractures > 0:
            raise ValueError(
                "Cannot generate fractures connecting vugs if the number of vugs is 0.")
        elif num_ellipsoids > 0 and num_connected_fractures > comb(num_ellipsoids, 2):
            raise ValueError(
                "The number of fractures must be inferior to the number of possible pairs of ellipsoids.")
        elif num_ellipsoids > 0 and ellipsoid_params_range is None:
            raise ValueError(
                "Please specify the range of parameters for vug ellipsoids.")

        if fracture_shape not in ("cylinder", "box", "ellipsoid"):
            raise ValueError("Invalid shape for fractures.")

        self.mesh = FineScaleMesh(mesh_file)
        self.ellipsoid_params_range = ellipsoid_params_range
        self.num_ellipsoids = num_ellipsoids
        self.num_connected_fractures = num_connected_fractures
        self.num_free_fractures = num_free_fractures
        self.fracture_shape = fracture_shape
        self.random_rng = np.random.default_rng()

    def run(self):
        """
        Main method. Creates random sized and rotated ellipsoids and
        assigns the volumes inside the vugs.
        """
        centroids = self.mesh.volumes.center[:]
        xs, ys, zs = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        x_range = xs.min(), xs.max()
        y_range = ys.min(), ys.max()
        z_range = zs.min(), zs.max()

        # Generate the vugs to supplied specs.
        print("Computing vugs.")

        centers, params, angles = self.get_random_ellipsoids(
            x_range, y_range, z_range)
        vols_per_ellipsoid = self.compute_vugs(
            centers, angles, params, centroids)

        # Generate free fractures to specs.
        print("Computing free fractures.")

        num_endpoints = 2 * self.num_free_fractures
        pseudo_centers = np.zeros((num_endpoints, 3))
        pseudo_centers[:, 0] = self.random_rng.uniform(
            low=x_range[0], high=x_range[1], size=num_endpoints)
        pseudo_centers[:, 1] = self.random_rng.uniform(
            low=y_range[0], high=y_range[1], size=num_endpoints)
        pseudo_centers[:, 2] = self.random_rng.uniform(
            low=z_range[0], high=z_range[1], size=num_endpoints)

        self.compute_fractures(self.num_free_fractures,
                               vols_per_ellipsoid, pseudo_centers, True)

        # Generate fractures connecting vugs.
        print("Computing fractures between vugs.")
        self.compute_fractures(self.num_connected_fractures,
                               vols_per_ellipsoid, centers, False)

        print("Done!")

    def compute_vugs(self, centers, angles, params, centroids):
        """
        Generates random ellipsoids and computes the volumes inside those
        ellipsoids. If a volumes is inside a vug, the property "vug" from
        the mesh data structure is set to 1.

        Parameters
        ----------
        centers : numpy.array
            Array containing the cartesian coordinates of
            each ellipsoid center.

        angles : numpy.array
            Array containing the values (in radians) of the
            three rotation angles with respect to the cartesian axis.

        params : numpy.array
            Array containing the parameters of each ellipsoid, i.e,
            the size of the axis.

        centroids : numpy.array
            The centroids of the volumes compouding the mesh.

        Returns
        ------
        vols_per_ellipsoid : list
            A list of Pymoab's ranges describing the volumes
            inside each ellipsoid.

        """
        vols_per_ellipsoid = []
        for center, param, angle in zip(centers, params, angles):
            R = self.get_rotation_matrix(angle)
            X = (centroids - center).dot(R.T)
            vols_in_vug = (X / param)**2
            vols_in_vug = vols_in_vug.sum(axis=1)
            # Recuperar range dos volumes que estão no vug e salvar na lista vols_per_ellipsoid
            vols_per_ellipsoid.append(
                self.mesh.core.all_volumes[vols_in_vug < 1])
            self.mesh.vug[vols_in_vug < 1] = 1

        return vols_per_ellipsoid

    def compute_fractures(self, num_fractures, vols_per_ellipsoid, centers, is_free):
        """
        Generate fractures according to the instance parameters 
        `fracture_shape` and `num_fractures`.

        Parameters
        ----------
        vols_per_ellipsoid : list
            A list of Pymoab's ranges describing the volumes
            inside each ellipsoid.

        centers : numpy.array
            Array containing the cartesian coordinates of
            each vug ellipsoid center.

        Returns
        ------
        None

        """
        if self.fracture_shape == "cylinder":
            self.compute_fractures_as_cylinders(
                num_fractures, vols_per_ellipsoid, centers, is_free)
        elif self.fracture_shape == "box":
            self.compute_fractures_as_boxes(
                num_fractures, vols_per_ellipsoid, centers, is_free)
        elif self.fracture_shape == "ellipsoid":
            self.compute_fractures_as_ellipsoids(
                num_fractures, vols_per_ellipsoid, centers, is_free)

    def compute_fractures_as_cylinders(self, num_fractures, vols_per_ellipsoid, centers, is_free):
        """
        Generates random fractures shaped as cylinders connecting two vugs, 
        and computes the volumes inside them. If a volumes is inside a 
        fracture, then the property `fracture` of the mesh data 
        structure is set to 1.

        Parameters
        ----------
        vols_per_ellipsoid : list
            A list of Pymoab's ranges describing the volumes
            inside each ellipsoid.

        centers : numpy.array
            Array containing the cartesian coordinates of
            each vug ellipsoid center.

        Returns
        ------
        None

        """
        selected_pairs = []
        count = 0
        for i in range(num_fractures):
            if is_free:
                # If free fractures are being generated, then just use the
                # pseudo-centers in order.
                e1, e2 = count, count + 1
                count += 2
            else:
                # Else, vugs were generated, so find a pair of ellipsoids that are
                # not overlapped and are not already connected by a fracture.
                e1, e2 = self._find_a_pair_of_vugs(
                    vols_per_ellipsoid, selected_pairs)

            # Calculating the cylinder's parameters.
            L = np.linalg.norm(centers[e1] - centers[e2])   # Length
            r = 10 / L  # Radius

            print("Creating fracture {} of {}".format(i+1, num_fractures))
            self.check_intersections_for_cylinders(
                r, L, centers[e1], centers[e2])

    def compute_fractures_as_boxes(self, num_fractures, vols_per_ellipsoid, centers, is_free):
        """
        Generates random fractures shaped as boxes connecting two vugs, 
        and computes the volumes inside them. If a volumes is inside a 
        fracture, then the property `fracture` of the mesh data 
        structure is set to 1.

        Parameters
        ----------
        vols_per_ellipsoid : list
            A list of Pymoab's ranges describing the volumes
            inside each ellipsoid.

        centers : numpy.array
            Array containing the cartesian coordinates of
            each ellipsoid center.

        Returns
        ------
        None

        """
        # Compute minimal parameter size for boxes.
        all_edges_endpoints = self.mesh.edges.connectivities[:]
        N = len(all_edges_endpoints)
        all_edges_coords = self.mesh.nodes.coords[all_edges_endpoints.flatten()].reshape(
            (N, 2, 3))
        edges_length = np.linalg.norm(
            all_edges_coords[:, 0, :] - all_edges_coords[:, 1, :], axis=1)
        min_height = edges_length.min()
        min_length = edges_length.max()

        selected_pairs = []
        count = 0
        for i in range(num_fractures):
            if is_free:
                # If free fractures are being generated, then just use the
                # pseudo-centers in order.
                e1, e2 = count, count + 1
                count += 2
            else:
                # Else, vugs were generated, so find a pair of ellipsoids that are
                # not overlapped and are not already connected by a fracture.
                e1, e2 = self._find_a_pair_of_vugs(
                    vols_per_ellipsoid, selected_pairs)

            d = np.linalg.norm(centers[e1] - centers[e2])
            l = min_length if min_length > d / 20 else d / 20
            h = min_height

            print("Creating fracture {} of {}".format(i + 1, num_fractures))
            self.check_intersections_for_boxes(
                centers[e1], centers[e2], d, l, h)

    def compute_fractures_as_ellipsoids(self, num_fractures, vols_per_ellipsoid, centers, is_free):
        """
        Generates random fractures shaped as ellipsoids connecting two vugs, 
        and computes the volumes inside them. If a volumes is inside a 
        fracture, then the property `fracture` of the mesh data 
        structure is set to 1.

        Parameters
        ----------
        vols_per_ellipsoid : list
            A list of Pymoab's ranges describing the volumes
            inside each ellipsoid.

        centers : numpy.array
            Array containing the cartesian coordinates of
            each ellipsoid center.

        Returns
        ------
        None

        """
        # Compute minimal parameter size for ellipsoids.
        all_edges_endpoints = self.mesh.edges.connectivities[:]
        N = len(all_edges_endpoints)
        all_edges_coords = self.mesh.nodes.coords[all_edges_endpoints.flatten()].reshape(
            (N, 2, 3))
        edges_length = np.linalg.norm(
            all_edges_coords[:, 0, :] - all_edges_coords[:, 1, :], axis=1)
        param_c = edges_length.min()

        selected_pairs = []
        count = 0
        for i in range(num_fractures):
            if is_free:
                # If free fractures are being generated, then just use the
                # pseudo-centers in order.
                e1, e2 = count, count + 1
                count += 2
            else:
                # Else, vugs were generated, so find a pair of ellipsoids that are
                # not overlapped and are not already connected by a fracture.
                e1, e2 = self._find_a_pair_of_vugs(
                    vols_per_ellipsoid, selected_pairs)

            print("Creating fracture {} of {}".format(
                i + 1, num_fractures))

            c1, c2 = centers[e1], centers[e2]
            d = np.linalg.norm(c1 - c2)
            params = np.array((d / 2, d / 20, param_c))

            self.check_intersections_for_ellipsoids(c1, c2, params)

    def check_intersections_for_cylinders(self, R, L, c1, c2):
        """
        Check which volumes are inside the fracture.

        Parameters
        ----------
        R : float
            Cylinder's radius

        L : float
            Cylinder's length

        c1 : numpy.array
            Left end of the cylinder's axis.

        c2 : numpy.array
            Right end of the cylinder's axis.

        Returns
        ------
        None

        """
        vertices = self.mesh.nodes.coords[:]

        # Cylinder's vector parameters.
        e = c2 - c1
        m = np.cross(c1, c2)

        # Calculating the distance between the vertices and the main axis.
        d_vector = m + np.cross(e, vertices)
        d = np.linalg.norm(d_vector, axis=1) / L

        # Computing the projection of the vertices onto the cylinder's axis.
        u = vertices - c1
        proj_vertices = u.dot(e) / L

        # Checking which vertices are inside the cylinder.
        vertices_in_cylinder = self.mesh.nodes.all[(d <= R) & (
            proj_vertices >= 0) & (proj_vertices <= L)]
        if len(vertices_in_cylinder) > 0:
            volumes_in_cylinder = self.mesh.nodes.bridge_adjacencies(vertices_in_cylinder,
                                                                     "edges", "volumes").ravel()

            if volumes_in_cylinder.dtype == "object":
                volumes_in_cylinder = np.concatenate(volumes_in_cylinder)

            volumes_in_cylinder = np.unique(volumes_in_cylinder)
            volumes_vug_value = self.mesh.vug[volumes_in_cylinder].flatten()
            non_vug_volumes = volumes_in_cylinder[volumes_vug_value == 0]
            self.mesh.vug[non_vug_volumes] = 2

        self._check_intersections_along_axis(c1, c2)

    def check_intersections_for_boxes(self, c1, c2, d, l, h):
        """
        Check which volumes are inside the box shaped fracture.

        Parameters
        ----------

        c1 : numpy.array
            Left end of the cylinder's axis.

        c2 : numpy.array
            Right end of the cylinder's axis.

        d: float
            Depth of box.

        l: float
            Length of box.

        h: float
            Height of box.

        Returns
        ------
        None

        """
        # Box center.
        center = (c1 + c2) / 2

        # Defining an orientation vector so we can compute rotations.
        u = np.array([d / 2, 0.0, 0.0])
        v = c1 - center

        R = rotation_to_align(u, v)

        # Compute the rotated axis.
        rotated_ax = np.array([1.0, 0.0, 0.0]).dot(R)
        rotated_ay = np.array([0.0, 1.0, 0.0]).dot(R)
        rotated_az = np.array([0.0, 0.0, 1.0]).dot(R)

        # Compute volumes inside the box (what's in the box!?).
        vertices = self.mesh.nodes.coords[:]
        X = vertices - center
        vertices_in_x_range = np.abs(X.dot(rotated_ax)) <= (d / 2)
        vertices_in_y_range = np.abs(X.dot(rotated_ay)) <= (l / 2)
        vertices_in_z_range = np.abs(X.dot(rotated_az)) <= (h / 2)
        vertices_handles = self.mesh.nodes.all[vertices_in_x_range &
                                               vertices_in_y_range & vertices_in_z_range]

        if len(vertices_handles) > 0:
            vols_in_fracture = np.concatenate(self.mesh.nodes.bridge_adjacencies(
                vertices_handles, "edges", "volumes")).ravel()
            unique_vols_in_fracture = np.unique(vols_in_fracture)
            unique_volumes_vug_values = self.mesh.vug[unique_vols_in_fracture].flatten(
            )
            non_vug_volumes = unique_vols_in_fracture[unique_volumes_vug_values == 0]
            self.mesh.vug[non_vug_volumes] = 2

        self._check_intersections_along_axis(c1, c2)

    def check_intersections_for_ellipsoids(self, c1, c2, params):
        """
        Check which volumes are inside the ellipsoid shaped fracture.

        Parameters
        ----------

        c1 : numpy.array
            Left end of the cylinder's axis.

        c2 : numpy.array
            Right end of the cylinder's axis.

        Returns
        ------
        None

        """
        # Ellipsoid's parameters
        center = (c1 + c2) / 2

        # Defining orientation vectors.
        u = np.array([params[0], 0.0, 0.0])
        v = c1 - center

        R = rotation_to_align(u, v)

        vertices = self.mesh.nodes.coords[:]
        X = (vertices - center).dot(R.T)
        vertices_in_ellipsoid = ((X / params)**2).sum(axis=1)
        vertices_handles = self.mesh.nodes.all[vertices_in_ellipsoid < 1]

        if len(vertices_handles) > 0:
            vols_in_fracture = np.concatenate(self.mesh.nodes.bridge_adjacencies(
                vertices_handles, "edges", "volumes")).ravel()
            unique_vols_in_fracture = np.unique(vols_in_fracture)
            unique_volumes_vug_values = self.mesh.vug[unique_vols_in_fracture].flatten(
            )
            non_vug_volumes = unique_vols_in_fracture[unique_volumes_vug_values == 0]
            self.mesh.vug[non_vug_volumes] = 2

        self._check_intersections_along_axis(c1, c2)

    def _check_intersections_along_axis(self, c1, c2):
        # Check for intersection between the box's axis and the mesh faces.
        faces = self.mesh.faces.all[:]
        num_faces = len(faces)
        faces_nodes_handles = self.mesh.faces.connectivities[:]
        num_vertices_of_volume = faces_nodes_handles.shape[1]
        faces_vertices = self.mesh.nodes.coords[faces_nodes_handles.flatten()].reshape(
            (num_faces, num_vertices_of_volume, 3))

        # Plane parameters of each face.
        R_0 = faces_vertices[:, 0, :]
        N = np.cross(faces_vertices[:, 1, :] - R_0,
                     faces_vertices[:, 2, :] - R_0)

        # Compute the parameters of the main axis line.
        num = np.einsum("ij,ij->i", N, R_0 - c1)
        denom = N.dot(c2 - c1)

        non_zero_denom = denom[np.abs(denom) > 1e-6]
        non_zero_num = num[np.abs(denom) > 1e-6]
        r = non_zero_num / non_zero_denom

        # Check faces intersected by the axis' line.
        filtered_faces = faces[np.abs(denom) > 1e-6]
        filtered_faces = filtered_faces[(r >= 0) & (r <= 1)]
        filtered_nodes = faces_vertices[np.abs(denom) > 1e-6]
        filtered_nodes = filtered_nodes[(r >= 0) & (r <= 1)]

        r = r[(r >= 0) & (r <= 1)]
        P = c1 + r[:, np.newaxis]*(c2 - c1)

        # Compute the intersection point between the face plane and the axis
        # line and check if such point is in the face.
        angle_sum = np.zeros(filtered_nodes.shape[0])
        for i in range(num_vertices_of_volume):
            p0, p1 = filtered_nodes[:, i, :], filtered_nodes[:,
                                                             (i+1) % num_vertices_of_volume, :]
            a = p0 - P
            b = p1 - P
            norm_prod = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
            # If the point of intersection is too close to a vertex, then
            # take it as the vertex itself.
            angle_sum[norm_prod <= 1e-6] = 2*np.pi
            cos_theta = np.einsum("ij,ij->i", a, b) / norm_prod
            theta = np.arccos(cos_theta)
            angle_sum += theta

        # If the sum of the angles around the intersection point is 2*pi, then
        # the point is inside the polygon.
        intersected_faces = filtered_faces[np.abs(2*np.pi - angle_sum) < 1e-6]
        volumes_sharing_face = self.mesh.faces.bridge_adjacencies(
            intersected_faces, "faces", "volumes")
        unique_volumes = np.unique(volumes_sharing_face.ravel())
        unique_volumes_vug_values = self.mesh.vug[unique_volumes].flatten()
        non_vug_volumes = unique_volumes[unique_volumes_vug_values == 0]
        self.mesh.vug[non_vug_volumes] = 2

    def _find_a_pair_of_vugs(self, vols_per_ellipsoid, selected_pairs):
        while True:
            e1, e2 = self.random_rng.choice(
                np.arange(self.num_ellipsoids), size=2, replace=False)
            if (e1, e2) not in selected_pairs and \
                    rng.intersect(vols_per_ellipsoid[e1], vols_per_ellipsoid[e2]).empty():
                selected_pairs.extend([(e1, e2), (e2, e1)])
                break
        return e1, e2

    def write_file(self, path="results/vugs.vtk"):
        """
        Writes the resulting mesh into a file. Default path is 'results/vugs.vtk'.

        Parameters
        ----------
        path : str
            A string containing the file path.

        Returns
        -------
        None

        """
        vugs_meshset = self.mesh.core.mb.create_meshset()
        self.mesh.core.mb.add_entities(
            vugs_meshset, self.mesh.core.all_volumes)
        self.mesh.core.mb.write_file(path, [vugs_meshset])

    def get_random_ellipsoids(self, x_range, y_range, z_range):
        """
        Generates random points as the ellipsoids centers as the axis sizes 
        and random rotation angles with respect to the cartesian coordinates (x,y,z).

        Parameters
        ----------
        x_range : iterable
            An iterable containing the maximum and minimum 
            values of the x coordinate.

        y_range : iterable
            An iterable containing the maximum and minimum 
            values of the y coordinate.

        z_range : iterable
            An iterable containing the maximum and minimum 
            values of the z coordinate.

        Returns
        -------
        random_centers : numpy.array
            The generated center points for the ellipsoids.

        random_params : numpy.array
            The parameters a.k.a the size of the three axis of the ellipsoids.

        random_angles : numpy.array
            The rotation angles for each ellipsoid.

        """
        if self.num_ellipsoids == 0:
            return [], [], []

        random_centers = np.zeros((self.num_ellipsoids, 3))

        random_centers[:, 0] = self.random_rng.uniform(
            low=x_range[0], high=x_range[1], size=self.num_ellipsoids)
        random_centers[:, 1] = self.random_rng.uniform(
            low=y_range[0], high=y_range[1], size=self.num_ellipsoids)
        random_centers[:, 2] = self.random_rng.uniform(
            low=z_range[0], high=z_range[1], size=self.num_ellipsoids)
        random_params = self.random_rng.uniform(low=self.ellipsoid_params_range[0],
                                                high=self.ellipsoid_params_range[1],
                                                size=(self.num_ellipsoids, 3))
        random_angles = self.random_rng.uniform(
            low=0.0, high=2*np.pi, size=(self.num_ellipsoids, 3))

        return random_centers, random_params, random_angles

    def get_rotation_matrix(self, angle):
        """
        Calculates the 3D rotation matrix for the given angle.

        Parameters
        ----------
        angle : numpy.array
            The three rotation angles in radians.

        Returns
        -------
        R : numpy.array
            The rotation matrix.

        """
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)
        R = np.array([
            cos_ang[1] * cos_ang[2],
            cos_ang[2] * sin_ang[1] * sin_ang[0] - sin_ang[2] * cos_ang[0],
            cos_ang[2] * sin_ang[1] * cos_ang[0] + sin_ang[2] * sin_ang[0],
            sin_ang[2] * cos_ang[1],
            sin_ang[2] * sin_ang[1] * sin_ang[0] + cos_ang[2] * cos_ang[0],
            sin_ang[2] * sin_ang[1] * cos_ang[0] - cos_ang[2] * sin_ang[0],
            - sin_ang[1],
            cos_ang[1] * sin_ang[0],
            cos_ang[1] * cos_ang[0]
        ]).reshape((3, 3))

        return R
