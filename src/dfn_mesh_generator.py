import numpy as np
from scipy.special import comb
from pymoab import rng
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
from preprocessor.element_order.MPFAD2DOrdering import MPFAD2DOrdering

class DFNMeshGenerator(object):
    """
    A mesh generator for fracured and vuggy reservoirs.
    """
    def __init__(self, mesh_file, ellipsis_params_range, num_ellipsoids, num_fractures):
        """
        Constructor method.

        Inputs
        ------
        mesh_file (String): A string containing the path to the input mesh file.

        ellipsis_params_range (iterable of size 2): A iterable containing the maximum and the
        minimum value for the ellipsoids parameters.

        num_ellipsoids (int): the number of ellipsoids to create.
        
        """
        self.mesh = FineScaleMesh(mesh_file)
        self.ellipsis_params_range = ellipsis_params_range
        self.num_ellipsoids = num_ellipsoids
        if num_fractures > comb(self.num_ellipsoids, 2):
            raise ValueError("The number of fractures must be inferior to the number of possible pairs of ellipsoids.")
        self.num_fractures = num_fractures

    def run(self):
        """
        Main method. Creates random sized and rotated ellipsoids and
        assigns the volumes inside the vugs.
        
        Inputs
        ------
        None

        Output
        ------
        None

        """
        centroids = self.mesh.volumes.center[:]
        xs, ys, zs = centroids[:, 0], centroids[:, 1], centroids[:, 2]
        x_range = xs.min(), xs.max()
        y_range = ys.min(), ys.max()
        z_range = zs.min(), zs.max()
        centers, params, angles = self.get_random_ellipsoids(x_range, y_range, z_range)

        print('Computing vugs')
        vols_per_ellipsoid = self.compute_vugs(centers, angles, params, centroids)
        print('Computing fractures')
        self.compute_fractures(vols_per_ellipsoid, centers, angles, params, centroids)
        print('Done!')

    def compute_vugs(self, centers, angles, params, centroids):
        """
        Generates random ellipsoids and computes the volumes inside those
        ellipsoids. If a volumes is inside a vug, the property "vug" from
        the mesh data structure is set to 1.
        
        Inputs
        ------
        centers (numpy array): Array containing the cartesian coordinates of
        each ellipsoid center.

        angles (numpy array): Array containing the values (in radians) of the
        three rotation angles with respect to the cartesian axis.

        params (numpy array): Array containing the parameters of each ellipsoid, i.e,
        the size of the axis.

        centroids (numpy array): The centroids of the volumes compouding the mesh.

        Output
        ------
        vols_per_ellipsoid (list): A list of Pymoab's ranges describing the volumes
        inside each ellipsoid.

        """
        vols_per_ellipsoid = []
        for center, param, angle in zip(centers, params, angles):
            R = self.get_rotation_matrix(angle)
            X = (centroids - center).dot(R.T)
            vols_in_vug = (X / param)**2
            vols_in_vug = vols_in_vug.sum(axis=1)
            # Recuperar range dos volumes que estão no vug e salvar na lista vols_per_ellipsoid
            vols_per_ellipsoid.append(self.mesh.core.all_volumes[vols_in_vug < 1])
            self.mesh.vug[vols_in_vug < 1] = 1
        
        return vols_per_ellipsoid

    def compute_fractures(self, vols_per_ellipsoid, centers, angles, params, centroids):
        """
        Generates random fractures, i.e, cylinders connecting two vugs, 
        and computes the volumes inside them. If a volumes is inside a 
        fracture, then the property "fracture" from the mesh data 
        structure is set to 1.
        
        Inputs
        ------
        vols_per_ellipsoid (list): A list of Pymoab's ranges describing the volumes
        inside each ellipsoid.

        centers (numpy array): Array containing the cartesian coordinates of
        each ellipsoid center.

        angles (numpy array): Array containing the values (in radians) of the
        three rotation angles with respect to the cartesian axis.

        params (numpy array): Array containing the parameters of each ellipsoid, i.e,
        the size of the axis.

        centroids (numpy array): The centroids of the volumes compouding the mesh.

        Output
        ------
        None

        """
        random_rng = np.random.default_rng(42)
        selected_pairs = []
        for _ in range(self.num_fractures):
            # Find a pair of ellipsoids that are not overlapped and are
            # not already connected by a fracture.
            while True:
                e1, e2 = random_rng.choice(np.arange(self.num_ellipsoids), size=2, replace=False)
                if (e1, e2) not in selected_pairs and \
                        rng.intersect(vols_per_ellipsoid[e1], vols_per_ellipsoid[e2]).empty():
                    selected_pairs.extend([(e1, e2), (e2, e1)])
                    break
            # Calculating the cylinder's parameters.
            L = np.linalg.norm(centers[e1] - centers[e2])   # Length
            r = 10 / L  # Radius

            self.check_intersections(r, L, centers[e1], centers[e2])

    def check_intersections(self, R, L, c1, c2):
        """
        Verify which volumes are intersected by fracture.
        """
        print("Checking intersections for fracture")

        print("Checking nodes inside cylinder")
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
        vertices_in_cylinder = self.mesh.nodes.all[(d <= R) & (proj_vertices >= 0) & (proj_vertices <= L)]
        volumes_in_cylinder = np.concatenate(self.mesh.nodes.bridge_adjacencies(vertices_in_cylinder, 
                                                                                "edges", "volumes")).ravel()
        volumes_in_cylinder = np.unique(volumes_in_cylinder)
        volumes_vug_value = self.mesh.vug[volumes_in_cylinder].flatten()
        non_vug_volumes = volumes_in_cylinder[volumes_vug_value == 0]
        self.mesh.vug[non_vug_volumes] = 2
        
        # Check if the cylinder's axis intercept any of the faces of a volume.
        order = MPFAD2DOrdering(self.mesh.nodes, "edges")
        faces = self.mesh.faces.all[:]
        print("Checking intersection between axis and faces")
        for face in faces:
            face_nodes = self.mesh.faces.bridge_adjacencies(face, "edges", 
                                                            "nodes", ordering_inst=order)
            v0, v1, v2 = self.mesh.nodes.coords[face_nodes[0:3]]
            normal = np.cross(v1 - v0, v2 - v0)
            denom = normal.dot(c2 - c1)
            num = normal.dot(v0 - c1)
            if np.abs(denom) < 1e-6:
                continue
            elif num / denom < 0 or num / denom > 1:
                continue
            r = num / denom
            # Point of intersection between the plane defined by the face 
            # and the axis.
            P = c1 + r*(c2 - c1)
            angle_sum = 0
            n = len(face_nodes)
            face_nodes_coords = self.mesh.nodes.coords[face_nodes]
            for i in range(n):
                p0, p1 = face_nodes_coords[i], face_nodes_coords[(i+1) % n]
                a = p0 - P
                b = p1 - P
                norm_prod = np.linalg.norm(a)*np.linalg.norm(b)
                # If the point of intersection is too close to a vertex, then
                # take it as the vertex itself.
                if norm_prod <= 1e-6:
                    angle_sum = 2*np.pi
                    break
                cos_theta = a.dot(b) / norm_prod
                theta = np.arccos(cos_theta)
                angle_sum += theta
            # If the sum of the angles around the intersection point is 2*pi, then
            # the point is inside the polygon.
            volumes_sharing_face = self.mesh.faces.bridge_adjacencies(face, "faces", "volumes")
            non_vugs_volumes = [volume for volume in volumes_sharing_face if self.mesh.vug[volume] == 0]
            if np.abs(2*np.pi - angle_sum) < 1e-6 and len(non_vugs_volumes) > 0:
                self.mesh.vug[non_vugs_volumes] = 2

    def write_file(self, path="results/vugs.vtk"):
        """
        Writes the resultant mesh into a file. Default path is 'results/vugs.vtk'.

        Inputs
        ------
        path (String): A string containing the output file name.

        Outputs
        -------
        None
        
        """
        vugs_meshset = self.mesh.core.mb.create_meshset()
        self.mesh.core.mb.add_entities(vugs_meshset, self.mesh.core.all_volumes)
        self.mesh.core.mb.write_file(path, [vugs_meshset])
    
    def get_random_ellipsoids(self, x_range, y_range, z_range):
        """
        Generates random points as the ellipsoids centers as the axis sizes 
        and random rotation angles with respect to the cartesian coordinates (x,y,z).

        Inputs
        ------
        x_range (iterable of size 2): A iterable containing the maximum and minimum 
        values of the x coordinate.

        y_range (iterable of size 2): A iterable containing the maximum and minimum 
        values of the y coordinate.

        z_range (iterable of size 2): A iterable containing the maximum and minimum 
        values of the z coordinate.

        Outputs
        -------
        random_centers (num_ellipsoids x 3 numpy array): The generated center 
        points for the ellipsoids.

        random_params (num_ellipsoids x 3 numpy array): The parameters a.k.a 
        the size of the three axis of the ellipsoids.

        random_angles (num_ellipsoids x 3 numpy array): The rotation angles 
        for each ellipsoid.
        
        """
        random_rng = np.random.default_rng(42)
        random_centers = np.zeros((self.num_ellipsoids, 3))

        random_centers[:, 0] = random_rng.uniform(low=x_range[0], high=x_range[1], size=self.num_ellipsoids)
        random_centers[:, 1] = random_rng.uniform(low=y_range[0], high=y_range[1], size=self.num_ellipsoids)
        random_centers[:, 2] = random_rng.uniform(low=z_range[0], high=z_range[1], size=self.num_ellipsoids)
        random_params = random_rng.uniform(low=self.ellipsis_params_range[0], \
                                        high=self.ellipsis_params_range[1], \
                                        size=(self.num_ellipsoids, 3))
        random_angles = random_rng.uniform(low=0.0, high=2*np.pi, size=(self.num_ellipsoids, 3))
        
        return random_centers, random_params, random_angles
    
    def get_rotation_matrix(self, angle):
        """
        Calculates the rotation matrix for the given angle. Such matrix is the result of 
        three rotations: first with respect to x, then y, then z.

        Inputs
        ------
        angle (3 x 1 numpy array): The three rotation angles in radians.
        
        Outputs
        -------
        R (3 x 3 numpy array): The rotation matrix.

        """
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)
        R = np.array([
            cos_ang[1]*cos_ang[2], -cos_ang[1]*sin_ang[2], sin_ang[1], \
            sin_ang[0]*sin_ang[1]*cos_ang[2] + cos_ang[0]*sin_ang[2], \
            cos_ang[0]*cos_ang[2] - sin_ang[0]*sin_ang[1]*sin_ang[2], \
            -sin_ang[0]*sin_ang[1], sin_ang[0]*sin_ang[2] - cos_ang[0]*sin_ang[1]*sin_ang[2], \
            cos_ang[0]*sin_ang[1]*sin_ang[2] + sin_ang[0]*cos_ang[2], cos_ang[0]*cos_ang[1]
        ]).reshape((3,3))

        return R
