import numpy as np


class FractureGenerator2D(object):
    """
    A 2D fractures generator according to a D-1 model. Fractures are
    represented by line segments.
    """

    def __init__(self, num_fractures, bounding_box_dimensions):
        """
        Parameters
        ------
        num_fractures: int
            Number of fractures to be generated.
        bounding_box_dimensions: tuple
            Dimensions of the bounding box across the x and y axis.
        """
        self.num_fractures = num_fractures
        self.bbox_dimensions = bounding_box_dimensions
        self.random_rng = np.random.default_rng()
        self.fractures = np.zeros((self.num_fractures, 2, 2))

    def generate_fractures(self):
        """
        Generates the fractures and sets the fractures attribute.
        """
        self.fractures[:, :, 0] = self.random_rng.uniform(
            high=self.bbox_dimensions[0], low=0.0, size=(self.num_fractures, 2))
        self.fractures[:, :, 1] = self.random_rng.uniform(
            high=self.bbox_dimensions[1], low=0.0, size=(self.num_fractures, 2))

    def export_fractures_to_file(self, path):
        """
        Write the fractures matrix to a text file.
        """
        reshaped_fractures = self.fractures.reshape((self.num_fractures, 4))
        np.savetxt(path, reshaped_fractures)        
