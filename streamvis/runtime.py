import numpy as np


class Runtime:
    """A utility class that provides a view on the current data for each document.
    """

    def __init__(self):
        self.current_metadata = dict(shape=[1, 1])
        self.current_image = np.zeros((1, 1), dtype="float32")
        self.thresholded_image = np.zeros((1, 1), dtype="float32")
        self.aggregated_image = np.zeros((1, 1), dtype="float32")
        self.reset = True
