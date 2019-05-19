import numpy as np


class Runtime:
    def __init__(self):
        self.current_metadata = dict(shape=[1, 1])
        self.current_image = np.zeros((1, 1), dtype='float32')
