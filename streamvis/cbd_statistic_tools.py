import numpy as np


class NPFIFOArray:
    def __init__(self, dtype, empty_value, max_span=120_000, aggregate=np.average):
        self._x = np.full(shape=(max_span,), dtype=dtype, fill_value=empty_value)
        self._nan = empty_value
        self.last_value = self._nan
        self._aggregate = aggregate

    def update(self, values):
        self._x = np.roll(self._x, len(values))
        self._x[:len(values)] = values
        self.last_value = self._aggregate(values)

    def clear(self):
        self._x[...] = self._nan

    def __bool__(self):
        return bool(np.any(self._x != self._nan))

    def __call__(self, *args, **kwargs):
        return self._x[self._x != self._nan]

    @property
    def min(self):
        return np.min(self.__call__())

    @property
    def max(self):
        return np.max(self.__call__())
