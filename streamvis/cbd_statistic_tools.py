import numpy as np
from time import time

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


class AggregatorWithID:
    def __init__(self, dtype, empty_value,
                 max_span=500_000,
                 aggregate=np.sum,
                 sorting=False,
                 max_sort_n=100):
        self._x = np.full(shape=(max_span,), dtype=dtype, fill_value=empty_value)
        self._id = np.full(shape=(max_span,), dtype=int, fill_value=0)
        self._nan = empty_value
        self.last_value = self._nan
        self._aggregate = aggregate
        self._max_sort_n = max_sort_n
        self._sorting = sorting
        self.last_processed_index = 0

    def update(self, values, pulse_id):
        if pulse_id is None:
            print(f"Can't update the aggregator: pulse Id is None")
            return

        # Only store aggregated value per pulse Id
        value = self._aggregate(values)
        self._x = np.roll(self._x, -1)
        self._x[-1] = value
        self.last_value = value

        self._id = np.roll(self._id, -1)
        self._id[-1] = pulse_id

        self.last_processed_index -= 1

        if self._sorting:
            t0 = time()
            self.sort_by_id()
            t1 = time()
            print(f"Sorted update took {t1 - t0} for {self.count} elements")

    @property
    def count(self) -> int:
        return len(self._x[self._x != self._nan])

    def sort_by_id(self):
        cnt = max(self.count, self._max_sort_n)
        indices = self._id[:cnt].argsort()
        self._id[:cnt] = self._id[indices]
        self._x[:cnt] = self._x[indices]

    def clear(self):
        self._x[...] = self._nan
        self._id[...] = 0
        self.last_processed_index = 0

    def __call__(self, *args, **kwargs):
        start_id = self.last_processed_index
        self.last_processed_index = 0
        return self._x[start_id:], self._id[start_id:]

    @property
    def last(self):
        return self._x[-1], self._id[-1]
