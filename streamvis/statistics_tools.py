from collections import defaultdict
from functools import partial

import numpy as np


class Hitrate:
    def __init__(self, step_size=100, max_span=120_000):
        self._step_size = step_size
        self._max_num_steps = max_span // step_size

        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._hits = defaultdict(int)
        self._empty = defaultdict(int)

    def __bool__(self):
        return bool(self._hits or self._empty)

    @property
    def step_size(self):
        return self._step_size

    @property
    def max_span(self):
        return self._step_size * self._max_num_steps

    def update(self, pulse_id, is_hit):
        bin_id = pulse_id // self._step_size

        if self._start_bin_id == -1:
            self._start_bin_id = bin_id

        if bin_id < self._start_bin_id:
            # the data is too old
            return

        min_bin_id = max(bin_id - self._max_num_steps, 0)
        if self._start_bin_id < min_bin_id:
            # update start_bin_id and drop old data from the buffers
            for _bin_id in range(self._start_bin_id, min_bin_id):
                self._hits.pop(_bin_id, None)
                self._empty.pop(_bin_id, None)

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the counter
        if is_hit:
            self._hits[bin_id] += 1
        else:
            self._empty[bin_id] += 1

    def __call__(self):
        if not bool(self):
            # no data has been received yet
            return [], []

        # add an extra point for bokeh Step to display the last value
        x = np.arange(self._start_bin_id, self._stop_bin_id + 1)
        y = np.zeros_like(x, dtype=float)

        for ind, _bin_id in enumerate(x):
            hits = self._hits[_bin_id]
            total = hits + self._empty[_bin_id]
            if total:
                y[ind] = hits / total

        return x * self._step_size, y

    def clear(self):
        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._hits.clear()
        self._empty.clear()


class Profile:
    def __init__(self, step_size=100, max_span=10_000):
        self._step_size = step_size
        self._max_num_steps = max_span // step_size

        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._x_limits = []
        self._x = np.zeros(1)
        self._y = defaultdict(int)
        self._n = defaultdict(int)

    def __bool__(self):
        return bool(self._y and self._n)

    @property
    def step_size(self):
        return self._step_size

    @property
    def max_span(self):
        return self._step_size * self._max_num_steps

    def update_x(self, x):
        if isinstance(x, list) and self._x_limits != x:
            self._x_limits = x
            self._x = np.arange(*x)
            self._y.clear()
            self._n.clear()
        if isinstance(x, np.ndarray) and not np.array_equal(self._x_limits, x):
            self._x_limits = x
            self._x = x
            self._y.clear()
            self._n.clear()

    def update_y(self, pulse_id, y):
        bin_id = pulse_id // self._step_size

        if self._start_bin_id == -1:
            self._start_bin_id = bin_id

        if bin_id < self._start_bin_id:
            # the data is too old
            return

        min_bin_id = max(bin_id - self._max_num_steps, 0)
        if self._start_bin_id < min_bin_id:
            # drop old data from the buffers and update start_bin_id
            for _bin_id in range(self._start_bin_id, min_bin_id):
                self._y.pop(_bin_id, None)
                self._n.pop(_bin_id, None)

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the buffers
        self._y[bin_id] += np.array(y)
        self._n[bin_id] += 1

    def __call__(self, pulse_id_window):
        if not bool(self):
            # no data has been received yet
            return [], [], 0

        n_steps = pulse_id_window // self._step_size
        if n_steps > self._max_num_steps:
            raise ValueError("Requested pulse_id window is larger than the maximum pulse_id span.")

        y_sum = 0
        n_sum = 0
        for _bin_id in range(self._stop_bin_id - n_steps, self._stop_bin_id):
            y_sum += self._y[_bin_id]
            n_sum += self._n[_bin_id]

        if n_sum == 0:
            # TODO: is this even possible? i.e. this is probably captured by the first check at the
            # beginning of the function
            return [], [], 0

        return self._x, y_sum / n_sum, n_sum

    def clear(self):
        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._y.clear()
        self._n.clear()


class PumpProbe_nobkg:
    def __init__(self, step_size=100, max_span=120_000):
        self._step_size = step_size
        self._max_num_steps = max_span // step_size

        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._sig_lon = defaultdict(int)
        self._sig_loff = defaultdict(int)
        self._n_sig_lon = defaultdict(int)
        self._n_sig_loff = defaultdict(int)

    def __bool__(self):
        return bool(self._sig_lon and self._sig_loff)

    @property
    def step_size(self):
        return self._step_size

    @property
    def max_span(self):
        return self._step_size * self._max_num_steps

    def update(self, pulse_id, laser_on, sig):
        bin_id = pulse_id // self._step_size

        if self._start_bin_id == -1:
            self._start_bin_id = bin_id

        if bin_id < self._start_bin_id:
            # the data is too old
            return

        min_bin_id = max(bin_id - self._max_num_steps, 0)
        if self._start_bin_id < min_bin_id:
            # update start_bin_id and drop old data from the buffers
            for _bin_id in range(self._start_bin_id, min_bin_id):
                self._sig_lon.pop(_bin_id, None)
                self._sig_loff.pop(_bin_id, None)
                self._n_sig_lon.pop(_bin_id, None)
                self._n_sig_loff.pop(_bin_id, None)

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the buffers
        if laser_on:
            self._sig_lon[bin_id] += sig
            self._n_sig_lon[bin_id] += 1
        else:
            self._sig_loff[bin_id] += sig
            self._n_sig_loff[bin_id] += 1

    def __call__(self, pulse_id_window):
        if not bool(self):
            # no data has been received yet
            return [], []

        n_steps = pulse_id_window // self._step_size
        if n_steps > self._max_num_steps:
            raise ValueError("Requested pulse_id window is larger than the maximum pulse_id span.")

        # add an extra point for bokeh Step to display the last value
        start_bin_id = self._start_bin_id - self._start_bin_id % n_steps
        x = np.arange(start_bin_id, self._stop_bin_id + 1, n_steps)
        y = np.zeros_like(x, dtype=np.float64)

        for ind, _bin_id in enumerate(x):
            sum_sig_lon = sum_n_sig_lon = sum_sig_loff = sum_n_sig_loff = 0
            for shift in range(n_steps):
                sum_sig_lon += self._sig_lon[_bin_id + shift]
                sum_sig_loff += self._sig_loff[_bin_id + shift]
                sum_n_sig_lon += self._n_sig_lon[_bin_id + shift]
                sum_n_sig_loff += self._n_sig_loff[_bin_id + shift]

            if sum_n_sig_lon and sum_sig_loff and sum_n_sig_loff:
                y[ind] = (sum_sig_lon / sum_n_sig_lon) / (sum_sig_loff / sum_n_sig_loff) - 1

        return x * self._step_size, y

    def clear(self):
        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._sig_lon.clear()
        self._sig_loff.clear()
        self._n_sig_lon.clear()
        self._n_sig_loff.clear()


class PumpProbe:
    def __init__(self, step_size=100, max_span=120_000):
        self._step_size = step_size
        self._max_num_steps = max_span // step_size

        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._sig_lon = defaultdict(int)
        self._bkg_lon = defaultdict(int)
        self._sig_loff = defaultdict(int)
        self._bkg_loff = defaultdict(int)

    def __bool__(self):
        return bool(self._sig_lon and self._bkg_lon and self._sig_loff and self._bkg_loff)

    @property
    def step_size(self):
        return self._step_size

    @property
    def max_span(self):
        return self._step_size * self._max_num_steps

    def update(self, pulse_id, laser_on, sig, bkg):
        bin_id = pulse_id // self._step_size

        if self._start_bin_id == -1:
            self._start_bin_id = bin_id

        if bin_id < self._start_bin_id:
            # the data is too old
            return

        min_bin_id = max(bin_id - self._max_num_steps, 0)
        if self._start_bin_id < min_bin_id:
            # update start_bin_id and drop old data from the buffers
            for _bin_id in range(self._start_bin_id, min_bin_id):
                self._sig_lon.pop(_bin_id, None)
                self._bkg_lon.pop(_bin_id, None)
                self._sig_loff.pop(_bin_id, None)
                self._bkg_loff.pop(_bin_id, None)

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the buffers
        if laser_on:
            self._sig_lon[bin_id] += sig
            self._bkg_lon[bin_id] += bkg
        else:
            self._sig_loff[bin_id] += sig
            self._bkg_loff[bin_id] += bkg

    def __call__(self, pulse_id_window):
        if not bool(self):
            # no data has been received yet
            return [], []

        n_steps = pulse_id_window // self._step_size
        if n_steps > self._max_num_steps:
            raise ValueError("Requested pulse_id window is larger than the maximum pulse_id span.")

        # add an extra point for bokeh Step to display the last value
        start_bin_id = self._start_bin_id - self._start_bin_id % n_steps
        x = np.arange(start_bin_id, self._stop_bin_id + 1, n_steps)
        y = np.zeros_like(x, dtype=np.float64)

        for ind, _bin_id in enumerate(x):
            sum_sig_lon = sum_bkg_lon = sum_sig_loff = sum_bkg_loff = 0
            for shift in range(n_steps):
                sum_sig_lon += self._sig_lon[_bin_id + shift]
                sum_bkg_lon += self._bkg_lon[_bin_id + shift]
                sum_sig_loff += self._sig_loff[_bin_id + shift]
                sum_bkg_loff += self._bkg_loff[_bin_id + shift]

            if sum_bkg_lon and sum_sig_loff and sum_bkg_loff:
                y[ind] = (sum_sig_lon / sum_bkg_lon) / (sum_sig_loff / sum_bkg_loff) - 1

        return x * self._step_size, y

    def clear(self):
        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._sig_lon.clear()
        self._bkg_lon.clear()
        self._sig_loff.clear()
        self._bkg_loff.clear()


class Intensities:
    def __init__(self, step_size=100, max_span=120_000):
        self._step_size = step_size
        self._max_num_steps = max_span // step_size

        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._len = 0
        self._I = defaultdict(partial(np.zeros, shape=10))
        self._n_I = defaultdict(partial(np.zeros, shape=10))

    def __bool__(self):
        return bool(self._len)

    @property
    def step_size(self):
        return self._step_size

    @property
    def max_span(self):
        return self._step_size * self._max_num_steps

    def update(self, pulse_id, values):
        bin_id = pulse_id // self._step_size

        if self._start_bin_id == -1:
            self._start_bin_id = bin_id

        if bin_id < self._start_bin_id:
            # the data is too old
            return

        min_bin_id = max(bin_id - self._max_num_steps, 0)
        if self._start_bin_id < min_bin_id:
            # update start_bin_id and drop old data from the buffers
            for _bin_id in range(self._start_bin_id, min_bin_id):
                self._I.pop(_bin_id, None)
                self._n_I.pop(_bin_id, None)

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the buffers
        self._len = len(values)
        self._I[bin_id][: self._len] += values
        self._n_I[bin_id][: self._len] += 1

    def __call__(self):
        if not bool(self):
            # no data has been received yet
            return [], [[]]

        x = np.arange(self._start_bin_id, self._stop_bin_id)
        ys = np.zeros(shape=(self._len, len(x)), dtype=np.float64)

        for ind, bin_id in enumerate(x):
            with np.errstate(invalid="ignore"):
                # it's ok here to divide by 0 and get np.nan as a result
                ys[:, ind] += self._I[bin_id][: self._len] / self._n_I[bin_id][: self._len]

        return x * self._step_size, ys

    def clear(self):
        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._len = 0
        self._I.clear()
        self._n_I.clear()
