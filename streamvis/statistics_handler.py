import copy
from collections import Counter, defaultdict, deque
from functools import partial
from threading import RLock

import numpy as np
from bokeh.models import CustomJS, Dropdown

PULSE_ID_STEP = 10000


class StatisticsHandler:
    def __init__(self, hit_threshold, buffer_size=1):
        """Initialize a statistics handler.

        Args:
            hit_threshold (int): A number of spots, above which a shot is registered as 'hit'.
            buffer_size (int, optional): A peakfinder buffer size. Defaults to 1.
        """
        self.hit_threshold = hit_threshold
        self.last_hit = (None, None)
        self.peakfinder_buffer = deque(maxlen=buffer_size)
        self.hitrate_fast = Hitrate(step_size=100)
        self.hitrate_fast_lon = Hitrate(step_size=100)
        self.hitrate_fast_loff = Hitrate(step_size=100)
        self.hitrate_slow = Hitrate(step_size=1000)
        self.hitrate_slow_lon = Hitrate(step_size=1000)
        self.hitrate_slow_loff = Hitrate(step_size=1000)
        # TODO: fix maximum number of deques in the buffer
        self.roi_intensities = Intensities()
        self.roi_pump_probe = PumpProbe()
        self.roi_pump_probe_nobkg = PumpProbe_nobkg()
        self.radial_profile_lon = RadialProfile()
        self.radial_profile_loff = RadialProfile()
        self._lock = RLock()

        self.data = dict(
            pulse_id_bins=[],
            nframes=[],
            bad_frames=[],
            sat_pix_nframes=[],
            laser_on_nframes=[],
            laser_on_hits=[],
            laser_on_hits_ratio=[],
            laser_off_nframes=[],
            laser_off_hits=[],
            laser_off_hits_ratio=[],
        )

        self.sum_data = copy.deepcopy(self.data)
        for key, val in self.sum_data.items():
            if key == "pulse_id_bins":
                val.append("Summary")
            else:
                val.append(0)

    @property
    def auxiliary_apps_dropdown(self):
        """Return a button that opens statistics application.
        """
        js_code = """
        switch (this.item) {
            case "Statistics":
                window.open('/statistics');
                break;
            case "Hitrate":
                window.open('/hitrate');
                break;
            case "ROI Intensities":
                window.open('/roi_intensities');
                break;
            case "ROI Pump-Probe":
                window.open('/roi_pump_probe');
                break;
            case "Radial Profile":
                window.open('/radial_profile');
                break;
        }
        """
        auxiliary_apps_dropdown = Dropdown(
            label="Open Auxiliary App",
            menu=["Statistics", "Hitrate", "ROI Intensities", "ROI Pump-Probe", "Radial Profile"],
            default_size=145,
        )
        auxiliary_apps_dropdown.js_on_click(CustomJS(code=js_code))

        return auxiliary_apps_dropdown

    def parse(self, metadata, image):
        """Extract statistics from a metadata and an associated image.

        Args:
            metadata (dict): A dictionary with metadata.
            image (ndarray): An associated image.
        """
        number_of_spots = metadata.get("number_of_spots")
        sfx_hit = metadata.get("sfx_hit")
        if sfx_hit is None:
            sfx_hit = number_of_spots and number_of_spots > self.hit_threshold

        if image.shape != (2, 2) and sfx_hit:
            # add to buffer only if the recieved image is not dummy
            self.last_hit = (metadata, image)

        pulse_id = metadata.get("pulse_id")
        if pulse_id is None:
            # no further statistics is possible to collect
            return

        self.hitrate_fast.update(pulse_id, sfx_hit)
        self.hitrate_slow.update(pulse_id, sfx_hit)
        laser_on = metadata.get("laser_on")
        if laser_on is not None:
            if laser_on:
                self.hitrate_fast_lon.update(pulse_id, sfx_hit)
                self.hitrate_slow_lon.update(pulse_id, sfx_hit)
            else:
                self.hitrate_fast_loff.update(pulse_id, sfx_hit)
                self.hitrate_slow_loff.update(pulse_id, sfx_hit)

            radint_q = metadata.get("radint_q")
            if radint_q is not None:
                self.radial_profile_lon.update_q(radint_q)
                self.radial_profile_loff.update_q(radint_q)

            radint_I = metadata.get("radint_I")
            if radint_I is not None:
                if laser_on:
                    self.radial_profile_lon.update_I(pulse_id, radint_I)
                else:
                    self.radial_profile_loff.update_I(pulse_id, radint_I)

        roi_intensities = metadata.get("roi_intensities_normalised")
        if roi_intensities is not None:
            self.roi_intensities.update(pulse_id, roi_intensities)
            if len(roi_intensities) >= 1:
                self.roi_pump_probe_nobkg.update(pulse_id, laser_on, sig=roi_intensities[0])
            if len(roi_intensities) >= 2:
                self.roi_pump_probe.update(
                    pulse_id, laser_on, sig=roi_intensities[0], bkg=roi_intensities[1]
                )

        pulse_id_bin = pulse_id // PULSE_ID_STEP * PULSE_ID_STEP
        with self._lock:
            try:
                # since messages can have mixed pulse_id order, search for the current pulse_id_bin
                # in the last 5 entries (5 should be large enough for all data analysis to finish)
                bin_ind = self.data["pulse_id_bins"].index(pulse_id_bin, -5)
            except ValueError:
                # this is a new bin
                bin_ind = -1

            if bin_ind == -1:
                self.peakfinder_buffer.clear()
                for key, val in self.data.items():
                    if key == "pulse_id_bins":
                        val.append(pulse_id_bin)
                    else:
                        val.append(0)

            swissmx_x = metadata.get("swissmx_x")
            swissmx_y = metadata.get("swissmx_y")
            frame = metadata.get("frame")
            if swissmx_x and swissmx_y and frame and number_of_spots:
                self.peakfinder_buffer.append(
                    np.array([swissmx_x, swissmx_y, frame, number_of_spots])
                )

            self._increment("nframes", bin_ind)

            if "is_good_frame" in metadata and not metadata["is_good_frame"]:
                self._increment("bad_frames", bin_ind)

            if "saturated_pixels" in metadata:
                if metadata["saturated_pixels"] != 0:
                    self._increment("sat_pix_nframes", bin_ind)
            else:
                self.data["sat_pix_nframes"][bin_ind] = np.nan

            if laser_on is not None:
                switch = "laser_on" if laser_on else "laser_off"

                self._increment(f"{switch}_nframes", bin_ind)

                if sfx_hit:
                    self._increment(f"{switch}_hits", bin_ind)

                self.data[f"{switch}_hits_ratio"][bin_ind] = (
                    self.data[f"{switch}_hits"][bin_ind] / self.data[f"{switch}_nframes"][bin_ind]
                )
                self.sum_data[f"{switch}_hits_ratio"][-1] = (
                    self.sum_data[f"{switch}_hits"][-1] / self.sum_data[f"{switch}_nframes"][-1]
                )
            else:
                self.data["laser_on_nframes"][bin_ind] = np.nan
                self.data["laser_on_hits"][bin_ind] = np.nan
                self.data["laser_on_hits_ratio"][bin_ind] = np.nan
                self.data["laser_off_nframes"][bin_ind] = np.nan
                self.data["laser_off_hits"][bin_ind] = np.nan
                self.data["laser_off_hits_ratio"][bin_ind] = np.nan

    def _increment(self, key, ind):
        self.data[key][ind] += 1
        self.sum_data[key][-1] += 1

    def reset(self):
        """Reset statistics entries.
        """
        with self._lock:
            for val in self.data.values():
                val.clear()

            for key, val in self.sum_data.items():
                if key != "pulse_id_bins":
                    val[0] = 0


class Hitrate:
    def __init__(self, step_size=100, max_span=120_000):
        self._step_size = step_size
        self._max_num_steps = max_span // step_size

        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._hits = Counter()
        self._empty = Counter()

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
            # update start_bin_id and drop old data from the counters
            for _bin_id in range(self._start_bin_id, min_bin_id):
                del self._hits[_bin_id]
                del self._empty[_bin_id]

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the counter
        counter = self._hits if is_hit else self._empty
        counter.update([bin_id])

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


class RadialProfile:
    def __init__(self, step_size=100, max_span=10_000):
        self._step_size = step_size
        self._max_num_steps = max_span // step_size

        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._q_limits = []
        self._q = np.zeros(1)
        self._I = defaultdict(int)
        self._n = defaultdict(int)

    def __bool__(self):
        return bool(self._I and self._n)

    @property
    def step_size(self):
        return self._step_size

    @property
    def max_span(self):
        return self._step_size * self._max_num_steps

    def update_q(self, q):
        if self._q_limits != q:
            self._q_limits = q
            self._q = np.arange(*q)
            self._I.clear()
            self._n.clear()

    def update_I(self, pulse_id, I):
        bin_id = pulse_id // self._step_size

        if self._start_bin_id == -1:
            self._start_bin_id = bin_id

        if bin_id < self._start_bin_id:
            # the data is too old
            return

        min_bin_id = max(bin_id - self._max_num_steps, 0)
        if self._start_bin_id < min_bin_id:
            # drop old data from the counters and update start_bin_id
            for _bin_id in range(self._start_bin_id, min_bin_id):
                self._I.pop(_bin_id, None)
                self._n.pop(_bin_id, None)

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the counters
        self._I[bin_id] += np.array(I)
        self._n[bin_id] += 1

    def __call__(self, pulse_id_window):
        if not bool(self):
            # no data has been received yet
            return [], [], 0

        n_steps = pulse_id_window // self._step_size
        if n_steps > self._max_num_steps:
            raise ValueError("Requested pulse_id window is larger than the maximum pulse_id span.")

        I_sum = 0
        n_sum = 0
        for _bin_id in range(self._stop_bin_id - n_steps, self._stop_bin_id):
            I_sum += self._I[_bin_id]
            n_sum += self._n[_bin_id]

        I_avg = I_sum / n_sum if n_sum != 0 else np.zeros_like(self._q)

        return self._q, I_avg, n_sum

    def clear(self):
        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._I.clear()
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
            # update start_bin_id and drop old data from the counters
            for _bin_id in range(self._start_bin_id, min_bin_id):
                del self._sig_lon[_bin_id]
                del self._sig_loff[_bin_id]
                del self._n_sig_lon[_bin_id]
                del self._n_sig_loff[_bin_id]

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the counters
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

            if sum_sig_loff:
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
            # update start_bin_id and drop old data from the counters
            for _bin_id in range(self._start_bin_id, min_bin_id):
                del self._sig_lon[_bin_id]
                del self._bkg_lon[_bin_id]
                del self._sig_loff[_bin_id]
                del self._bkg_loff[_bin_id]

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the counters
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

        self._n = 0
        self._intensities = defaultdict(partial(np.zeros, shape=9))

    def __bool__(self):
        return bool(self._intensities)

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
            # update start_bin_id and drop old data from the counters
            for _bin_id in range(self._start_bin_id, min_bin_id):
                del self._intensities[_bin_id]

            self._start_bin_id = min_bin_id

        if self._stop_bin_id < bin_id + 1:
            self._stop_bin_id = bin_id + 1

        # update the counters
        self._n = len(values)
        self._intensities[bin_id][: len(values)] += values

    def __call__(self):
        if not bool(self):
            # no data has been received yet
            return [], [[]]

        x = np.arange(self._start_bin_id, self._stop_bin_id)
        ys = np.zeros(shape=(self._n, len(x)), dtype=np.float64)

        for ind, bin_id in enumerate(x):
            ys[:, ind] += self._intensities[bin_id][: self._n]

        return x * self._step_size, ys

    def clear(self):
        self._start_bin_id = -1
        self._stop_bin_id = -1

        self._n = 0
        self._intensities.clear()
