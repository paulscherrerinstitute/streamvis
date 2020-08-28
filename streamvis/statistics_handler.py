import copy
from collections import deque, Counter
from threading import RLock

import numpy as np
from bokeh.models import Button, CustomJS
from jungfrau_utils import StreamAdapter

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
        self.hitrate_slow = Hitrate(step_size=1000)
        # TODO: fix maximum number of deques in the buffer
        self.roi_intensities_buffers = [deque(maxlen=50) for _ in range(9)]
        self._lock = RLock()

        self.jf_adapter = StreamAdapter()

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
    def open_stats_tab_button(self):
        """Return a button that opens statistics application.
        """
        open_stats_tab_button = Button(label="Open Statistics Tab")
        open_stats_tab_button.js_on_click(CustomJS(code="window.open('/statistics');"))

        return open_stats_tab_button

    @property
    def open_hitrate_plot_button(self):
        """Return a button that opens hitrate plot.
        """
        open_hitrate_plot_button = Button(label="Open Hitrate Plot")
        open_hitrate_plot_button.js_on_click(CustomJS(code="window.open('/hitrate');"))

        return open_hitrate_plot_button

    @property
    def open_roi_intensities_plot_button(self):
        """Return a button that opens ROI intensities application.
        """
        open_roi_intensities_plot_button = Button(label="Open ROI Intensities Plot")
        open_roi_intensities_plot_button.js_on_click(
            CustomJS(code="window.open('/roi_intensities');")
        )

        return open_roi_intensities_plot_button

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

        roi_intensities = metadata.get("roi_intensities_normalised")
        if roi_intensities is not None:
            for buf_ind, buffer in enumerate(self.roi_intensities_buffers):
                if buf_ind < len(roi_intensities):
                    buffer.append(roi_intensities[buf_ind])
                else:
                    buffer.clear()
        else:
            for buffer in self.roi_intensities_buffers:
                buffer.clear()

        pulse_id = metadata.get("pulse_id")
        if pulse_id is None:
            # no further statistics is possible to collect
            return

        self.hitrate_fast.update(pulse_id, sfx_hit)
        self.hitrate_slow.update(pulse_id, sfx_hit)

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

            laser_on = metadata.get("laser_on")
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

    def get_last_hit(self):
        """Get metadata and last hit image.
        """
        metadata, raw_image = self.last_hit
        image = self.jf_adapter.process(raw_image, metadata)

        if (
            self.jf_adapter.handler
            and "saturated_pixels" not in metadata
            and raw_image.dtype == np.uint16
        ):
            saturated_pixels_coord = self.jf_adapter.handler.get_saturated_pixels(
                raw_image, mask=True, gap_pixels=True, geometry=True
            )

            metadata["saturated_pixels_coord"] = saturated_pixels_coord
            metadata["saturated_pixels"] = len(saturated_pixels_coord[0])

        return metadata, image

    def get_last_hit_gains(self):
        """Get metadata and gains of last hit image.
        """
        metadata, image = self.last_hit
        if image.dtype != np.uint16:
            return metadata, image

        if self.jf_adapter.handler:
            image = self.jf_adapter.handler.get_gains(
                image, mask=False, gap_pixels=True, geometry=True
            )

        return metadata, image


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

    @property
    def values(self):
        # add an extra point for bokeh Step to display the last value
        x = np.arange(self._start_bin_id, self._stop_bin_id + 1)
        y = np.zeros_like(x, dtype=np.float)

        for i, _bin_id in enumerate(x):
            hits = self._hits[_bin_id]
            total = hits + self._empty[_bin_id]
            if total:
                y[i] = hits / total

        return x * self._step_size, y
