import copy
from collections import deque
from threading import RLock

import numpy as np
from bokeh.models import Button, CustomJS
from jungfrau_utils import StreamAdapter


class StatisticsHandler:
    def __init__(self, hit_threshold, buffer_size=1):
        """Initialize a statistics handler.

        Args:
            hit_threshold (int): A number of spots, above which a shot is registered as 'hit'.
            buffer_size (int, optional): A peakfinder buffer size. Defaults to 1.
        """
        self.hit_threshold = hit_threshold
        self.current_run_name = None
        self.expected_nframes = None
        self.received_nframes = None
        self.last_hit = (None, None)
        self.peakfinder_buffer = deque(maxlen=buffer_size)
        self.hitrate_buffer_fast = deque(maxlen=50)
        self.hitrate_buffer_slow = deque(maxlen=500)
        # TODO: fix maximum number of deques in the buffer
        self.roi_intensities_buffers = [deque(maxlen=50) for _ in range(9)]
        self._lock = RLock()

        self.jf_adapter = StreamAdapter()

        self.data = dict(
            run_names=[],
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
            if key == "run_names":
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
        is_hit = number_of_spots and number_of_spots > self.hit_threshold

        run_name = metadata.get("run_name")
        if run_name:
            with self._lock:
                if run_name != self.current_run_name:
                    self.peakfinder_buffer.clear()
                    self.current_run_name = run_name
                    for key, val in self.data.items():
                        if key == "run_names":
                            val.append(run_name)
                        else:
                            val.append(0)

                swissmx_x = metadata.get("swissmx_x")
                swissmx_y = metadata.get("swissmx_y")
                frame = metadata.get("frame")
                if swissmx_x and swissmx_y and frame and number_of_spots:
                    self.peakfinder_buffer.append(
                        np.array([swissmx_x, swissmx_y, frame, number_of_spots])
                    )

                self._increment("nframes")

                if "is_good_frame" in metadata and not metadata["is_good_frame"]:
                    self._increment("bad_frames")

                if "saturated_pixels" in metadata:
                    if metadata["saturated_pixels"] != 0:
                        self._increment("sat_pix_nframes")
                else:
                    self.data["sat_pix_nframes"][-1] = np.nan

                laser_on = metadata.get("laser_on")
                if laser_on is not None:
                    switch = "laser_on" if laser_on else "laser_off"

                    self._increment(f"{switch}_nframes")

                    if is_hit:
                        self._increment(f"{switch}_hits")

                    self.data[f"{switch}_hits_ratio"][-1] = (
                        self.data[f"{switch}_hits"][-1] / self.data[f"{switch}_nframes"][-1]
                    )
                    self.sum_data[f"{switch}_hits_ratio"][-1] = (
                        self.sum_data[f"{switch}_hits"][-1] / self.sum_data[f"{switch}_nframes"][-1]
                    )
                else:
                    self.data["laser_on_nframes"][-1] = np.nan
                    self.data["laser_on_hits"][-1] = np.nan
                    self.data["laser_on_hits_ratio"][-1] = np.nan
                    self.data["laser_off_nframes"][-1] = np.nan
                    self.data["laser_off_hits"][-1] = np.nan
                    self.data["laser_off_hits_ratio"][-1] = np.nan

        self.expected_nframes = metadata.get("number_frames_expected")
        if self.data["nframes"]:
            self.received_nframes = self.data["nframes"][-1]

        if is_hit:
            self.last_hit = (metadata, image)
            self.hitrate_buffer_fast.append(1)
            self.hitrate_buffer_slow.append(1)
        else:
            self.hitrate_buffer_fast.append(0)
            self.hitrate_buffer_slow.append(0)

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

    def _increment(self, key):
        self.data[key][-1] += 1
        self.sum_data[key][-1] += 1

    def reset(self):
        """Reset statistics entries.
        """
        with self._lock:
            self.current_run_name = None

            for val in self.data.values():
                val.clear()

            for key, val in self.sum_data.items():
                if key != "run_names":
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
