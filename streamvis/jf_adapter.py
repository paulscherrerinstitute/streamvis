import copy
import logging
import re
from collections import deque
from datetime import datetime
from threading import RLock, Thread

import numpy as np
import zmq
from bokeh.models import CustomJS, Dropdown
from jungfrau_utils import JFDataHandler
from numba import njit

from streamvis.statistics_tools import Hitrate, Intensities, Profile, PumpProbe, PumpProbe_nobkg

PULSE_ID_STEP = 10000

logger = logging.getLogger(__name__)


class JFAdapter:
    def __init__(self, buffer_size, io_threads, connection_mode, address, handler_cls):
        """Initialize a jungfrau adapter.

        Args:
            buffer_size (int): A number of last received zmq messages to keep in memory.
            io_threads (int): A size of the zmq thread pool to handle I/O operations.
            connection_mode (str): Use either 'connect' or 'bind' zmq_socket methods.
            address (str): The address string, e.g. 'tcp://127.0.0.1:9001'.
            handler_cls (class): Class for statistics handler
        """
        # a placeholder for jf data handler to be initiated with detector name
        self.handler = None

        self.buffer = deque(maxlen=buffer_size)
        # initialize buffer with dummy data
        self.buffer.append((dict(shape=[1, 1]), np.zeros((1, 1), dtype="float32")))
        self.state = "polling"

        # StatisticsHandler is used to parse metadata information to be displayed in 'statistics'
        # application. All messages are processed.
        self.stats = handler_cls()

        # Start receiving messages in a separate thread
        t = Thread(target=self.start, args=(io_threads, connection_mode, address), daemon=True)
        t.start()

    def start(self, io_threads, connection_mode, address):
        """Start a receiver loop.

        Args:
            io_threads (int): The size of the zmq thread pool to handle I/O operations.
            connection_mode (str): Use either 'connect' or 'bind' zmq_socket methods.
            address (str): The address string, e.g. 'tcp://127.0.0.1:9001'.

        Raises:
            RuntimeError: Unknown connection mode.
        """
        zmq_context = zmq.Context(io_threads=io_threads)
        zmq_socket = zmq_context.socket(zmq.SUB)  # pylint: disable=E1101
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # pylint: disable=E1101

        if connection_mode == "connect":
            zmq_socket.connect(address)
        elif connection_mode == "bind":
            zmq_socket.bind(address)
        else:
            raise RuntimeError("Unknown connection mode {connection_mode}")

        poller = zmq.Poller()
        poller.register(zmq_socket, zmq.POLLIN)

        while True:
            events = dict(poller.poll(1000))
            if zmq_socket not in events:
                self.state = "polling"
                continue

            time_poll = datetime.now()
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=False, track=False)
            metadata["time_poll"] = time_poll
            metadata["time_recv"] = datetime.now() - time_poll

            dtype = metadata.get("type")
            shape = metadata.get("shape")
            if dtype is None or shape is None:
                logger.error("Cannot find 'type' and/or 'shape' in received metadata")
                continue

            image = np.frombuffer(image.buffer, dtype=dtype).reshape(shape)

            # add to buffer only if the recieved image is not dummy
            if image.shape != (2, 2):
                self.buffer.append((metadata, image))

            self.state = "receiving"

            self.stats.parse(metadata, image)

    def process(
        self, image, metadata, mask=True, gap_pixels=True, double_pixels="keep", geometry=True
    ):
        """Perform jungfrau detector data processing on an image received via stream.

        Args:
            image (ndarray): An image to be processed.
            metadata (dict): A corresponding image metadata.

        Returns:
            ndarray: Resulting image.
        """
        # Eiger workaround
        detector_name = metadata.get("detector_name")
        if detector_name and re.match("(^[A-Za-z]*).EG([0-9A-Za-z]*)", detector_name):
            return np.copy(np.flipud(image))

        # parse metadata
        self._update_handler(metadata)

        if self.handler is None or image.dtype != np.uint16:
            return image.astype(np.float32, copy=True)

        # skip conversion step if jungfrau data handler cannot do it, thus avoiding Exception raise
        conversion = self.handler.can_convert()
        proc_image = self.handler.process(
            image,
            conversion=conversion,
            mask=False,
            gap_pixels=gap_pixels,
            double_pixels=double_pixels,
            geometry=geometry,
        )

        if mask:
            proc_image = self._apply_mask(proc_image, gap_pixels, double_pixels, geometry)

        return proc_image

    def get_gains(
        self, image, metadata, mask=True, gap_pixels=True, double_pixels="keep", geometry=True
    ):
        # parse metadata
        self._update_handler(metadata)

        if self.handler is None or image.dtype != np.uint16:
            return np.zeros((2, 2), dtype="float32")

        gains = self.handler.get_gains(
            image, mask=mask, gap_pixels=gap_pixels, double_pixels=double_pixels, geometry=geometry
        )

        if mask:
            if double_pixels == "interp":
                double_pixels = "keep"
            gains = self._apply_mask(gains, gap_pixels, double_pixels, geometry)

        return gains

    def _update_handler(self, metadata):
        # as a first step, try to set the detector_name, skip if detector_name is empty
        detector_name = metadata.get("detector_name")
        if detector_name and detector_name.startswith("JF"):
            # check if jungfrau data handler is already set for this detector
            if self.handler is None or self.handler.detector_name != detector_name:
                try:
                    self.handler = JFDataHandler(detector_name)
                except ValueError:
                    logging.exception(f"Error creating data handler for detector {detector_name}")
                    self.handler = None
                    return
        else:
            self.handler = None
            return

        # gain file
        gain_file = metadata.get("gain_file", "")
        try:
            self.handler.gain_file = gain_file
        except Exception:
            logging.exception(f"Error loading gain file {gain_file}")
            self.handler.gain_file = ""

        # pedestal file
        pedestal_file = metadata.get("pedestal_file", "")
        try:
            self.handler.pedestal_file = pedestal_file
        except Exception:
            logging.exception(f"Error loading pedestal file {pedestal_file}")
            self.handler.pedestal_file = ""

        # module map
        module_map = metadata.get("module_map")
        self.handler.module_map = None if (module_map is None) else np.array(module_map)

        # highgain
        daq_rec = metadata.get("daq_rec")
        self.handler.highgain = False if (daq_rec is None) else bool(daq_rec & 0b1)

    def _apply_mask(self, image, gap_pixels, double_pixels, geometry):
        # assign masked values to np.nan
        if self.handler.pixel_mask is not None:
            mask = self.handler.get_pixel_mask(
                gap_pixels=gap_pixels, double_pixels=double_pixels, geometry=geometry
            )

            # cast to np.float32 in case there was no conversion, but mask should still be applied
            if image.dtype != np.float32:
                image = image.astype(np.float32)

            if image.shape == mask.shape:
                _apply_mask_njit(image, mask)
            else:
                raise ValueError("Image and mask shapes are not the same")

        return image


@njit
def _apply_mask_njit(image, mask):
    sy, sx = image.shape
    for i in range(sx):
        for j in range(sy):
            if not mask[j, i]:
                image[j, i] = np.nan


class StatisticsHandler:
    def __init__(self):
        """Initialize a statistics handler.

        Args:
            buffer_size (int, optional): A peakfinder buffer size. Defaults to 1.
        """
        self.last_hit = (dict(shape=[1, 1]), np.zeros((1, 1), dtype="float32"))
        self.hitrate_fast = Hitrate(step_size=100)
        self.hitrate_fast_lon = Hitrate(step_size=100)
        self.hitrate_fast_loff = Hitrate(step_size=100)
        self.hitrate_slow = Hitrate(step_size=1000)
        self.hitrate_slow_lon = Hitrate(step_size=1000)
        self.hitrate_slow_loff = Hitrate(step_size=1000)
        self.roi_labels = []
        self.roi_intensities = Intensities()
        self.roi_intensities_fast = Intensities(step_size=1, max_span=1200)
        self.roi_intensities_hit = Intensities()
        self.roi_intensities_hit_fast = Intensities(step_size=1, max_span=1200)
        self.roi_intensities_nohit = Intensities()
        self.roi_intensities_nohit_fast = Intensities(step_size=1, max_span=1200)
        self.roi_pump_probe = PumpProbe()
        self.roi_pump_probe_nobkg = PumpProbe_nobkg()
        self.radial_profile_lon = Profile()
        self.radial_profile_loff = Profile()
        self.projections_lon = [Profile() for _ in range(10)]
        self.projections_loff = [Profile() for _ in range(10)]
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
        """Return a button that opens statistics application."""
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
            case "ROI Projections":
                window.open('/roi_projections');
                break;
            case "Radial Profile":
                window.open('/radial_profile');
                break;
        }
        """
        auxiliary_apps_dropdown = Dropdown(
            label="Open Auxiliary App",
            menu=[
                "Statistics",
                "Hitrate",
                "ROI Intensities",
                "ROI Pump-Probe",
                "ROI Projections",
                "Radial Profile",
            ],
            width=145,
        )
        auxiliary_apps_dropdown.js_on_click(CustomJS(code=js_code))

        return auxiliary_apps_dropdown

    def parse(self, metadata, image):
        """Extract statistics from a metadata and an associated image.

        Args:
            metadata (dict): A dictionary with metadata.
            image (ndarray): An associated image.
        """
        is_hit_frame = metadata.get("is_hit_frame", False)

        if image.shape != (2, 2) and is_hit_frame:
            # add to buffer only if the recieved image is not dummy
            self.last_hit = (metadata, image)

        pulse_id = metadata.get("pulse_id")
        if pulse_id is None:
            # no further statistics is possible to collect
            return

        self.hitrate_fast.update(pulse_id, is_hit_frame)
        self.hitrate_slow.update(pulse_id, is_hit_frame)
        laser_on = metadata.get("laser_on")
        if laser_on is not None:
            if laser_on:
                self.hitrate_fast_lon.update(pulse_id, is_hit_frame)
                self.hitrate_slow_lon.update(pulse_id, is_hit_frame)
            else:
                self.hitrate_fast_loff.update(pulse_id, is_hit_frame)
                self.hitrate_slow_loff.update(pulse_id, is_hit_frame)

            radint_q = metadata.get("radint_q")
            if radint_q is not None:
                self.radial_profile_lon.update_x(radint_q)
                self.radial_profile_loff.update_x(radint_q)

            radint_I = metadata.get("radint_I")
            if radint_I is not None:
                if laser_on:
                    self.radial_profile_lon.update_y(pulse_id, radint_I)
                else:
                    self.radial_profile_loff.update_y(pulse_id, radint_I)

            roi_intensities_x = metadata.get("roi_intensities_x")
            if roi_intensities_x is not None:
                for ind, x in enumerate(roi_intensities_x):
                    self.projections_lon[ind].update_x(x)
                    self.projections_loff[ind].update_x(x)

            roi_intensities_proj_x = metadata.get("roi_intensities_proj_x")
            if roi_intensities_proj_x is not None:
                projections = self.projections_lon if laser_on else self.projections_loff
                for projection, proj_I in zip(projections, roi_intensities_proj_x):
                    projection.update_y(pulse_id, proj_I)

            roi_intensities = metadata.get("roi_intensities_normalised")
            if roi_intensities is not None:
                self.roi_labels = [f"ROI_{i}" for i in range(len(roi_intensities))]
                self.roi_intensities.update(pulse_id, roi_intensities)
                self.roi_intensities_fast.update(pulse_id, roi_intensities)
                if is_hit_frame:
                    self.roi_intensities_hit.update(pulse_id, roi_intensities)
                    self.roi_intensities_hit_fast.update(pulse_id, roi_intensities)
                else:
                    self.roi_intensities_nohit.update(pulse_id, roi_intensities)
                    self.roi_intensities_nohit_fast.update(pulse_id, roi_intensities)

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
                for key, val in self.data.items():
                    if key == "pulse_id_bins":
                        val.append(pulse_id_bin)
                    else:
                        val.append(0)

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

                if is_hit_frame:
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
        """Reset statistics entries."""
        with self._lock:
            for val in self.data.values():
                val.clear()

            for key, val in self.sum_data.items():
                if key != "pulse_id_bins":
                    val[0] = 0
