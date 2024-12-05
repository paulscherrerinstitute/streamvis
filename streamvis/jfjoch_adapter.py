# Based on https://github.com/dectris/documentation/blob/main/stream_v2/examples/client.py

import copy
import logging
import warnings
from collections import deque
from datetime import datetime
from threading import RLock, Thread

import cbor2
import numpy as np
import zmq
from bokeh.models import CustomJS, Dropdown
from dectris.compression import decompress
from numba import njit

from streamvis.statistics_tools import Hitrate, Intensities, Profile, PumpProbe, PumpProbe_nobkg

PULSE_ID_STEP = 10000

logger = logging.getLogger(__name__)


def decode_multi_dim_array(tag, order):
    dimensions, contents = tag.value
    if isinstance(contents, list):
        array = np.empty((len(contents),), dtype=object)
        array[:] = contents
    elif isinstance(contents, (np.ndarray, np.generic)):
        array = contents
    else:
        raise cbor2.CBORDecodeValueError("expected array or typed array")
    return array.reshape(dimensions, order=order)


def decode_typed_array(tag, dtype):
    if not isinstance(tag.value, bytes):
        raise cbor2.CBORDecodeValueError("expected byte string in typed array")
    return np.frombuffer(tag.value, dtype=dtype)


def decode_dectris_compression(tag):
    algorithm, elem_size, encoded = tag.value
    return decompress(encoded, algorithm, elem_size=elem_size)


tag_decoders = {
    40: lambda tag: decode_multi_dim_array(tag, order="C"),
    64: lambda tag: decode_typed_array(tag, dtype="u1"),
    65: lambda tag: decode_typed_array(tag, dtype=">u2"),
    66: lambda tag: decode_typed_array(tag, dtype=">u4"),
    67: lambda tag: decode_typed_array(tag, dtype=">u8"),
    68: lambda tag: decode_typed_array(tag, dtype="u1"),
    69: lambda tag: decode_typed_array(tag, dtype="<u2"),
    70: lambda tag: decode_typed_array(tag, dtype="<u4"),
    71: lambda tag: decode_typed_array(tag, dtype="<u8"),
    72: lambda tag: decode_typed_array(tag, dtype="i1"),
    73: lambda tag: decode_typed_array(tag, dtype=">i2"),
    74: lambda tag: decode_typed_array(tag, dtype=">i4"),
    75: lambda tag: decode_typed_array(tag, dtype=">i8"),
    77: lambda tag: decode_typed_array(tag, dtype="<i2"),
    78: lambda tag: decode_typed_array(tag, dtype="<i4"),
    79: lambda tag: decode_typed_array(tag, dtype="<i8"),
    80: lambda tag: decode_typed_array(tag, dtype=">f2"),
    81: lambda tag: decode_typed_array(tag, dtype=">f4"),
    82: lambda tag: decode_typed_array(tag, dtype=">f8"),
    83: lambda tag: decode_typed_array(tag, dtype=">f16"),
    84: lambda tag: decode_typed_array(tag, dtype="<f2"),
    85: lambda tag: decode_typed_array(tag, dtype="<f4"),
    86: lambda tag: decode_typed_array(tag, dtype="<f8"),
    87: lambda tag: decode_typed_array(tag, dtype="<f16"),
    1040: lambda tag: decode_multi_dim_array(tag, order="F"),
    56500: lambda tag: decode_dectris_compression(tag),
}


def tag_hook(decoder, tag):
    tag_decoder = tag_decoders.get(tag.tag)
    return tag_decoder(tag) if tag_decoder else tag


class JFJochAdapter:
    def __init__(self, buffer_size, io_threads, connection_mode, address):
        """Initialize a jungfrau adapter.

        Args:
            buffer_size (int): A number of last received zmq messages to keep in memory.
            io_threads (int): A size of the zmq thread pool to handle I/O operations.
            connection_mode (str): Use either 'connect' or 'bind' zmq_socket methods.
            address (str): The address string, e.g. 'tcp://127.0.0.1:9001'.
        """
        self.handler = None
        self.pixel_mask = None
        self.start_message_metadata = {}

        self.buffer = deque(maxlen=buffer_size)
        # initialize buffer with dummy data
        self.buffer.append((dict(shape=[1, 1]), np.zeros((1, 1), dtype="float32")))
        self.state = "polling"

        # StatisticsHandler is used to parse metadata information to be displayed in 'statistics'
        # application. All messages are processed.
        self.stats = StatisticsHandler()

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
            message = zmq_socket.recv(flags=0, copy=False, track=False)
            metadata = cbor2.loads(message, tag_hook=tag_hook)
            if metadata["type"] == "start":
                self.start_message_metadata = metadata
                self.pixel_mask = np.invert(
                    metadata["pixel_mask"]["default"].astype(bool, copy=True)
                )
                image = np.zeros((2, 2), dtype="float32")
                self.stats.parse(metadata, image)

            elif metadata["type"] == "image":
                # Merge with "start" message metadata
                metadata.update(self.start_message_metadata)

                image = metadata["data"]["default"]
                metadata["time_poll"] = time_poll
                metadata["time_recv"] = datetime.now() - time_poll

                # add to buffer only if the recieved image is not dummy
                if image.shape != (2, 2):
                    self.buffer.append((metadata, image))

                self.state = "receiving"
                self.stats.parse(metadata, image)

            else:
                warnings.warn(f"Unhandled message type: {metadata['type']}")

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
        if image.dtype != np.float32:
            image = image.astype(np.float32, copy=True)

        if mask and self.pixel_mask is not None:
            image = self._apply_mask(image)

        return image

    def _apply_mask(self, image):
        # assign masked values to np.nan
        if image.shape == self.pixel_mask.shape:
            _apply_mask_njit(image, self.pixel_mask)
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
        is_hit_frame = metadata.get("indexing_result", False)

        if image.shape != (2, 2) and is_hit_frame:
            # add to buffer only if the recieved image is not dummy
            self.last_hit = (metadata, image)

        pulse_id = metadata.get("xfel_pulse_id")
        if pulse_id is None:
            # no further statistics is possible to collect
            return

        self.hitrate_fast.update(pulse_id, is_hit_frame)
        self.hitrate_slow.update(pulse_id, is_hit_frame)
        laser_on = metadata.get("laser_on")
        if laser_on:
            self.hitrate_fast_lon.update(pulse_id, is_hit_frame)
            self.hitrate_slow_lon.update(pulse_id, is_hit_frame)
        else:
            self.hitrate_fast_loff.update(pulse_id, is_hit_frame)
            self.hitrate_slow_loff.update(pulse_id, is_hit_frame)

        radint_q = metadata.get("az_int_bin_to_q")
        if radint_q is not None:
            self.radial_profile_lon.update_x(radint_q)
            self.radial_profile_loff.update_x(radint_q)

        radint_I = metadata.get("az_int_profile")
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

        roi_integrals = metadata.get("roi_integrals")
        if roi_integrals is not None:
            roi_labels = []
            roi_intensities = []
            for roi_label, roi_intensity in roi_integrals.items():
                roi_intensities.append(roi_intensity["sum"])
                roi_labels.append(roi_label)
            self.roi_labels = roi_labels

            if roi_intensities:
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
