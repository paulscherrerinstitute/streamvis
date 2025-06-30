import logging
import logging
import re
from collections import deque
from datetime import datetime
from threading import Thread

import numpy as np
import zmq
from bokeh.models import CustomJS, Dropdown
from jungfrau_utils import JFDataHandler
from numba import njit

from streamvis.cbd_statistic_tools import NPFIFOArray

PULSE_ID_STEP = 10000

logger = logging.getLogger(__name__)


class JFCBDAdapter:
    def __init__(self, buffer_size, io_threads, connection_mode, address):
        """Initialize a jungfrau adapter specific for Convergend Beam Diffraction experiment.

        The only difference from classic Jungfrau Adapter is the statistics handling

        Args:
            buffer_size (int): A number of last received zmq messages to keep in memory.
            io_threads (int): A size of the zmq thread pool to handle I/O operations.
            connection_mode (str): Use either 'connect' or 'bind' zmq_socket methods.
            address (str): The address string, e.g. 'tcp://127.0.0.1:9001'.
        """
        # a placeholder for jf data handler to be initiated with detector name
        self.handler = None

        self.buffer = deque(maxlen=buffer_size)
        # initialize buffer with dummy data
        self.buffer.append((dict(shape=[1, 1]), np.zeros((1, 1), dtype="float32")))
        self.state = "polling"

        # StatisticsHandler is used to parse metadata information to be displayed in 'statistics'
        # application. All messages are processed.
        self.stats = CBDStatisticsHandler()

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


class CBDStatisticsHandler:
    def __init__(self):
        """Initialize a statistics handler specific for CBD experiments.

        Statistics collected:
            - Number of streaks detected;
            - Length of streaks detected;
            - Bragg Intensity;
            - Background count (Total intensity - Bragg intensity);

        """
        self.streak_rate = NPFIFOArray(dtype=int, empty_value=-1, max_span=5_000)
        self.streak_length = NPFIFOArray(dtype=float, empty_value=np.nan, max_span=50_000)
        self.bragg_counts = NPFIFOArray(dtype=float, empty_value=np.nan, max_span=50_000, aggregate=np.sum)

    @property
    def auxiliary_apps_dropdown(self):
        """Return a button that opens statistics application."""
        js_code = """
        switch (this.item) {
            case "Convergent Beam Diffraction stats":
                window.open('/cbd_stats');
                break;
        }
        """
        auxiliary_apps_dropdown = Dropdown(
            label="Open Auxiliary App",
            menu=[
                "Convergent Beam Diffraction stats",
            ],
            width=165,
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

        if image.shape == (2, 2) or not is_hit_frame:
            print(f"Non-hit, skipping")
            return

        streak_rate: int = metadata.get("number_of_streaks")
        self.streak_rate.update(np.array([streak_rate]))

        streak_length: list[float] = metadata.get("streak_len")
        self.streak_length.update(np.array(streak_length))

        bragg_counts: list[float] = metadata.get("bragg_counts")
        self.bragg_counts.update(np.array(bragg_counts))

    def reset(self):
        self.streak_rate.clear()
        self.streak_length.clear()
        self.bragg_counts.clear()
