import logging

import numpy as np
from bokeh.models import CustomJS, Dropdown

from streamvis.statistics_tools import AggregatorWithID, NPFIFOArray

logger = logging.getLogger(__name__)


class CBDStatisticsHandler:
    def __init__(self):
        """Initialize a statistics handler specific for CBD experiments.

        Statistics collected:
            - Number of streaks detected;
            - Length of streaks detected;
            - Bragg Intensity;
            - Background count (Total intensity - Bragg intensity);

        """
        self.number_of_streaks = NPFIFOArray(dtype=int, empty_value=-1, max_span=5_000)
        self.streak_lengths = NPFIFOArray(dtype=float, empty_value=np.nan, max_span=50_000)
        self.bragg_counts = NPFIFOArray(
            dtype=float, empty_value=np.nan, max_span=50_000, aggregate=np.sum
        )
        self.bragg_aggregator = AggregatorWithID(dtype=float, empty_value=np.nan, max_span=750_000)

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
            label="Open Auxiliary App", menu=["Convergent Beam Diffraction stats"], width=165
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

        if image.shape == (2, 2):
            logger.debug(f"Dummy, skipping")
            return

        # Update Bragg aggregator with hits and non-hits alike
        bragg_counts: list[float] = metadata.get("bragg_counts", [0])
        pulse_id = metadata.get("pulse_id", None)
        self.bragg_aggregator.update(np.array(bragg_counts), pulse_id)

        if not is_hit_frame:
            logger.debug(f"Not hit frame, skipping")
            return

        self.bragg_counts.update(np.array([np.sum(bragg_counts)]))

        number_of_streaks: int = metadata.get("number_of_streaks", 0)
        self.number_of_streaks.update(np.array([number_of_streaks]))

        streak_lengths: list[float] = metadata.get("streak_lengths", [0])
        self.streak_lengths.update(np.array(streak_lengths))

    def reset(self):
        self.number_of_streaks.clear()
        self.streak_lengths.clear()
        self.bragg_counts.clear()
        self.bragg_aggregator.clear()
