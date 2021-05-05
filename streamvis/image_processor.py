import numpy as np
from bokeh.models import Spinner, TextInput, Toggle


class ImageProcessor:
    def __init__(self):
        """Initialize an image processor.
        """
        self.aggregated_image = np.zeros((1, 1), dtype=np.float32)

        # Intensity threshold toggle
        def threshold_toggle_callback(state):
            self.threshold_toggle.button_type = "primary" if state else "default"

        threshold_toggle = Toggle(label="Apply Thresholding", active=False, button_type="default")
        threshold_toggle.on_click(threshold_toggle_callback)
        self.threshold_toggle = threshold_toggle

        # Threshold min/max value spinners
        self.threshold_min_spinner = Spinner(
            title="Min Intensity:", value=0, step=0.1, default_size=145
        )
        self.threshold_max_spinner = Spinner(
            title="Max Intensity:", value=1000, step=0.1, default_size=145
        )

        # Aggregation time toggle
        def aggregate_toggle_callback(state):
            self.aggregate_toggle.button_type = "primary" if state else "default"

        aggregate_toggle = Toggle(
            label="Apply Aggregation", active=False, button_type="default", default_size=145
        )
        aggregate_toggle.on_click(aggregate_toggle_callback)
        self.aggregate_toggle = aggregate_toggle

        # Aggregate time spinner
        self.aggregate_time_spinner = Spinner(
            title="Aggregate Time:", value=0, low=0, step=1, default_size=145
        )

        # Aggregate time counter textinput
        aggregate_time_counter_textinput = TextInput(
            title="Time Counter:", value=str(1), disabled=True, default_size=145
        )
        self.aggregate_time_counter_textinput = aggregate_time_counter_textinput

        # Show Average toggle
        def average_toggle_callback(state):
            self.average_toggle.button_type = "primary" if state else "default"

        average_toggle = Toggle(
            label="Show Average", active=False, button_type="default", default_size=145
        )
        average_toggle.on_click(average_toggle_callback)
        self.average_toggle = average_toggle

    @property
    def threshold_flag(self):
        """Threshold toggle state (readonly).
        """
        return self.threshold_toggle.active

    @property
    def threshold_min(self):
        """Minimal image threshold value (readonly).
        """
        return self.threshold_min_spinner.value

    @property
    def threshold_max(self):
        """Maximal image threshold value (readonly).
        """
        return self.threshold_max_spinner.value

    @property
    def aggregate_flag(self):
        """Aggregate toggle state (readonly).
        """
        return self.aggregate_toggle.active

    @property
    def aggregate_time(self):
        """A number of image aggregation before resetting (readonly).
        """
        return self.aggregate_time_spinner.value

    @property
    def aggregate_counter(self):
        """A current number of aggregated images (readonly).
        """
        return int(self.aggregate_time_counter_textinput.value)

    @aggregate_counter.setter
    def aggregate_counter(self, value):
        self.aggregate_time_counter_textinput.value = str(value)

    def update(self, image):
        """Trigger an update for the image processor.

        Args:
            image (ndarray): Input image to be processed.

        Returns:
            (ndarray, ndarray, bool): Resulting thresholding image, aggregated image and reset flag.
        """
        thr_image = image.copy()
        if self.threshold_flag:
            ind = (thr_image < self.threshold_min) | (self.threshold_max < thr_image)
            thr_image[ind] = 0

        if (
            self.aggregate_flag
            and (self.aggregate_time == 0 or self.aggregate_time > self.aggregate_counter)
            and self.aggregated_image.shape == image.shape
        ):
            self.aggregated_image += thr_image
            self.aggregate_counter += 1
            reset = False
        else:
            self.aggregated_image = thr_image
            self.aggregate_counter = 1
            reset = True

        if self.average_toggle.active:
            aggregated_image = self.aggregated_image / self.aggregate_counter
        else:
            aggregated_image = self.aggregated_image

        return thr_image, aggregated_image, reset
