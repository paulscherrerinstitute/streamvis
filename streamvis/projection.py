import bottleneck as bn
import numpy as np
from bokeh.models import ColumnDataSource, DataRange1d
from bokeh.plotting import figure

DEFAULT_PLOT_SIZE = 200


class Projection:
    def __init__(self, image_view, direction, height=None, width=None):
        """Initialize a projection plot.

        Args:
            image_view (ImageView): Associated streamvis image view instance.
            direction (str): Display plot projection along the direction - 'horizontal', 'vertical'.
            height (int, optional): Height of plot area in screen pixels. Defaults to None.
            width (int, optional): Width of plot area in screen pixels. Defaults to None.

        Raises:
            ValueError: Projection direction can be either 'horizontal' or 'vertical'.
        """
        if direction not in ("horizontal", "vertical"):
            raise ValueError("Projection direction can be either 'horizontal' or 'vertical'")

        self._image_view = image_view
        self._direction = direction

        if direction == "vertical":
            if height is None:
                height = DEFAULT_PLOT_SIZE

            if width is None:
                width = image_view.plot.width

            plot = figure(
                y_axis_location="right",
                x_range=image_view.plot.x_range,
                y_range=DataRange1d(),
                height=height,
                width=width,
                toolbar_location=None,
            )

            plot.xaxis.major_label_text_font_size = "0pt"
            plot.yaxis.major_label_orientation = "vertical"

        elif direction == "horizontal":
            if height is None:
                height = image_view.plot.height

            if width is None:
                width = DEFAULT_PLOT_SIZE

            plot = figure(
                x_axis_location="above",
                x_range=DataRange1d(),
                y_range=image_view.plot.y_range,
                height=height,
                width=width,
                toolbar_location=None,
            )

            plot.yaxis.major_label_text_font_size = "0pt"

        # ---- line glyph
        self._line_source = ColumnDataSource(dict(x=[], y=[]))
        plot.line(source=self._line_source, x="x", y="y", line_color="steelblue")

        self.plot = plot

    @property
    def x(self):
        """Current x-axis values (readonly)."""
        return self._line_source.data["x"]

    @property
    def y(self):
        """Current y-axis values (readonly)."""
        return self._line_source.data["y"]

    def update(self, image):
        """Trigger an update for the projection plot.

        Args:
            image (ndarray): A source image for projection.
        """
        im_y_len, im_x_len = image.shape

        if self._direction == "vertical":
            x_val = np.linspace(
                self._image_view.x_start + 0.5, self._image_view.x_end - 0.5, im_x_len
            )
            y_val = bn.nanmean(image, axis=0)

        elif self._direction == "horizontal":
            x_val = bn.nanmean(image, axis=1)
            y_val = np.linspace(
                self._image_view.y_start + 0.5, self._image_view.y_end - 0.5, im_y_len
            )

        self._line_source.data.update(x=x_val, y=y_val)
