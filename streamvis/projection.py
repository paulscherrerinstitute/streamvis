import bottleneck as bn
import numpy as np
from bokeh.models import BasicTicker, ColumnDataSource, DataRange1d, Grid, Line, LinearAxis, Plot

DEFAULT_PLOT_SIZE = 200


class Projection:
    def __init__(self, image_view, direction, plot_height=None, plot_width=None):
        """Initialize a projection plot.

        Args:
            image_view (ImageView): Associated streamvis image view instance.
            direction (str): Display plot projection along the direction - 'horizontal', 'vertical'.
            plot_height (int, optional): Height of plot area in screen pixels. Defaults to None.
            plot_width (int, optional): Width of plot area in screen pixels. Defaults to None.

        Raises:
            ValueError: Projection direction can be either 'horizontal' or 'vertical'.
        """
        if direction not in ("horizontal", "vertical"):
            raise ValueError("Projection direction can be either 'horizontal' or 'vertical'")

        self._image_view = image_view
        self._direction = direction

        if direction == "vertical":
            if plot_height is None:
                plot_height = DEFAULT_PLOT_SIZE

            if plot_width is None:
                plot_width = image_view.plot.plot_width

            plot = Plot(
                x_range=image_view.plot.x_range,
                y_range=DataRange1d(),
                plot_height=plot_height,
                plot_width=plot_width,
                toolbar_location=None,
            )

            # ---- axes
            plot.add_layout(LinearAxis(major_label_orientation="vertical"), place="right")
            plot.add_layout(LinearAxis(major_label_text_font_size="0pt"), place="below")

        elif direction == "horizontal":
            if plot_height is None:
                plot_height = image_view.plot.plot_height

            if plot_width is None:
                plot_width = DEFAULT_PLOT_SIZE

            plot = Plot(
                x_range=DataRange1d(),
                y_range=image_view.plot.y_range,
                plot_height=plot_height,
                plot_width=plot_width,
                toolbar_location=None,
            )

            # ---- axes
            plot.add_layout(LinearAxis(), place="above")
            plot.add_layout(LinearAxis(major_label_text_font_size="0pt"), place="left")

        # ---- grid lines
        plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
        plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

        # ---- line glyph
        self._line_source = ColumnDataSource(dict(x=[], y=[]))
        plot.add_glyph(self._line_source, Line(x="x", y="y", line_color="steelblue"))

        self.plot = plot

    @property
    def x(self):
        """Current x-axis values (readonly).
        """
        return self._line_source.data["x"]

    @property
    def y(self):
        """Current y-axis values (readonly).
        """
        return self._line_source.data["y"]

    def update(self, image):
        """Trigger an update for the projection plot.

        Args:
            image (ndarray): A source image for projection.
        """
        y_start = int(np.floor(self._image_view.y_start))
        y_end = int(np.ceil(self._image_view.y_end))
        x_start = int(np.floor(self._image_view.x_start))
        x_end = int(np.ceil(self._image_view.x_end))

        image = image[y_start:y_end, x_start:x_end]

        if self._direction == "vertical":
            x_val = np.arange(x_start, x_end) + 0.5  # shift to a pixel center
            y_val = bn.nanmean(image, axis=0)

        elif self._direction == "horizontal":
            x_val = bn.nanmean(image, axis=1)
            y_val = np.arange(y_start, y_end) + 0.5  # shift to a pixel center

        self._line_source.data.update(x=x_val, y=y_val)
