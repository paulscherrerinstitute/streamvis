from collections import deque
from datetime import datetime
from itertools import islice

from bokeh.models import (
    BasicTickFormatter,
    BoxZoomTool,
    Button,
    ColumnDataSource,
    DataRange1d,
    Legend,
    PanTool,
    ResetTool,
    Spinner,
    WheelZoomTool,
)
from bokeh.plotting import figure

MAXLEN = 100


class StreamGraph:
    def __init__(self, nplots, height=200, width=1000, rollover=10800, mode="time"):
        """Initialize stream graph plots.

        Args:
            nplots (int): Number of stream plots that will share common controls.
            height (int, optional): Height of plot area in screen pixels. Defaults to 200.
            width (int, optional): Width of plot area in screen pixels. Defaults to 1000.
            rollover (int, optional): A maximum number of points, above which data from the start
                begins to be discarded. If None, then graph will grow unbounded. Defaults to 10800.
            mode (str, optional): stream update mode, 'time' - uses the local wall time,
                'number' - uses a image number counter. Defaults to 'time'.
        """
        self.rollover = rollover
        self.mode = mode
        self._stream_t = 0
        self._buffers = []
        self._window = 30

        # Stream graphs
        self.plots = []
        self._sources = []

        # share x_range and tools between plots
        shared_x_range = DataRange1d()
        shared_tools = [PanTool(), BoxZoomTool(), WheelZoomTool(dimensions="width"), ResetTool()]
        for ind in range(nplots):
            if mode == "time":
                x_axis_type = "datetime"
            elif mode == "number":
                x_axis_type = "linear"
            else:
                raise ValueError("Parameter `mode` should be either `time` or `number`")

            plot = figure(
                x_axis_type=x_axis_type,
                x_range=shared_x_range,
                y_range=DataRange1d(),
                height=height,
                width=width,
                tools=shared_tools,
                toolbar_location=("left" if ind == 0 else None),  # toolbar only on the first plot
            )

            plot.toolbar.logo = None

            # Custom tick formatter for displaying large numbers
            plot.yaxis.formatter = BasicTickFormatter(precision=1)

            source = ColumnDataSource(dict(x=[], y=[], x_avg=[], y_avg=[]))
            line_renderer = plot.line(source=source, x="x", y="y", line_color="gray")
            line_avg_renderer = plot.line(source=source, x="x_avg", y="y_avg", line_color="red")

            # ---- legend
            plot.add_layout(
                Legend(
                    items=[("per frame", [line_renderer]), ("moving average", [line_avg_renderer])],
                    location="top_left",
                )
            )
            plot.legend.click_policy = "hide"

            self.plots.append(plot)
            self._sources.append(source)
            self._buffers.append(deque(maxlen=MAXLEN))

        # Moving average spinner
        def moving_average_spinner_callback(_attr, _old_value, new_value):
            if moving_average_spinner.low <= new_value <= moving_average_spinner.high:
                self._window = new_value

        moving_average_spinner = Spinner(
            title="Moving Average Window:", value=self._window, low=1, high=MAXLEN, width=145
        )
        moving_average_spinner.on_change("value", moving_average_spinner_callback)
        self.moving_average_spinner = moving_average_spinner

        # Reset button
        def reset_button_callback():
            # keep the latest point in order to prevent a full axis reset
            if mode == "time":
                pass  # update with the lastest time
            elif mode == "number":
                self._stream_t = 1

            for source in self._sources:
                if source.data["x"]:
                    source.data.update(
                        x=[self._stream_t],
                        y=[source.data["y"][-1]],
                        x_avg=[self._stream_t],
                        y_avg=[source.data["y_avg"][-1]],
                    )

        reset_button = Button(label="Reset", button_type="default", width=145)
        reset_button.on_click(reset_button_callback)
        self.reset_button = reset_button

    def update(self, values):
        """Trigger an update for the stream graph plots.

        Args:
            values (ndarray): Source values for stream graph plots.
        """
        if self.mode == "time":
            self._stream_t = datetime.now()
        elif self.mode == "number":
            self._stream_t += 1

        for value, source, buffer in zip(values, self._sources, self._buffers):
            buffer.append(value)
            average = sum(islice(reversed(buffer), self._window)) / min(self._window, len(buffer))
            source.stream(
                dict(x=[self._stream_t], y=[value], x_avg=[self._stream_t], y_avg=[average]),
                rollover=self.rollover,
            )
