from collections import deque
from datetime import datetime
from itertools import islice

from bokeh.models import (
    BasicTicker,
    BasicTickFormatter,
    BoxZoomTool,
    Button,
    ColumnDataSource,
    DataRange1d,
    DatetimeAxis,
    Grid,
    Legend,
    Line,
    LinearAxis,
    PanTool,
    Plot,
    ResetTool,
    Spinner,
    WheelZoomTool,
)

MAXLEN = 100


class StreamGraph:
    def __init__(self, nplots, plot_height=200, plot_width=1000, rollover=3600, mode='time'):
        self.rollover = rollover
        self.mode = mode
        self._stream_t = 0
        self._buffers = []
        self._window = 1

        # Custom tick formatter for displaying large numbers
        tick_formatter = BasicTickFormatter(precision=1)

        # Stream graphs
        self.plots = []
        self.glyphs = []
        self._sources = []
        for ind in range(nplots):
            # share x_range between plots
            if ind == 0:
                x_range = DataRange1d()

            plot = Plot(
                x_range=x_range,
                y_range=DataRange1d(),
                plot_height=plot_height,
                plot_width=plot_width,
            )

            # ---- tools
            plot.toolbar.logo = None
            plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

            # ---- axes
            plot.add_layout(LinearAxis(formatter=tick_formatter), place='left')
            if mode == 'time':
                plot.add_layout(DatetimeAxis(), place='below')
            elif mode == 'number':
                plot.add_layout(LinearAxis(), place='below')
            else:
                pass

            # ---- grid lines
            plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
            plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

            # ---- line glyph
            source = ColumnDataSource(dict(x=[], y=[], x_avg=[], y_avg=[]))
            line = Line(x='x', y='y')
            line_avg = Line(x='x_avg', y='y_avg', line_color='red')
            line_renderer = plot.add_glyph(source, line)
            line_avg_renderer = plot.add_glyph(source, line_avg)

            # ---- legend
            plot.add_layout(
                Legend(
                    items=[("per frame", [line_renderer]), ("moving average", [line_avg_renderer])],
                    location='top_left',
                )
            )
            plot.legend.click_policy = "hide"

            self.plots.append(plot)
            self.glyphs.append(line)
            self._sources.append(source)
            self._buffers.append(deque(maxlen=MAXLEN))

        # Moving average spinner
        def moving_average_spinner_callback(_attr, _old_value, new_value):
            if moving_average_spinner.low <= new_value <= moving_average_spinner.high:
                self._window = new_value

        moving_average_spinner = Spinner(
            title='Moving Average Window:', value=self._window, low=1, high=MAXLEN
        )
        moving_average_spinner.on_change('value', moving_average_spinner_callback)
        self.moving_average_spinner = moving_average_spinner

        # Reset button
        def reset_button_callback():
            # keep the latest point in order to prevent a full axis reset
            if mode == 'time':
                pass  # update with the lastest time
            elif mode == 'number':
                self._stream_t = 1

            for source in self._sources:
                source.data.update(
                    x=[self._stream_t],
                    y=[source.data['y'][-1]],
                    x_avg=[self._stream_t],
                    y_avg=[source.data['y_avg'][-1]],
                )

        reset_button = Button(label="Reset", button_type='default')
        reset_button.on_click(reset_button_callback)
        self.reset_button = reset_button

    def update(self, values):
        if self.mode == 'time':
            self._stream_t = datetime.now()
        elif self.mode == 'number':
            self._stream_t += 1

        for value, source, buffer in zip(values, self._sources, self._buffers):
            buffer.append(value)
            average = sum(islice(reversed(buffer), self._window)) / self._window
            source.stream(
                dict(x=[self._stream_t], y=[value], x_avg=[self._stream_t], y_avg=[average]),
                rollover=self.rollover,
            )
