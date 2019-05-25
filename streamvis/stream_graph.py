from datetime import datetime

from bokeh.models import (
    BasicTicker,
    BasicTickFormatter,
    BoxZoomTool,
    Button,
    ColumnDataSource,
    DataRange1d,
    DatetimeAxis,
    Grid,
    Line,
    LinearAxis,
    PanTool,
    Plot,
    ResetTool,
    WheelZoomTool,
)


class StreamGraph:
    def __init__(self, nplots, plot_height=200, plot_width=1000, rollover=3600, mode='time'):
        self.rollover = rollover
        self.mode = mode
        self._stream_t = 0

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
            source = ColumnDataSource(dict(x=[], y=[]))
            line = Line(x='x', y='y')
            plot.add_glyph(source, line)

            self.plots.append(plot)
            self.glyphs.append(line)
            self._sources.append(source)

        # Reset button
        def reset_button_callback():
            # keep the latest point in order to prevent a full axis reset
            if mode == 'time':
                pass  # update with the lastest time
            elif mode == 'number':
                self._stream_t = 1

            for source in self._sources:
                source.data.update(x=[self._stream_t], y=[source.data['y'][-1]])

        reset_button = Button(label="Reset", button_type='default')
        reset_button.on_click(reset_button_callback)
        self.reset_button = reset_button

    def update(self, values):
        if self.mode == 'time':
            self._stream_t = datetime.now()
        elif self.mode == 'number':
            self._stream_t += 1

        for ind, source in enumerate(self._sources):
            source.stream(dict(x=[self._stream_t], y=[values[ind]]), rollover=self.rollover)
