import math

import numpy as np
from bokeh.models import (
    BasicTicker,
    BoxZoomTool,
    ColumnDataSource,
    DataRange1d,
    Grid,
    LinearAxis,
    PanTool,
    Plot,
    Quad,
    ResetTool,
    SaveTool,
    Spinner,
    Toggle,
    WheelZoomTool,
)


class Histogram:
    def __init__(self, nplots, plot_height=350, plot_width=700, lower=0, upper=1000, nbins=100):

        self._lower = lower
        self._upper = upper
        self._nbins = nbins

        self._counts = [0 for _ in range(nplots)]

        # Histogram plots
        self.plots = []
        self._plot_sources = []
        for ind in range(nplots):
            plot = Plot(
                x_range=DataRange1d(),
                y_range=DataRange1d(),
                plot_height=plot_height,
                plot_width=plot_width,
                toolbar_location='left',
            )

            # ---- tools
            plot.toolbar.logo = None
            # share 'pan', 'boxzoom', and 'wheelzoom' tools between all plots
            if ind == 0:
                pantool = PanTool()
                boxzoomtool = BoxZoomTool()
                wheelzoomtool = WheelZoomTool()
            plot.add_tools(pantool, boxzoomtool, wheelzoomtool, SaveTool(), ResetTool())

            # ---- axes
            plot.add_layout(LinearAxis(axis_label="Intensity"), place='below')
            plot.add_layout(
                LinearAxis(axis_label="Counts", major_label_orientation='vertical'), place='left'
            )

            # ---- grid lines
            plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
            plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

            # ---- quad (single bin) glyph
            plot_source = ColumnDataSource(dict(left=[], right=[], top=[]))
            plot.add_glyph(
                plot_source,
                Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"),
            )

            self.plots.append(plot)
            self._plot_sources.append(plot_source)

        # Histogram controls
        # ---- histogram range toggle button
        def auto_toggle_callback(state):
            if state:  # Automatic
                lower_spinner.disabled = True
                upper_spinner.disabled = True

            else:  # Manual
                lower_spinner.disabled = False
                upper_spinner.disabled = False

        auto_toggle = Toggle(label="Auto Range", active=True)
        auto_toggle.on_click(auto_toggle_callback)
        self.auto_toggle = auto_toggle

        # ---- histogram lower range
        def lower_spinner_callback(_attr, old_value, new_value):
            if new_value < self._upper:
                self._lower = new_value
                self._counts = [0 for _ in range(nplots)]
            else:
                lower_spinner.value = old_value

        lower_spinner = Spinner(
            title='Lower Range:', value=self._lower, step=0.1, disabled=auto_toggle.active
        )
        lower_spinner.on_change('value', lower_spinner_callback)
        self.lower_spinner = lower_spinner

        # ---- histogram upper range
        def upper_spinner_callback(_attr, old_value, new_value):
            if new_value > self._lower:
                self._upper = new_value
                self._counts = [0 for _ in range(nplots)]
            else:
                upper_spinner.value = old_value

        upper_spinner = Spinner(
            title='Upper Range:', value=self._upper, step=0.1, disabled=auto_toggle.active
        )
        upper_spinner.on_change('value', upper_spinner_callback)
        self.upper_spinner = upper_spinner

        # ---- histogram number of bins
        def nbins_spinner_callback(_attr, old_value, new_value):
            if isinstance(new_value, int):
                if new_value > 0:
                    self._nbins = new_value
                    self._counts = [0 for _ in range(nplots)]
                else:
                    nbins_spinner.value = old_value
            else:
                nbins_spinner.value = old_value

        nbins_spinner = Spinner(title='Number of Bins:', value=self._nbins, step=1)
        nbins_spinner.on_change('value', nbins_spinner_callback)
        self.nbins_spinner = nbins_spinner

        # ---- histogram log10 of counts toggle button
        def log10counts_toggle_callback(state):
            self._counts = [0 for _ in range(nplots)]
            for plot in self.plots:
                if state:
                    plot.yaxis[0].axis_label = 'log⏨(Counts)'
                else:
                    plot.yaxis[0].axis_label = 'Counts'

        log10counts_toggle = Toggle(label='log⏨(Counts)', button_type='default')
        log10counts_toggle.on_click(log10counts_toggle_callback)
        self.log10counts_toggle = log10counts_toggle

    def update(self, input_data, accumulate=False):
        if self.auto_toggle.active and not accumulate:  # automatic
            lower = math.floor(min([np.amin(im) for im in input_data]))
            upper = math.ceil(max([np.amax(im) for im in input_data]))
            if lower == upper:
                upper += 1

            # this will also update self._lower and self._upper
            self.lower_spinner.value = lower
            self.upper_spinner.value = upper

        for ind, plot_source in enumerate(self._plot_sources):
            data_i = input_data[ind]
            counts, edges = np.histogram(
                data_i[data_i != 0], bins=self._nbins, range=(self._lower, self._upper)
            )

            if self.log10counts_toggle.active:
                counts = np.log10(counts, where=counts > 0)

            if accumulate:
                self._counts[ind] += counts
            else:
                self._counts[ind] = counts

            plot_source.data.update(left=edges[:-1], right=edges[1:], top=self._counts[ind])
