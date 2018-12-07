import numpy as np
from bokeh.models import BasicTicker, BoxZoomTool, ColumnDataSource, \
    DataRange1d, Grid, LinearAxis, PanTool, Plot, Quad, RadioButtonGroup, \
    ResetTool, SaveTool, TextInput, Toggle, WheelZoomTool

class Histogram:
    def __init__(
            self, nplots=1, plot_height=350, plot_width=700,
            init_lower=0, init_upper=1000, init_nbins=100,
        ):

        self._lower = init_lower
        self._upper = init_upper
        self._nbins = init_nbins

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
                LinearAxis(axis_label="Counts", major_label_orientation='vertical'),
                place='left',
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
        # ---- histogram radio button
        def radiobuttongroup_callback(selection):
            if selection == 0:  # Automatic
                lower_textinput.disabled = True
                upper_textinput.disabled = True
                nbins_textinput.disabled = True

            else:  # Manual
                lower_textinput.disabled = False
                upper_textinput.disabled = False
                nbins_textinput.disabled = False

        radiobuttongroup = RadioButtonGroup(labels=["Automatic", "Manual"], active=0, width=150)
        radiobuttongroup.on_click(radiobuttongroup_callback)
        self.radiobuttongroup = radiobuttongroup

        # ---- histogram lower range
        def lower_textinput_callback(_attr, old, new):
            try:
                new_value = float(new)
                if new_value < self._upper:
                    self._lower = new_value
                else:
                    lower_textinput.value = old

            except ValueError:
                lower_textinput.value = old

        lower_textinput = TextInput(title='Lower Range:', value=str(self._lower), disabled=True)
        lower_textinput.on_change('value', lower_textinput_callback)
        self.lower_textinput = lower_textinput

        # ---- histogram upper range
        def upper_textinput_callback(_attr, old, new):
            try:
                new_value = float(new)
                if new_value > self._lower:
                    self._upper = new_value
                else:
                    upper_textinput.value = old

            except ValueError:
                upper_textinput.value = old

        upper_textinput = TextInput(title='Upper Range:', value=str(self._upper), disabled=True)
        upper_textinput.on_change('value', upper_textinput_callback)
        self.upper_textinput = upper_textinput

        # ---- histogram number of bins
        def nbins_textinput_callback(_attr, old, new):
            try:
                new_value = int(new)
                if new_value > 0:
                    self._nbins = new_value
                else:
                    nbins_textinput.value = old

            except ValueError:
                nbins_textinput.value = old

        nbins_textinput = TextInput(title='Number of Bins:', value=str(self._nbins), disabled=True)
        nbins_textinput.on_change('value', nbins_textinput_callback)
        self.nbins_textinput = nbins_textinput

        # ---- histogram log10 of counts toggle button
        def log10counts_toggle_callback(state):
            for plot in self.plots:
                if state:
                    plot.yaxis[0].axis_label = 'log⏨(Counts)'
                else:
                    plot.yaxis[0].axis_label = 'Counts'

        log10counts_toggle = Toggle(label='log⏨(Counts)', button_type='default')
        log10counts_toggle.on_click(log10counts_toggle_callback)
        self.log10counts_toggle = log10counts_toggle

    def update(self, input_data):
        if self.radiobuttongroup.active == 0:  # automatic
            kwarg = dict(bins='scott')
        else:  # manual
            kwarg = dict(bins=self._nbins, range=(self._lower, self._upper))

        for ind in range(len(self.plots)):
            data_i = input_data[ind]
            counts, edges = np.histogram(data_i[data_i != 0], **kwarg)

            if self.log10counts_toggle.active:
                counts = np.log10(counts, where=counts > 0)

            self._plot_sources[ind].data.update(left=edges[:-1], right=edges[1:], top=counts)
