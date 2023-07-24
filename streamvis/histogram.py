import bottleneck as bn
import numpy as np
from bokeh.models import CheckboxGroup, ColumnDataSource, DataRange1d, Spinner
from bokeh.plotting import figure

STEP = 0.1


class Histogram:
    def __init__(self, nplots, height=350, width=700, lower=0, upper=1000, nbins=100):
        """Initialize histogram plots.

        Args:
            nplots (int): Number of histogram plots that will share common controls.
            height (int, optional): Height of plot area in screen pixels. Defaults to 350.
            width (int, optional): Width of plot area in screen pixels. Defaults to 700.
            lower (int, optional): Initial lower range of the bins. Defaults to 0.
            upper (int, optional): Initial upper range of the bins. Defaults to 1000.
            nbins (int, optional): Initial number of the bins. Defaults to 100.
        """
        # Histogram plots
        self.plots = []
        self._plot_sources = []
        for ind in range(nplots):
            plot = figure(
                x_range=DataRange1d(),
                y_range=DataRange1d(),
                height=height,
                width=width,
                toolbar_location="left",
                tools="pan,box_zoom,wheel_zoom,save,reset",
            )

            plot.toolbar.logo = None
            # share 'pan', 'boxzoom', and 'wheelzoom' tools between all plots
            if ind == 0:
                shared_tools = plot.toolbar.tools[:3]
            else:
                plot.toolbar.tools[:3] = shared_tools

            plot.yaxis.major_label_orientation = "vertical"

            plot_source = ColumnDataSource(dict(left=[], right=[], top=[]))
            plot.quad(
                source=plot_source,
                left="left",
                right="right",
                top="top",
                bottom=0,
                line_color="black",
            )

            self.plots.append(plot)
            self._plot_sources.append(plot_source)

        self._counts = []
        self._empty_counts()

        # Histogram controls
        # ---- histogram range switch
        def auto_switch_callback(_attr, _old, new):
            if 0 in new:  # Automatic
                lower_spinner.disabled = True
                upper_spinner.disabled = True

            else:  # Manual
                lower_spinner.disabled = False
                upper_spinner.disabled = False

        auto_switch = CheckboxGroup(labels=["Auto Hist Range"], active=[0], width=145)
        auto_switch.on_change("active", auto_switch_callback)
        self.auto_switch = auto_switch

        # ---- histogram lower range
        def lower_spinner_callback(_attr, _old_value, new_value):
            self.upper_spinner.low = new_value + STEP
            self._empty_counts()

        lower_spinner = Spinner(
            title="Lower Range:",
            high=upper - STEP,
            value=lower,
            step=STEP,
            disabled=bool(auto_switch.active),
            width=145,
        )
        lower_spinner.on_change("value", lower_spinner_callback)
        self.lower_spinner = lower_spinner

        # ---- histogram upper range
        def upper_spinner_callback(_attr, _old_value, new_value):
            self.lower_spinner.high = new_value - STEP
            self._empty_counts()

        upper_spinner = Spinner(
            title="Upper Range:",
            low=lower + STEP,
            value=upper,
            step=STEP,
            disabled=bool(auto_switch.active),
            width=145,
        )
        upper_spinner.on_change("value", upper_spinner_callback)
        self.upper_spinner = upper_spinner

        # ---- histogram number of bins
        def nbins_spinner_callback(_attr, _old_value, _new_value):
            self._empty_counts()

        nbins_spinner = Spinner(title="Number of Bins:", low=1, value=nbins, width=145)
        nbins_spinner.on_change("value", nbins_spinner_callback)
        self.nbins_spinner = nbins_spinner

        # ---- histogram log10 of counts switch
        def log10counts_switch_callback(_attr, _old, new):
            self._empty_counts()
            for plot in self.plots:
                if 0 in new:
                    plot.yaxis[0].axis_label = "log⏨(Counts)"
                else:
                    plot.yaxis[0].axis_label = "Counts"

        log10counts_switch = CheckboxGroup(labels=["log⏨(Counts)"], width=145)
        log10counts_switch.on_change("active", log10counts_switch_callback)
        self.log10counts_switch = log10counts_switch

    def _empty_counts(self):
        self._counts = [0 for _ in range(len(self.plots))]

    @property
    def lower(self):
        """Lower range of the bins (readonly)"""
        return self.lower_spinner.value

    @property
    def upper(self):
        """Upper range of the bins (readonly)"""
        return self.upper_spinner.value

    @property
    def nbins(self):
        """Number of the bins (readonly)"""
        return self.nbins_spinner.value

    def update(self, input_data, accumulate=False):
        """Trigger an update for the histogram plots.

        Args:
            input_data (ndarray): Source values for histogram plots.
            accumulate (bool, optional): Add together bin values of the previous and current data.
                Defaults to False.
        """
        if self.auto_switch.active and not accumulate:  # automatic
            # find the lowest and the highest value in input data
            lower = 0
            upper = 1

            for data in input_data:
                min_val = bn.nanmin(data)
                min_val = 0 if np.isnan(min_val) else min_val
                lower = min(lower, min_val)

                max_val = bn.nanmax(data)
                max_val = 1 if np.isnan(max_val) else max_val
                upper = max(upper, max_val)

            self.lower_spinner.value = int(np.floor(lower))
            self.upper_spinner.value = int(np.ceil(upper))

        # get histogram counts and update plots
        for i, data in enumerate(input_data):
            # np.histogram on 16M values can take around 0.5 sec, which is too much, thus reduce the
            # number of processed values (not the ideal implementation, but should be good enough)
            ratio = np.sqrt(data.size / 2_000_000)
            if ratio > 1:
                shape_x, shape_y = data.shape
                stride_x = ratio * shape_x / shape_y
                stride_y = ratio * shape_y / shape_x

                if stride_x < 1:
                    stride_y = int(np.ceil(stride_y * stride_x))
                    stride_x = 1
                elif stride_y < 1:
                    stride_x = int(np.ceil(stride_y * stride_x))
                    stride_y = 1
                else:
                    stride_x = int(np.ceil(stride_x))
                    stride_y = int(np.ceil(stride_y))

                data = data[::stride_y, ::stride_x]

            next_counts, edges = np.histogram(data, bins=self.nbins, range=(self.lower, self.upper))

            if self.log10counts_switch.active:
                next_counts = np.log10(next_counts, where=next_counts > 0)

            if accumulate:
                self._counts[i] += next_counts
            else:
                self._counts[i] = next_counts

            self._plot_sources[i].data.update(left=edges[:-1], right=edges[1:], top=self._counts[i])
