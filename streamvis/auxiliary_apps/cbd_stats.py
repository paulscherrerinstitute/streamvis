import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, grid, row
from bokeh.models import Button, Spacer, Title

import streamvis as sv
from streamvis.jfcbd_statistics_handler import CBDStatisticsHandler

PLOT_WIDTH = 600
PLOT_HEIGHT = 600

doc = curdoc()
stats: CBDStatisticsHandler = doc.stream_adapter.stats
doc.title = f"{doc.title} CBD Statistics"


sv_streamgraph = sv.StreamGraph(nplots=3, height=PLOT_HEIGHT, rollover=5000, width=PLOT_WIDTH)
sv_streamgraph.plots[0].title = Title(text="# Streaks")
sv_streamgraph.plots[1].title = Title(text="Avg Streak length")
sv_streamgraph.plots[2].title = Title(text="Sum Bragg Counts")

sv_hist_1 = sv.Histogram(nplots=1, height=PLOT_HEIGHT, width=PLOT_WIDTH)
sv_hist_1.plots[0].title = Title(text="# Streaks")
sv_hist_2 = sv.Histogram(nplots=1, height=PLOT_HEIGHT, width=PLOT_WIDTH)
sv_hist_2.plots[0].title = Title(text="Streak length", text_color="red")
sv_hist_3 = sv.Histogram(nplots=1, height=PLOT_HEIGHT, width=PLOT_WIDTH)
sv_hist_3.plots[0].title = Title(text="Bragg Counts", text_color="green")

sv_stxm = sv.ScatterPlot(PLOT_WIDTH * 2, PLOT_HEIGHT, 10, "STXM Bragg Counts")

hist_layout = row(
    column(
        sv_hist_1.plots[0],
        row(
            sv_hist_1.auto_switch,
            sv_hist_1.lower_spinner,
            sv_hist_1.upper_spinner,
            sv_hist_1.nbins_spinner,
        ),
    ),
    column(
        sv_hist_2.plots[0],
        row(
            sv_hist_2.auto_switch,
            sv_hist_2.lower_spinner,
            sv_hist_2.upper_spinner,
            sv_hist_2.nbins_spinner,
        ),
    ),
    column(
        sv_hist_3.plots[0],
        row(
            sv_hist_3.auto_switch,
            sv_hist_3.lower_spinner,
            sv_hist_3.upper_spinner,
            sv_hist_3.nbins_spinner,
        ),
    ),
)


reset_button = Button(label="Reset", button_type="default", width=145)


def reset_button_callback():
    stats.reset()
    # Clear stream graph manually
    for source in sv_streamgraph._sources:
        if source.data["x"]:
            source.data.update(x=[], y=[], x_avg=[], y_avg=[])
    sv_stxm.clear()


reset_button.on_click(reset_button_callback)


stream_layout = grid(sv_streamgraph.plots, nrows=1)

layout = column(
    row(hist_layout),
    Spacer(height=10),
    row(stream_layout),
    Spacer(height=10),
    sv_stxm.default_layout,
    Spacer(height=10),
    reset_button,
)

doc.add_root(layout)


def update():
    # Line plots
    if stats.number_of_streaks and stats.streak_lengths and stats.bragg_counts:
        # Histograms - update with full data accumulated until now
        sv_hist_1.update([np.array([stats.number_of_streaks()])])
        sv_hist_2.update([np.array([stats.streak_lengths()])])
        sv_hist_3.update([np.array([stats.bragg_counts()])])
        # Stream line - just update last value
        sv_streamgraph.update(
            [
                stats.number_of_streaks.last_value,
                stats.streak_lengths.last_value,
                stats.bragg_counts.last_value,
            ]
        )

    # STXM - update with values not yet retrieved
    if stats.bragg_aggregator:
        vs, pids = stats.bragg_aggregator()
        sv_stxm.update(values=vs, pulse_ids=pids)


doc.add_periodic_callback(update, 2000)
