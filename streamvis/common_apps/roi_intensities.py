from datetime import datetime

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (
    BasicTicker,
    BoxZoomTool,
    Button,
    ColumnDataSource,
    DataRange1d,
    DatetimeAxis,
    Grid,
    Legend,
    LegendItem,
    Line,
    LinearAxis,
    PanTool,
    Plot,
    ResetTool,
    SaveTool,
    Title,
    WheelZoomTool,
)

from bokeh.palettes import Set1

cm = Set1[9]

doc = curdoc()
stats = doc.stats
doc.title = f"{doc.title} ROI intensities"

ROLLOVER = 3600

# ROI intensities plot
plot = Plot(
    title=Title(text="ROI intensities"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    toolbar_location="left",
)

# ---- tools
plot.toolbar.logo = None
plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
plot.add_layout(DatetimeAxis(), place="below")
plot.add_layout(LinearAxis(), place="left")

# ---- grid lines
plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- legend
plot.add_layout(Legend(items=[], location="top_left"))
plot.legend.click_policy = "hide"

sources = []
lines = []

for ind in range(len(stats.roi_intensities_buffers)):
    line_source = ColumnDataSource(dict(x=[], y=[]))
    line = plot.add_glyph(line_source, Line(x="x", y="y", line_color=cm[ind], line_width=2))
    sources.append(line_source)
    lines.append(line)

# Reset button
def reset_button_callback():
    for source in sources:
        if source.data["x"]:
            source.data.update(dict(x=[source.data["x"][-1]], y=[source.data["y"][-1]]))


reset_button = Button(label="Reset", button_type="default")
reset_button.on_click(reset_button_callback)


# Update ROI intensities plot
def update():
    stream_t = datetime.now()

    n_buf = 0
    for buffer, source in zip(stats.roi_intensities_buffers, sources):
        if buffer:
            n_buf += 1
            source.stream(new_data=dict(x=[stream_t], y=[np.mean(buffer)]), rollover=ROLLOVER)
        else:
            source.data.update(dict(x=[], y=[]))

    if len(plot.legend.items) != n_buf:
        plot.legend.items.clear()
        for i in range(n_buf):
            plot.legend.items.append(LegendItem(label=f"ROI_{i}", renderers=[lines[i]]))


doc.add_root(
    column(column(plot, sizing_mode="stretch_both"), reset_button, sizing_mode="stretch_width")
)
doc.add_periodic_callback(update, 1000)
