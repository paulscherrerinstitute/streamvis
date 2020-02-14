from datetime import datetime

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
    Line,
    LinearAxis,
    PanTool,
    Plot,
    Range1d,
    ResetTool,
    SaveTool,
    Title,
    WheelZoomTool,
)

doc = curdoc()
stats = doc.stats
doc.title = f"{doc.title} Hitrate"

HITRATE_ROLLOVER = 1200

# Hitrate plot
plot = Plot(
    title=Title(text="Hitrate"),
    x_range=DataRange1d(),
    y_range=Range1d(0, 1, bounds=(0, 1)),
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

# ---- red line glyph
line_red_source = ColumnDataSource(dict(x=[], y=[]))
red_line = plot.add_glyph(line_red_source, Line(x="x", y="y", line_color="red", line_width=2))

# ---- blue line glyph
line_blue_source = ColumnDataSource(dict(x=[], y=[]))
blue_line = plot.add_glyph(
    line_blue_source, Line(x="x", y="y", line_color="steelblue", line_width=2)
)

# ---- legend
plot.add_layout(
    Legend(
        items=[
            (f"{stats.hitrate_buffer_fast.maxlen} shots avg", [red_line]),
            (f"{stats.hitrate_buffer_slow.maxlen} shots avg", [blue_line]),
        ],
        location="top_left",
    )
)
plot.legend.click_policy = "hide"


# Reset button
def reset_button_callback():
    data = line_red_source.data
    if data["x"]:
        line_red_source.data.update(dict(x=[data["x"][-1]], y=[data["y"][-1]]))

    data = line_blue_source.data
    if data["x"]:
        line_blue_source.data.update(dict(x=[data["x"][-1]], y=[data["y"][-1]]))


reset_button = Button(label="Reset", button_type="default")
reset_button.on_click(reset_button_callback)


# Update hitrate plot
def update():
    if not (stats.hitrate_buffer_fast and stats.hitrate_buffer_slow):
        # Do not update graphs if data is not yet received
        return

    stream_t = datetime.now()

    line_red_source.stream(
        new_data=dict(
            x=[stream_t], y=[sum(stats.hitrate_buffer_fast) / len(stats.hitrate_buffer_fast)]
        ),
        rollover=HITRATE_ROLLOVER,
    )

    line_blue_source.stream(
        new_data=dict(
            x=[stream_t], y=[sum(stats.hitrate_buffer_slow) / len(stats.hitrate_buffer_slow)]
        ),
        rollover=HITRATE_ROLLOVER,
    )


doc.add_root(
    column(column(plot, sizing_mode="stretch_both"), reset_button, sizing_mode="stretch_width")
)
doc.add_periodic_callback(update, 1000)
