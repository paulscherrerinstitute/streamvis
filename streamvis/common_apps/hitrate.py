from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    BasicTickFormatter,
    BoxZoomTool,
    Button,
    ColumnDataSource,
    DataRange1d,
    Grid,
    Legend,
    LinearAxis,
    PanTool,
    Plot,
    ResetTool,
    SaveTool,
    Step,
    Title,
    WheelZoomTool,
)

doc = curdoc()
stats = doc.stats
doc.title = f"{doc.title} Hitrate"

plot = Plot(
    title=Title(text="Hitrate Plot"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    toolbar_location="left",
)

plot.toolbar.logo = None
plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(maintain_focus=False), SaveTool(), ResetTool()
)

plot.add_layout(
    LinearAxis(axis_label="pulse_id", formatter=BasicTickFormatter(use_scientific=False)),
    place="below",
)
plot.add_layout(LinearAxis(axis_label="Hitrate"), place="left")

plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

step_fast_source = ColumnDataSource(dict(x=[], y=[]))
step_fast = plot.add_glyph(
    step_fast_source, Step(x="x", y="y", mode="after", line_color="steelblue", line_width=2)
)

step_fast_lon_source = ColumnDataSource(dict(x=[], y=[]))
step_fast_lon = plot.add_glyph(
    step_fast_lon_source,
    Step(x="x", y="y", mode="after", line_color="steelblue", line_width=2, line_dash="dashed"),
    visible=False,
)

step_fast_loff_source = ColumnDataSource(dict(x=[], y=[]))
step_fast_loff = plot.add_glyph(
    step_fast_loff_source,
    Step(x="x", y="y", mode="after", line_color="steelblue", line_width=2, line_dash="dotted"),
    visible=False,
)

step_slow_source = ColumnDataSource(dict(x=[], y=[]))
step_slow = plot.add_glyph(
    step_slow_source, Step(x="x", y="y", mode="after", line_color="red", line_width=2)
)

step_slow_lon_source = ColumnDataSource(dict(x=[], y=[]))
step_slow_lon = plot.add_glyph(
    step_slow_lon_source,
    Step(x="x", y="y", mode="after", line_color="red", line_width=2, line_dash="dashed"),
    visible=False,
)

step_slow_loff_source = ColumnDataSource(dict(x=[], y=[]))
step_slow_loff = plot.add_glyph(
    step_slow_loff_source,
    Step(x="x", y="y", mode="after", line_color="red", line_width=2, line_dash="dotted"),
    visible=False,
)

hitrate_sources = (
    (stats.hitrate_fast, step_fast_source),
    (stats.hitrate_fast_lon, step_fast_lon_source),
    (stats.hitrate_fast_loff, step_fast_loff_source),
    (stats.hitrate_slow, step_slow_source),
    (stats.hitrate_slow_lon, step_slow_lon_source),
    (stats.hitrate_slow_loff, step_slow_loff_source),
)

plot.add_layout(
    Legend(
        items=[
            (f"{stats.hitrate_fast.step_size} pulse avg", [step_fast]),
            ("    laser on", [step_fast_lon]),
            ("    laser off", [step_fast_loff]),
            (f"{stats.hitrate_slow.step_size} pulse avg", [step_slow]),
            ("    laser on", [step_slow_lon]),
            ("    laser off", [step_slow_loff]),
        ],
        location="top_left",
    )
)
plot.legend.click_policy = "hide"


def reset_button_callback():
    for hitrate, _ in hitrate_sources:
        hitrate.clear()


reset_button = Button(label="Reset", button_type="default")
reset_button.on_click(reset_button_callback)


def update():
    for hitrate, source in hitrate_sources:
        x, y = hitrate()
        if len(y) > 1:
            y[-1] = y[-2]
        source.data.update(dict(x=x, y=y))


doc.add_root(
    column(column(plot, sizing_mode="stretch_both"), row(reset_button), sizing_mode="stretch_width")
)
doc.add_periodic_callback(update, 1000)
