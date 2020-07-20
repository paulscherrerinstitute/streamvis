from bokeh.io import curdoc
from bokeh.layouts import column
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

# Hitrate plot
plot = Plot(
    title=Title(text="Hitrate Plot"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    toolbar_location="left",
)

# ---- tools
plot.toolbar.logo = None
plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(maintain_focus=False), SaveTool(), ResetTool(),
)

# ---- axes
plot.add_layout(
    LinearAxis(axis_label="pulse_id", formatter=BasicTickFormatter(use_scientific=False)),
    place="below",
)
plot.add_layout(LinearAxis(axis_label="Hitrate"), place="left")

# ---- grid lines
plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- blue step glyph
step_fast_source = ColumnDataSource(dict(x=[], y=[]))
step_fast = plot.add_glyph(
    step_fast_source, Step(x="x", y="y", mode="after", line_color="steelblue", line_width=2)
)

# ---- red step glyph
step_slow_source = ColumnDataSource(dict(x=[], y=[]))
step_slow = plot.add_glyph(
    step_slow_source, Step(x="x", y="y", mode="after", line_color="red", line_width=2)
)

# ---- legend
plot.add_layout(
    Legend(
        items=[
            (f"{stats.hitrate_fast.step_size} pulse ids avg", [step_fast]),
            (f"{stats.hitrate_slow.step_size} pulse ids avg", [step_slow]),
        ],
        location="top_left",
    )
)
plot.legend.click_policy = "hide"


# Reset button
def reset_button_callback():
    data = step_fast_source.data
    if data["x"]:
        step_fast_source.data.update(dict(x=[data["x"][-1]], y=[data["y"][-1]]))

    data = step_slow_source.data
    if data["x"]:
        step_slow_source.data.update(dict(x=[data["x"][-1]], y=[data["y"][-1]]))


reset_button = Button(label="Reset", button_type="default", disabled=True)
reset_button.on_click(reset_button_callback)


# Update hitrate plot
def update():
    if not (stats.hitrate_fast and stats.hitrate_slow):
        # Do not update graphs if data is not yet received
        return

    x_fast, y_fast = stats.hitrate_fast.values
    y_fast[-1] = y_fast[-2]
    step_fast_source.data.update(dict(x=x_fast, y=y_fast))

    x_slow, y_slow = stats.hitrate_slow.values
    y_slow[-1] = y_slow[-2]
    step_slow_source.data.update(dict(x=x_slow, y=y_slow))


doc.add_root(
    column(column(plot, sizing_mode="stretch_both"), reset_button, sizing_mode="stretch_width")
)
doc.add_periodic_callback(update, 1000)
