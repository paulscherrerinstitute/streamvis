from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (
    BasicTicker,
    BoxZoomTool,
    Button,
    ColumnDataSource,
    DataRange1d,
    Grid,
    Legend,
    LinearAxis,
    PanTool,
    Plot,
    Range1d,
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
    title=Title(text="Hitrate"),
    x_range=DataRange1d(),
    y_range=Range1d(0, 1, bounds=(0, 1)),
    toolbar_location="left",
)

# ---- tools
plot.toolbar.logo = None
plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
plot.add_layout(LinearAxis(), place="below")
plot.add_layout(LinearAxis(), place="left")

# ---- grid lines
plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- red step glyph
step_red_source = ColumnDataSource(dict(x=[], y=[]))
step_red = plot.add_glyph(
    step_red_source, Step(x="x", y="y", mode="after", line_color="red", line_width=2)
)

# ---- blue step glyph
step_blue_source = ColumnDataSource(dict(x=[], y=[]))
step_blue = plot.add_glyph(
    step_blue_source, Step(x="x", y="y", mode="after", line_color="steelblue", line_width=2)
)

# ---- legend
plot.add_layout(
    Legend(
        items=[
            (f"{stats.hitrate_fast.step_size} pulse ids avg", [step_red]),
            (f"{stats.hitrate_slow.step_size} pulse ids avg", [step_blue]),
        ],
        location="top_left",
    )
)
plot.legend.click_policy = "hide"


# Reset button
def reset_button_callback():
    data = step_red_source.data
    if data["x"]:
        step_red_source.data.update(dict(x=[data["x"][-1]], y=[data["y"][-1]]))

    data = step_blue_source.data
    if data["x"]:
        step_blue_source.data.update(dict(x=[data["x"][-1]], y=[data["y"][-1]]))


reset_button = Button(label="Reset", button_type="default", disabled=True)
reset_button.on_click(reset_button_callback)


# Update hitrate plot
def update():
    if not (stats.hitrate_fast and stats.hitrate_slow):
        # Do not update graphs if data is not yet received
        return

    x_fast, y_fast = stats.hitrate_fast.values
    step_red_source.data.update(dict(x=x_fast, y=y_fast))

    x_slow, y_slow = stats.hitrate_slow.values
    step_blue_source.data.update(dict(x=x_slow, y=y_slow))


doc.add_root(
    column(column(plot, sizing_mode="stretch_both"), reset_button, sizing_mode="stretch_width")
)
doc.add_periodic_callback(update, 1000)
