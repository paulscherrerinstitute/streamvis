from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    BoxZoomTool,
    ColumnDataSource,
    DataRange1d,
    Grid,
    Legend,
    Line,
    LinearAxis,
    PanTool,
    Plot,
    ResetTool,
    SaveTool,
    Spinner,
    Title,
    WheelZoomTool,
)


doc = curdoc()
stats = doc.stats
doc.title = f"{doc.title} Radial Profile"

# Radial Profile plot
plot = Plot(
    title=Title(text="Radial Profile"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
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

# ---- black line glyph
line_off_source = ColumnDataSource(dict(x=[], y=[]))
line_off = plot.add_glyph(line_off_source, Line(x="x", y="y", line_color="black", line_width=2))

# ---- blue line glyph
line_on_source = ColumnDataSource(dict(x=[], y=[]))
line_on = plot.add_glyph(line_on_source, Line(x="x", y="y", line_color="blue", line_width=2))

# ---- red line glyph
line_diff_source = ColumnDataSource(dict(x=[], y=[]))
line_diff = plot.add_glyph(line_diff_source, Line(x="x", y="y", line_color="red", line_width=2))

# ---- legend
plot.add_layout(
    Legend(
        items=[("Laser off", [line_off]), ("Laser on", [line_on]), ("Difference", [line_diff])],
        location="top_left",
    )
)
plot.legend.click_policy = "hide"

# Average window spinner
average_window_spinner = Spinner(title="Pulse ID Window:", value=100, low=100, high=10000, step=100)
frames_off_spinner = Spinner(title="Frames laser off:", value=0, disabled=True)
frames_on_spinner = Spinner(title="Frames laser on:", value=0, disabled=True)

# Update ROI intensities plot
def update():
    if not (stats.radial_profile_lon or stats.radial_profile_loff):
        # Do not update graphs if data is not yet received
        return

    q, avg_I_on, num_on = stats.radial_profile_lon(average_window_spinner.value)
    _, avg_I_off, num_off = stats.radial_profile_loff(average_window_spinner.value)

    frames_off_spinner.value = num_off
    frames_on_spinner.value = num_on

    if num_off:
        line_off_source.data.update(x=q, y=avg_I_off)
    else:
        line_off_source.data.update(x=[], y=[])

    if num_on:
        line_on_source.data.update(x=q, y=avg_I_on)
    else:
        line_on_source.data.update(x=[], y=[])

    if num_off and num_on:
        line_diff_source.data.update(x=q, y=avg_I_on - avg_I_off)
    else:
        line_diff_source.data.update(x=[], y=[])


doc.add_root(
    column(
        column(plot, sizing_mode="stretch_both"),
        row(average_window_spinner, frames_off_spinner, frames_on_spinner),
        sizing_mode="stretch_width",
    )
)
doc.add_periodic_callback(update, 1000)
