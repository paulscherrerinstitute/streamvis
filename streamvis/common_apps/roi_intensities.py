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
    LegendItem,
    Line,
    LinearAxis,
    PanTool,
    Plot,
    ResetTool,
    SaveTool,
    Select,
    Spacer,
    Title,
    WheelZoomTool,
)

from bokeh.palettes import Category10

N_BUF = 10
cm = Category10[N_BUF]

doc = curdoc()
stats = doc.stats
doc.title = f"{doc.title} ROI intensities"

plot = Plot(
    title=Title(text="ROI intensities"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    toolbar_location="left",
)

plot.toolbar.logo = None
plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

plot.add_layout(
    LinearAxis(axis_label="pulse_id", formatter=BasicTickFormatter(use_scientific=False)),
    place="below",
)
plot.add_layout(LinearAxis(), place="left")

plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

plot.add_layout(Legend(items=[], location="top_left"))
plot.legend.click_policy = "hide"

line_sources = []
lines = []

for ind in range(N_BUF):
    line_source = ColumnDataSource(dict(x=[], y=[]))
    line = plot.add_glyph(line_source, Line(x="x", y="y", line_color=cm[ind], line_width=2))
    line_sources.append(line_source)
    lines.append(line)


def reset_button_callback():
    stats.roi_intensities.clear()
    stats.roi_intensities_fast.clear()


step_size = stats.roi_intensities.step_size
step_size_fast = stats.roi_intensities_fast.step_size
step_size_select = Select(
    title="Step size (pulse_ids):",
    options=[str(step_size), str(step_size_fast)],
    value=str(step_size),
)


reset_button = Button(label="Reset", button_type="default")
reset_button.on_click(reset_button_callback)


def update():
    if int(step_size_select.value) == step_size:
        x, ys = stats.roi_intensities()
    else:
        x, ys = stats.roi_intensities_fast()

    for y, source in zip(ys, line_sources):
        source.data.update(dict(x=x, y=y))

    for source in line_sources[len(ys) :]:
        source.data.update(dict(x=[], y=[]))

    if len(plot.legend.items) != len(ys):
        plot.legend.items.clear()
        for i in range(len(ys)):
            plot.legend.items.append(LegendItem(label=f"ROI_{i}", renderers=[lines[i]]))


doc.add_root(
    column(
        column(plot, sizing_mode="stretch_both"),
        row(step_size_select, column(Spacer(height=18), reset_button)),
        sizing_mode="stretch_width",
    )
)
doc.add_periodic_callback(update, 1000)
