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
dash_line_sources = []
dash_lines = []

for ind in range(N_BUF):
    line_source = ColumnDataSource(dict(x=[], y=[]))
    line = plot.add_glyph(line_source, Line(x="x", y="y", line_color=cm[ind], line_width=2))
    line_sources.append(line_source)
    lines.append(line)

    dash_line_source = ColumnDataSource(dict(x=[], y=[]))
    dash_line = plot.add_glyph(
        dash_line_source, Line(x="x", y="y", line_color=cm[ind], line_width=2, line_dash="dashed")
    )
    dash_line_sources.append(dash_line_source)
    dash_lines.append(dash_line)


step_size = stats.roi_intensities.step_size
step_size_fast = stats.roi_intensities_fast.step_size
step_size_select = Select(
    title="Step size (pulse_ids):",
    options=[str(step_size), str(step_size_fast)],
    value=str(step_size),
)


hit_nohit_select = Select(
    title="Intensities", options=["Total", "Split to hit/no_hit"], value="Total"
)


def reset_button_callback():
    stats.roi_intensities.clear()
    stats.roi_intensities_fast.clear()
    stats.roi_intensities_hit.clear()
    stats.roi_intensities_hit_fast.clear()
    stats.roi_intensities_nohit.clear()
    stats.roi_intensities_nohit_fast.clear()


reset_button = Button(label="Reset", button_type="default")
reset_button.on_click(reset_button_callback)


def update():
    is_slow = int(step_size_select.value) == step_size
    is_total = hit_nohit_select.value == "Total"
    if is_slow and is_total:
        x, ys = stats.roi_intensities()
        x_dash, ys_dash = [], [[]]
    elif is_slow and not is_total:
        x, ys = stats.roi_intensities_hit()
        x_dash, ys_dash = stats.roi_intensities_nohit()
    elif not is_slow and is_total:
        x, ys = stats.roi_intensities_fast()
        x_dash, ys_dash = [], [[]]
    else:  # not is_slow and not is_total
        x, ys = stats.roi_intensities_hit_fast()
        x_dash, ys_dash = stats.roi_intensities_nohit_fast()

    for y, source in zip(ys, line_sources):
        source.data.update(dict(x=x, y=y))

    for source in line_sources[len(ys) :]:
        source.data.update(dict(x=[], y=[]))

    for y_dash, source in zip(ys_dash, dash_line_sources):
        source.data.update(dict(x=x_dash, y=y_dash))

    for source in dash_line_sources[len(ys_dash) :]:
        source.data.update(dict(x=[], y=[]))

    if len(plot.legend.items) != len(ys):
        plot.legend.items.clear()
        for i in range(len(ys)):
            plot.legend.items.append(
                LegendItem(label=f"ROI_{i}", renderers=[lines[i], dash_lines[i]])
            )


doc.add_root(
    column(
        column(plot, sizing_mode="stretch_both"),
        row(step_size_select, hit_nohit_select, column(Spacer(height=18), reset_button)),
        sizing_mode="stretch_width",
    )
)
doc.add_periodic_callback(update, 1000)
