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
    Spacer,
    Spinner,
    Step,
    Title,
    WheelZoomTool,
)

doc = curdoc()
stats = doc.stats
doc.title = f"{doc.title} ROI Pump-Probe"

plot = Plot(
    title=Title(text="ROI Pump-Probe"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    toolbar_location="left",
)

plot.toolbar.logo = None
plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(maintain_focus=False), SaveTool(), ResetTool(),
)

plot.add_layout(
    LinearAxis(axis_label="pulse_id", formatter=BasicTickFormatter(use_scientific=False)),
    place="below",
)
plot.add_layout(LinearAxis(), place="left")

plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

step_source = ColumnDataSource(dict(x=[], y=[]))
step = plot.add_glyph(
    step_source, Step(x="x", y="y", mode="after", line_color="steelblue", line_width=2)
)

step_nobkg_source = ColumnDataSource(dict(x=[], y=[]))
step_nobkg = plot.add_glyph(
    step_nobkg_source, Step(x="x", y="y", mode="after", line_color="firebrick", line_width=2)
)

plot.add_layout(
    Legend(
        items=[("ROI Pump-probe", [step]), ("ROI Pump-probe nobkg", [step_nobkg])],
        location="top_left",
    )
)
plot.legend.click_policy = "hide"

step_size = stats.roi_pump_probe.step_size
max_span = stats.roi_pump_probe.max_span
average_window_spinner = Spinner(
    title="Pulse ID Window:", value=step_size, low=step_size, high=max_span, step=step_size
)


def reset_button_callback():
    stats.roi_pump_probe.clear()
    stats.roi_pump_probe_nobkg.clear()


reset_button = Button(label="Reset", button_type="default")
reset_button.on_click(reset_button_callback)


def update():
    x, y = stats.roi_pump_probe(average_window_spinner.value)
    if len(y) > 1:
        y[-1] = y[-2]
    step_source.data.update(dict(x=x, y=y))

    x, y = stats.roi_pump_probe_nobkg(average_window_spinner.value)
    if len(y) > 1:
        y[-1] = y[-2]
    step_nobkg_source.data.update(dict(x=x, y=y))


doc.add_root(
    column(
        column(plot, sizing_mode="stretch_both"),
        row(average_window_spinner, column(Spacer(height=19), reset_button)),
        sizing_mode="stretch_width",
    )
)
doc.add_periodic_callback(update, 1000)
