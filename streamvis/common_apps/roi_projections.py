from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, DataRange1d, Legend, Select, Spacer, Spinner
from bokeh.plotting import figure

doc = curdoc()
stats = doc.stats
doc.title = f"{doc.title} ROI Projections"

plot = figure(
    title="ROI Projections",
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    toolbar_location="left",
    tools="pan,box_zoom,wheel_zoom,save,reset",
)

plot.toolbar.logo = None

line_off_source = ColumnDataSource(dict(x=[], y=[]))
line_off = plot.line(source=line_off_source, x="x", y="y", line_color="black", line_width=2)

line_on_source = ColumnDataSource(dict(x=[], y=[]))
line_on = plot.line(source=line_on_source, x="x", y="y", line_color="blue", line_width=2)

line_diff_source = ColumnDataSource(dict(x=[], y=[]))
line_diff = plot.line(source=line_diff_source, x="x", y="y", line_color="red", line_width=2)

plot.add_layout(
    Legend(
        items=[("Laser off", [line_off]), ("Laser on", [line_on]), ("Difference", [line_diff])],
        location="top_left",
    )
)
plot.legend.click_policy = "hide"

step_size = stats.projections_lon[0].step_size
max_span = stats.projections_lon[0].max_span
average_window_spinner = Spinner(
    title="Pulse ID Window:", value=step_size, low=step_size, high=max_span, step=step_size
)
frames_off_spinner = Spinner(title="Frames laser off:", value=0, disabled=True)
frames_on_spinner = Spinner(title="Frames laser on:", value=0, disabled=True)


def reset_button_callback():
    for projection in stats.projections_lon + stats.projections_loff:
        projection.clear()


reset_button = Button(label="Reset", button_type="default")
reset_button.on_click(reset_button_callback)


roi_select = Select(title="Select ROI", value="0", options=list(map(str, range(10))))


def update():
    projection_lon = stats.projections_lon[int(roi_select.value)]
    projection_loff = stats.projections_loff[int(roi_select.value)]
    x_on, avg_I_on, num_on = projection_lon(average_window_spinner.value)
    x_off, avg_I_off, num_off = projection_loff(average_window_spinner.value)

    frames_off_spinner.value = num_off
    frames_on_spinner.value = num_on

    line_off_source.data.update(x=x_off, y=avg_I_off)
    line_on_source.data.update(x=x_on, y=avg_I_on)

    if num_off and num_on:
        # in this case, x_on is equal to x_off
        line_diff_source.data.update(x=x_on, y=avg_I_on - avg_I_off)
    else:
        line_diff_source.data.update(x=[], y=[])


doc.add_root(
    column(
        column(plot, sizing_mode="stretch_both"),
        row(
            average_window_spinner,
            frames_off_spinner,
            frames_on_spinner,
            column(Spacer(height=19), reset_button),
            roi_select,
        ),
        sizing_mode="stretch_width",
    )
)
doc.add_periodic_callback(update, 1000)
