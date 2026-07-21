from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, DataRange1d, Legend, Spacer, Spinner
from bokeh.plotting import figure

doc = curdoc()
stats = doc.stream_adapter.stats
doc.title = f"{doc.title} Radial Profile"

plot = figure(
    title="Radial Profile",
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

step_size = stats.radial_profile_lon.step_size
max_span = stats.radial_profile_lon.max_span
average_window_spinner = Spinner(
    title="Pulse ID Window:", value=step_size, low=step_size, high=max_span, step=step_size
)
frames_off_spinner = Spinner(title="Frames laser off:", value=0, disabled=True)
frames_on_spinner = Spinner(title="Frames laser on:", value=0, disabled=True)


def reset_button_callback():
    stats.radial_profile_lon.clear()
    stats.radial_profile_loff.clear()


reset_button = Button(label="Reset", button_type="default")
reset_button.on_click(reset_button_callback)


def update():
    q_on, avg_I_on, num_on = stats.radial_profile_lon(average_window_spinner.value)
    q_off, avg_I_off, num_off = stats.radial_profile_loff(average_window_spinner.value)

    frames_off_spinner.value = num_off
    frames_on_spinner.value = num_on

    line_off_source.data.update(x=q_off, y=avg_I_off)
    line_on_source.data.update(x=q_on, y=avg_I_on)

    if num_off and num_on:
        # in this case, q_on is equal to q_off
        line_diff_source.data.update(x=q_on, y=avg_I_on - avg_I_off)
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
        ),
        sizing_mode="stretch_width",
    )
)
doc.add_periodic_callback(update, 1000)
