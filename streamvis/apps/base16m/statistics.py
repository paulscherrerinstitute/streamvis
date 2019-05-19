from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, DataTable, NumberFormatter, TableColumn

import receiver

doc = curdoc()
doc.title = f"{receiver.args.page_title} Statistics"

table_columns = [
    TableColumn(field='run_names', title="Run Name"),
    TableColumn(field='nframes', title="Total Frames"),
    TableColumn(field='bad_frames', title="Bad Frames"),
    TableColumn(field='sat_pix_nframes', title="Sat pix frames"),
    TableColumn(field='laser_on_nframes', title="Laser ON frames"),
    TableColumn(field='laser_on_hits', title="Laser ON hits"),
    TableColumn(
        field='laser_on_hits_ratio',
        title="Laser ON hits ratio",
        formatter=NumberFormatter(format='(0.00 %)'),
    ),
    TableColumn(field='laser_off_nframes', title="Laser OFF frames"),
    TableColumn(field='laser_off_hits', title="Laser OFF hits"),
    TableColumn(
        field='laser_off_hits_ratio',
        title="Laser OFF hits ratio",
        formatter=NumberFormatter(format='(0.00 %)'),
    ),
]

table_source = ColumnDataSource(receiver.stats_table_dict)
table = DataTable(source=table_source, columns=table_columns, height=50, index_position=None)

sum_table_source = ColumnDataSource(receiver.sum_stats_table_dict)
sum_table = DataTable(
    source=sum_table_source, columns=table_columns, height=50, index_position=None,
)


# update statistics callback
def update_statistics():
    table_source.data = receiver.stats_table_dict
    sum_table_source.data = receiver.sum_stats_table_dict


# reset statistics button
def reset_stats_button_callback():
    receiver.run_name = ''

    receiver.run_names.clear()
    receiver.nframes.clear()
    receiver.bad_frames.clear()
    receiver.sat_pix_nframes.clear()
    receiver.laser_on_nframes.clear()
    receiver.laser_on_hits.clear()
    receiver.laser_on_hits_ratio.clear()
    receiver.laser_off_nframes.clear()
    receiver.laser_off_hits.clear()
    receiver.laser_off_hits_ratio.clear()

    receiver.sum_nframes[0] = 0
    receiver.sum_bad_frames[0] = 0
    receiver.sum_sat_pix_nframes[0] = 0
    receiver.sum_laser_on_nframes[0] = 0
    receiver.sum_laser_on_hits[0] = 0
    receiver.sum_laser_on_hits_ratio[0] = 0
    receiver.sum_laser_off_nframes[0] = 0
    receiver.sum_laser_off_hits[0] = 0
    receiver.sum_laser_off_hits_ratio[0] = 0


reset_stats_button = Button(label="Reset Statistics", button_type='default')
reset_stats_button.on_click(reset_stats_button_callback)

layout = column(
    column(table, sizing_mode="stretch_both"),
    sum_table,
    reset_stats_button,
    sizing_mode="stretch_width",
)

doc.add_root(layout)
doc.add_periodic_callback(update_statistics, 1000)
