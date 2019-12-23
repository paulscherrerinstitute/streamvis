from copy import copy

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, DataTable, NumberFormatter, TableColumn

doc = curdoc()
stats = doc.stats
doc.title = f"{doc.title} Statistics"

all_table_columns = {
    'run_names': TableColumn(field='run_names', title="Run Name"),
    'nframes': TableColumn(field='nframes', title="Total Frames"),
    'bad_frames': TableColumn(field='bad_frames', title="Bad Frames"),
    'sat_pix_nframes': TableColumn(field='sat_pix_nframes', title="Sat pix frames"),
    'laser_on_nframes': TableColumn(field='laser_on_nframes', title="Laser ON frames"),
    'laser_on_hits': TableColumn(field='laser_on_hits', title="Laser ON hits"),
    'laser_on_hits_ratio': TableColumn(
        field='laser_on_hits_ratio',
        title="Laser ON hits ratio",
        formatter=NumberFormatter(format='(0.00 %)'),
    ),
    'laser_off_nframes': TableColumn(field='laser_off_nframes', title="Laser OFF frames"),
    'laser_off_hits': TableColumn(field='laser_off_hits', title="Laser OFF hits"),
    'laser_off_hits_ratio': TableColumn(
        field='laser_off_hits_ratio',
        title="Laser OFF hits ratio",
        formatter=NumberFormatter(format='(0.00 %)'),
    ),
}

table_columns = copy(all_table_columns)

table_source = ColumnDataSource(stats.data)
table = DataTable(
    source=table_source, columns=list(table_columns.values()), height=50, index_position=None
)

sum_table_source = ColumnDataSource(stats.sum_data)
sum_table = DataTable(
    source=sum_table_source, columns=list(table_columns.values()), height=50, index_position=None
)


# update statistics callback
def update_statistics():
    update_columns = False
    if np.all(np.isnan(stats.data['sat_pix_nframes'])):
        if 'sat_pix_nframes' in table_columns:
            del table_columns['sat_pix_nframes']
            update_columns = True

    else:
        if 'sat_pix_nframes' not in table_columns:
            table_columns['sat_pix_nframes'] = all_table_columns['sat_pix_nframes']
            update_columns = True

    if np.all(np.isnan(stats.data['laser_on_nframes'])):
        if 'laser_on_nframes' in table_columns:
            del table_columns['laser_on_nframes']
            del table_columns['laser_on_hits']
            del table_columns['laser_on_hits_ratio']
            del table_columns['laser_off_nframes']
            del table_columns['laser_off_hits']
            del table_columns['laser_off_hits_ratio']
            update_columns = True

    else:
        if 'laser_on_nframes' not in table_columns:
            table_columns['laser_on_nframes'] = all_table_columns['laser_on_nframes']
            table_columns['laser_on_hits'] = all_table_columns['laser_on_hits']
            table_columns['laser_on_hits_ratio'] = all_table_columns['laser_on_hits_ratio']
            table_columns['laser_off_nframes'] = all_table_columns['laser_off_nframes']
            table_columns['laser_off_hits'] = all_table_columns['laser_off_hits']
            table_columns['laser_off_hits_ratio'] = all_table_columns['laser_off_hits_ratio']
            update_columns = True

    if update_columns:
        table.columns = list(table_columns.values())
        sum_table.columns = list(table_columns.values())

    table_source.data = stats.data
    sum_table_source.data = stats.sum_data


# reset statistics button
def reset_stats_button_callback():
    stats.reset()


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
