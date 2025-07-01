from time import time
from typing import Any

import colorcet as cc
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, ColorBar, Spinner, Select, Spacer, CheckboxGroup
from bokeh.palettes import Cividis256, Greys256, Plasma256
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from numpy import ndarray, dtype

cmap_dict = {
    "gray": Greys256,
    "gray_r": Greys256[::-1],
    "plasma": Plasma256,
    "coolwarm": cc.coolwarm,
    "cividis": Cividis256,
}

def extend_last_value(arr: ndarray[Any, dtype[np.floating]], num_ext: int) -> ndarray[Any, dtype[Any]]:
    return np.hstack([arr, np.full(shape=(num_ext,), fill_value=arr[-1])])

class ScatterPlot:
    """
    Colour-mapped scatter plot of value against X and Y coordinates
    """

    def __init__(self, width, height, glyph_size, title, colormap="coolwarm",
                 x_step_mm=20e-2, y_step_mm=10e-1, x_size_mm=30., y_size_mm=10.,
                 slow_step_delay_frames=2,
                 frame_rate_hz=100, snake=True):

        self.cmap = cmap_dict.get(colormap, "coolwarm")

        self.snake = snake
        self.pulse_id_increment = 100 / frame_rate_hz
        xnum = int(x_size_mm // x_step_mm) + 1
        ynum = int(y_size_mm // y_step_mm) + 1

        self.x_coords_direct = np.linspace(0, x_size_mm, xnum, endpoint=True)
        self.x_coords_inverse = np.flip(self.x_coords_direct)
        # Extend with "hang" points on slow motor move
        self.x_coords_direct = extend_last_value(self.x_coords_direct, slow_step_delay_frames)
        self.x_coords_inverse = extend_last_value(self.x_coords_inverse, slow_step_delay_frames)

        self.y_coords = np.linspace(0, y_size_mm, ynum, endpoint=True)
        self.shape = (xnum, ynum)
        self.max_index = xnum * ynum
        print(f"Shape is {self.shape}")

        self.val = []
        self.x = []
        self.y = []
        self.rel_pulse_ids = []
        self.first_pulse_id = None

        self.size = 0

        self.data_source = ColumnDataSource(dict(x=self.x, y=self.y, val=self.val))

        self.cmap = linear_cmap(
            field_name='val',
            palette=cmap_dict.get(colormap, "plasma"),
            low=-1000, high=1000
        )
        self.mapper = self.cmap["transform"]

        self.plot = figure(
            width=width,
            height=height,
            title=title,
            tools="pan,wheel_zoom,xwheel_zoom,ywheel_zoom,box_zoom,save,reset"
        )

        self.renderer = self.plot.square(
            x='x', y='y',
            line_color=self.cmap,
            color=self.cmap,
            fill_alpha=1,
            size=glyph_size,
            source=self.data_source
        )

        self.color_bar = ColorBar(
            color_mapper=self.mapper,
            location=(0, -5),
            orientation="horizontal",
            height=20,
            width=width,
            padding=5,
        )
        self.plot.add_layout(
            self.color_bar, place="below"
        )

        #-------------------------- Controls -----------------------------------------#

        # Glyph size
        def glyph_size_chage_callback(_attr, _old_value, new_value):
            self.renderer.glyph.size = new_value

        self.glyph_size_spinner = Spinner(
            title="Glyph Size",
            high=100,
            value=glyph_size,
            step=5,
            disabled=False,
            width=145,
        )
        self.glyph_size_spinner.on_change("value", glyph_size_chage_callback)

        # Color map selector
        def select_callback(_attr, _old, new):
            if new in cmap_dict:
                self.mapper.palette = cmap_dict[new]

        self.cmap_select = Select(
            title="Colormap:", value=colormap, options=list(cmap_dict.keys()), width=100
        )
        self.cmap_select.on_change("value", select_callback)

        # Snake switch
        def snake_switch_callback(_attr, _old, new):
            print(f"Snake switch switched to {new}")
            self.snake = 0 in new
            self.reindex_xy()

        self.snake_switch = CheckboxGroup(labels=["Snake scan"], width=145, active=[0])
        self.snake_switch.on_change("active", snake_switch_callback)

        # Scan parameters


    def update(self, values: list, pulse_ids: list):
        print(f"Proc {len(values)} bragg frames")
        for v, pid in zip(values, pulse_ids):
            self.update_one(value=v, pulse_id=pid)

    def update_one(self, value, pulse_id):
        if self.first_pulse_id is None:
            self.first_pulse_id = pulse_id
        if pulse_id < self.first_pulse_id:
            print(f"Got pulse id {pulse_id} less than {self.first_pulse_id}. Shift and replot.")
            self.rel_pulse_ids = [0] + (np.array(self.rel_pulse_ids) + 1).tolist()
            self.val = [value] + self.val
            self.reindex_xy()
            return
        pulse_index = int((pulse_id - self.first_pulse_id) / self.pulse_id_increment)
        if pulse_index >= self.max_index:
            print(f"Maximum index reached, rolling over")
            pulse_index -= self.max_index
            self.first_pulse_id = pulse_id - pulse_index
        self.rel_pulse_ids.append(pulse_index)
        self.val.append(value)
        if max(self.val) != min(self.val):
            self.mapper.low = min(self.val)
            self.mapper.high = max(self.val)

        self.add_x_y(pulse_index)
        self.size += 1

        self.data_source.data.update(x=self.x, y=self.y, val=self.val)

    def reindex_xy(self):
        self.x = []
        self.y = []
        t0 = time()
        # TODO: optimize
        for pulse_index in self.rel_pulse_ids:
            self.add_x_y(pulse_index)
        t1 = time()
        print(f"Reindexing axes took {t1 - t0} for {len(self.rel_pulse_ids)} values")
        self.data_source.data.update(x=self.x, y=self.y, val=self.val)

    def add_x_y(self, pulse_index):
        x_index, y_index = np.unravel_index(pulse_index, self.shape, order="F")
        if self.snake and y_index % 2 == 1:
            self.x.append(self.x_coords_inverse[x_index])
        else:
            self.x.append(self.x_coords_direct[x_index])
        self.y.append(self.y_coords[y_index])

    def clear(self):
        self.val = []
        self.x = []
        self.y = []
        self.rel_pulse_ids = []
        self.first_pulse_id = None
        self.size = 0
        self.data_source.data.update(x=self.x, y=self.y, val=self.val)

    @property
    def default_layout(self):
        return column(
            self.plot,
            row(
                self.cmap_select,
                Spacer(width=15),
                self.glyph_size_spinner,
                Spacer(width=15),
                self.snake_switch
            )
        )
