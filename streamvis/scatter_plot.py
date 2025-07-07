import logging
from threading import Lock
from time import time

import colorcet as cc
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, ColorBar, Spinner, Select, Spacer, CheckboxGroup
from bokeh.palettes import Cividis256, Greys256, Plasma256
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

cmap_dict = {
    "gray": Greys256,
    "gray_r": Greys256[::-1],
    "plasma": Plasma256,
    "coolwarm": cc.coolwarm,
    "cividis": Cividis256,
}

logger = logging.getLogger(__name__)


def extend_last_value(arr, num_ext: int):
    return np.hstack([arr, np.full(shape=(num_ext,), fill_value=arr[-1])])


class ScatterPlot:
    """
    Colour-mapped scatter plot of value against X and Y coordinates
    """

    def __init__(self, width, height, glyph_size, title, colormap="coolwarm",
                 x_step_mm=20e-2, y_step_mm=10e-1, x_size_mm=30., y_size_mm=10.,
                 slow_step_delay_frames=2, frame_rate_hz=100):

        self._lock = Lock()
        self.cmap = cmap_dict.get(colormap, "coolwarm")

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
            logger.info(f"Snake switch switched to {new}")
            with self._lock:
                self.reindex_xy()

        self.snake_switch = CheckboxGroup(labels=["Snake scan"], width=145, active=[0])
        self.snake_switch.on_change("active", snake_switch_callback)

        # Scan parameters
        self.x_step_um_spinner = Spinner(
            title="Step X (um)",
            high=1000,
            value=x_step_mm*1e3,
            step=10,
            disabled=False,
            width=145,
        )
        self.y_step_um_spinner = Spinner(
            title="Step Y (um)",
            high=1000,
            value=y_step_mm * 1e3,
            step=10,
            disabled=False,
            width=145,
        )
        self.x_size_mm_spinner = Spinner(
            title="Size X (mm)",
            high=30,
            value=x_size_mm,
            step=10,
            disabled=False,
            width=145,
        )
        self.y_size_mm_spinner = Spinner(
            title="Size Y (mm)",
            high=10,
            value=y_size_mm,
            step=10,
            disabled=False,
            width=145,
        )
        self.slow_step_delay_frames_spinner = Spinner(
            title="Slow step delay (# Pulses)",
            high=1000,
            value=slow_step_delay_frames,
            step=10,
            disabled=False,
            width=145,
        )

        def scan_parameters_changed_callback(_attr, _old_value, new_value):
            with self._lock:
                self.calculate_coords()
                self.reindex_xy()

        self.x_step_um_spinner.on_change("value", scan_parameters_changed_callback)
        self.x_size_mm_spinner.on_change("value", scan_parameters_changed_callback)
        self.y_step_um_spinner.on_change("value", scan_parameters_changed_callback)
        self.y_size_mm_spinner.on_change("value", scan_parameters_changed_callback)
        self.slow_step_delay_frames_spinner.on_change("value", scan_parameters_changed_callback)

        self.frame_rate_hz_spinner = Spinner(
            title="Repetition Rate (Hz)",
            high=100,
            value=frame_rate_hz,
            step=10,
            disabled=False,
            width=145,
        )

        def frame_rate_changed_callback(_attr, _old_value, new_value):
            with self._lock:
                self.reindex_xy()

        self.frame_rate_hz_spinner.on_change("value", frame_rate_changed_callback)

        self.calculate_coords()


    def update(self, values: list, pulse_ids: list):
        with self._lock:
            logger.debug(f"Proc {len(values)} bragg frames")
            for v, pid in zip(values, pulse_ids):
                self.update_one(value=v, pulse_id=pid)

    def update_one(self, value, pulse_id):
        if self.first_pulse_id is None:
            self.first_pulse_id = pulse_id
        if pulse_id < self.first_pulse_id:
            logger.info(f"Got pulse id {pulse_id} less than {self.first_pulse_id}. Shift and replot.")
            self.rel_pulse_ids = [0] + (np.array(self.rel_pulse_ids) + 1).tolist()
            self.val = [value] + self.val
            self.reindex_xy()
            return
        pulse_index = pulse_id - self.first_pulse_id
        logger.debug(f"Pulse index is {pulse_index}")
        self.rel_pulse_ids.append(pulse_index)
        self.val.append(value)
        if max(self.val) != min(self.val):
            self.mapper.low = min(self.val)
            self.mapper.high = max(self.val)

        self.add_x_y(pulse_index)
        self.size += 1
        self.data_source.data.update(x=self.x, y=self.y, val=self.val)

    def calculate_coords(self):
        self.x_coords_direct = np.linspace(0, self.x_size_mm, self.xnum, endpoint=True)
        self.x_coords_inverse = np.flip(self.x_coords_direct)
        # Extend with "hang" points on slow motor move
        self.x_coords_direct = extend_last_value(self.x_coords_direct, self.slow_step_delay_frames)
        self.x_coords_inverse = extend_last_value(self.x_coords_inverse, self.slow_step_delay_frames)

        self.y_coords = np.linspace(0, self.y_size_mm, self.ynum, endpoint=True)
        logger.info(f"New Shape is {self.shape}, Max index {self.max_index}")

    def reindex_xy(self):
        self.x = []
        self.y = []
        t0 = time()
        # TODO: optimize
        for pulse_index in self.rel_pulse_ids:
            self.add_x_y(pulse_index)
        t1 = time()
        logger.info(f"Reindexing axes took {t1 - t0} for {len(self.rel_pulse_ids)} values")
        self.data_source.data.update(x=self.x, y=self.y, val=self.val)

    def add_x_y(self, pulse_index):
        xy_pulse_index = int(pulse_index / self.pulse_id_increment) % self.max_index
        x_index, y_index = np.unravel_index(
            xy_pulse_index,
            self.shape,
            order="F"
        )
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
        return row(
            self.plot,
            column(
                row(
                    self.x_step_um_spinner,
                    Spacer(width=15),
                    self.x_size_mm_spinner,
                ),
                Spacer(height=10),
                row(
                    self.y_step_um_spinner,
                    Spacer(width=15),
                    self.y_size_mm_spinner,
                ),
                Spacer(height=10),
                row(
                    self.slow_step_delay_frames_spinner,
                    Spacer(width=15),
                    self.frame_rate_hz_spinner,
                ),
                Spacer(height=10),
                self.snake_switch,
                Spacer(height=50),
                row(
                    self.cmap_select,
                    Spacer(width=15),
                    self.glyph_size_spinner,
                ),
            )
        )

    @property
    def snake(self):
        return 0 in self.snake_switch.active

    @property
    def frame_rate_hz(self):
        return self.frame_rate_hz_spinner.value

    @property
    def pulse_id_increment(self):
        return 100 / self.frame_rate_hz

    @property
    def x_size_mm(self):
        return self.x_size_mm_spinner.value

    @property
    def y_size_mm(self):
        return self.y_size_mm_spinner.value

    @property
    def x_step_mm(self):
        return self.x_step_um_spinner.value * 1e-3

    @property
    def y_step_mm(self):
        return self.y_step_um_spinner.value * 1e-3

    @property
    def slow_step_delay_frames(self):
        return int(self.slow_step_delay_frames_spinner.value)

    @property
    def xnum(self):
        return int(self.x_size_mm // self.x_step_mm) + 1 + self.slow_step_delay_frames

    @property
    def ynum(self):
        return int(self.y_size_mm // self.y_step_mm) + 1

    @property
    def shape(self):
        return (self.xnum, self.ynum)

    @property
    def max_index(self):
        return self.xnum * self.ynum
