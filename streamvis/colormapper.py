import colorcet as cc
import numpy as np
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, \
    LogColorMapper, LogTicker, RadioButtonGroup, Select, Spinner, Toggle
from bokeh.palettes import Cividis256, Greys256, Plasma256  # pylint: disable=E0611
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

cmap_dict = {
    'gray': Greys256,
    'gray_r': Greys256[::-1],
    'plasma': Plasma256,
    'coolwarm': cc.coolwarm,
    'cividis': Cividis256,
}


class ColorMapper:
    def __init__(self, init_disp_min=0, init_disp_max=1000, init_colormap='plasma'):
        self._disp_min = init_disp_min
        self._disp_max = init_disp_max

        lin_colormapper = LinearColorMapper(
            palette=cmap_dict[init_colormap],
            low=init_disp_min,
            high=init_disp_max,
        )

        log_colormapper = LogColorMapper(
            palette=cmap_dict[init_colormap],
            low=init_disp_min,
            high=init_disp_max,
        )

        color_bar = ColorBar(
            color_mapper=lin_colormapper,
            location=(0, 0),
            orientation='horizontal',
            height=15,
            width=100,
            padding=0,
        )
        self.color_bar = color_bar

        color_lin_norm = Normalize(vmin=init_disp_min, vmax=init_disp_max)
        color_log_norm = LogNorm(vmin=init_disp_min, vmax=init_disp_max)

        self._image_color_mapper = ScalarMappable(norm=color_lin_norm, cmap=init_colormap)

        # ---- colormap selector
        def select_callback(_attr, _old, new):
            self._image_color_mapper.set_cmap(new)
            if new in cmap_dict:
                lin_colormapper.palette = cmap_dict[new]
                log_colormapper.palette = cmap_dict[new]

        select = Select(
            title="Colormap:",
            value=init_colormap,
            options=list(cmap_dict.keys()),
        )
        select.on_change('value', select_callback)
        self.select = select

        # ---- colormap auto toggle button
        def auto_toggle_callback(state):
            if state:
                display_min_spinner.disabled = True
                display_max_spinner.disabled = True
            else:
                display_min_spinner.disabled = False
                display_max_spinner.disabled = False

        auto_toggle = Toggle(label="Auto", active=False, button_type='default')
        auto_toggle.on_click(auto_toggle_callback)
        self.auto_toggle = auto_toggle

        # ---- colormap scale radiobutton group
        def scale_radiobuttongroup_callback(selection):
            if selection == 0:  # Linear
                color_bar.color_mapper = lin_colormapper
                color_bar.ticker = BasicTicker()
                self._image_color_mapper.norm = color_lin_norm

            else:  # Logarithmic
                if self._disp_min > 0:
                    color_bar.color_mapper = log_colormapper
                    color_bar.ticker = LogTicker()
                    self._image_color_mapper.norm = color_log_norm
                else:
                    scale_radiobuttongroup.active = 0

        scale_radiobuttongroup = RadioButtonGroup(labels=["Linear", "Logarithmic"], active=0)
        scale_radiobuttongroup.on_click(scale_radiobuttongroup_callback)
        self.scale_radiobuttongroup = scale_radiobuttongroup

        # ---- colormap display max value
        def display_max_spinner_callback(_attr, old_value, new_value):
            if new_value > self._disp_min:
                self._disp_max = new_value

                if new_value <= 0:
                    scale_radiobuttongroup.active = 0

                color_lin_norm.vmax = self._disp_max
                color_log_norm.vmax = self._disp_max
                lin_colormapper.high = self._disp_max
                log_colormapper.high = self._disp_max
            else:
                display_max_spinner.value = old_value

        display_max_spinner = Spinner(
            title='Maximal Display Value:', value=init_disp_max, step=0.1,
            disabled=auto_toggle.active,
        )
        display_max_spinner.on_change('value', display_max_spinner_callback)
        self.display_max_spinner = display_max_spinner

        # ---- colormap display min value
        def display_min_spinner_callback(_attr, old_value, new_value):
            if new_value < self._disp_max:
                self._disp_min = new_value

                if new_value <= 0:
                    scale_radiobuttongroup.active = 0

                color_lin_norm.vmin = self._disp_min
                color_log_norm.vmin = self._disp_min
                lin_colormapper.low = self._disp_min
                log_colormapper.low = self._disp_min
            else:
                display_min_spinner.value = old_value

        display_min_spinner = Spinner(
            title='Minimal Display Value:', value=init_disp_min, step=0.1,
            disabled=auto_toggle.active,
        )
        display_min_spinner.on_change('value', display_min_spinner_callback)
        self.display_min_spinner = display_min_spinner

    def update(self, image):
        if self.auto_toggle.active:
            image_min = int(np.min(image))
            image_max = int(np.max(image))

            if image_min <= 0:  # switch to linear colormap
                self.scale_radiobuttongroup.active = 0

            self._disp_max = np.inf  # force update independently on current display values
            self.display_min_spinner.value = image_min
            self.display_max_spinner.value = image_max

    def convert(self, image):
        return self._image_color_mapper.to_rgba(image, bytes=True)
