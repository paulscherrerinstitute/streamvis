import colorcet as cc
import numpy as np
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, \
    LogColorMapper, LogTicker, RadioButtonGroup, Select, TextInput, Toggle
from bokeh.palettes import Cividis256, Greys256, Plasma256  # pylint: disable=E0611
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize


class ColorMapper:
    def __init__(self, init_disp_min=0, init_disp_max=1000):
        self._disp_min = init_disp_min
        self._disp_max = init_disp_max

        lin_colormapper = LinearColorMapper(
            palette=Plasma256,
            low=init_disp_min,
            high=init_disp_max,
        )

        log_colormapper = LogColorMapper(
            palette=Plasma256,
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

        color_lin_norm = Normalize()
        color_log_norm = LogNorm()
        self._image_color_mapper = ScalarMappable(norm=color_lin_norm, cmap='plasma')

        # ---- colormap selector
        def select_callback(_attr, _old, new):
            self._image_color_mapper.set_cmap(new)
            if new == 'gray':
                lin_colormapper.palette = Greys256
                log_colormapper.palette = Greys256

            elif new == 'gray_r':
                lin_colormapper.palette = Greys256[::-1]
                log_colormapper.palette = Greys256[::-1]

            elif new == 'plasma':
                lin_colormapper.palette = Plasma256
                log_colormapper.palette = Plasma256

            elif new == 'coolwarm':
                lin_colormapper.palette = cc.coolwarm
                log_colormapper.palette = cc.coolwarm

            elif new == 'cividis':
                lin_colormapper.palette = Cividis256
                log_colormapper.palette = Cividis256

        select = Select(
            title="Colormap:", value='plasma',
            options=['gray', 'gray_r', 'plasma', 'coolwarm', 'cividis'],
        )
        select.on_change('value', select_callback)
        self.select = select

        # ---- colormap auto toggle button
        def auto_toggle_callback(state):
            if state:
                display_min_textinput.disabled = True
                display_max_textinput.disabled = True
            else:
                display_min_textinput.disabled = False
                display_max_textinput.disabled = False

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
        def display_max_callback(_attr, old, new):
            try:
                new_value = float(new)
                if new_value > self._disp_min:
                    if new_value <= 0:
                        scale_radiobuttongroup.active = 0

                    self._disp_max = new_value
                    color_lin_norm.vmax = self._disp_max
                    color_log_norm.vmax = self._disp_max
                    lin_colormapper.high = self._disp_max
                    log_colormapper.high = self._disp_max
                else:
                    display_max_textinput.value = old

            except ValueError:
                display_max_textinput.value = old

        display_max_textinput = TextInput(
            title='Maximal Display Value:', value=str(init_disp_max), disabled=auto_toggle.active)
        display_max_textinput.on_change('value', display_max_callback)
        self.display_max_textinput = display_max_textinput

        # ---- colormap display min value
        def display_min_callback(_attr, old, new):
            try:
                new_value = float(new)
                if new_value < self._disp_max:
                    if new_value <= 0:
                        scale_radiobuttongroup.active = 0

                    self._disp_min = new_value
                    color_lin_norm.vmin = self._disp_min
                    color_log_norm.vmin = self._disp_min
                    lin_colormapper.low = self._disp_min
                    log_colormapper.low = self._disp_min
                else:
                    display_min_textinput.value = old

            except ValueError:
                display_min_textinput.value = old

        display_min_textinput = TextInput(
            title='Minimal Display Value:', value=str(init_disp_min), disabled=auto_toggle.active)
        display_min_textinput.on_change('value', display_min_callback)
        self.display_min_textinput = display_min_textinput

    def update(self, image):
        if self.auto_toggle.active:
            self._disp_min = int(np.min(image))
            if self._disp_min <= 0:  # switch to linear colormap
                self.scale_radiobuttongroup.active = 0
            self.display_min_textinput.value = str(self._disp_min)

            self._disp_max = int(np.max(image))
            self.display_max_textinput.value = str(self._disp_max)

    def convert(self, image):
        return self._image_color_mapper.to_rgba(image, bytes=True)
