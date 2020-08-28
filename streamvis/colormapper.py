import colorcet as cc
import numpy as np
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColorPicker,
    LinearColorMapper,
    LogColorMapper,
    LogTicker,
    RadioButtonGroup,
    Select,
    Spinner,
    Toggle,
)

from bokeh.palettes import Cividis256, Greys256, Plasma256

cmap_dict = {
    "gray": Greys256,
    "gray_r": Greys256[::-1],
    "plasma": Plasma256,
    "coolwarm": cc.coolwarm,
    "cividis": Cividis256,
}

# TODO: Can be changed back to 0.1 when https://github.com/bokeh/bokeh/issues/9408 is fixed
STEP = 1


class ColorMapper:
    def __init__(self, image_views, disp_min=0, disp_max=1000, colormap="plasma"):
        """Initialize a colormapper.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            disp_min (int, optional): Initial minimal display value. Defaults to 0.
            disp_max (int, optional): Initial maximal display value. Defaults to 1000.
            colormap (str, optional): Initial colormap. Defaults to 'plasma'.
        """
        lin_colormapper = LinearColorMapper(
            palette=cmap_dict[colormap], low=disp_min, high=disp_max
        )

        log_colormapper = LogColorMapper(palette=cmap_dict[colormap], low=disp_min, high=disp_max)

        for image_view in image_views:
            image_view.image_glyph.color_mapper = lin_colormapper

        color_bar = ColorBar(
            color_mapper=lin_colormapper,
            location=(0, -5),
            orientation="horizontal",
            height=15,
            width=100,
            padding=5,
        )
        self.color_bar = color_bar

        # ---- colormap selector
        def select_callback(_attr, _old, new):
            if new in cmap_dict:
                lin_colormapper.palette = cmap_dict[new]
                log_colormapper.palette = cmap_dict[new]
                display_high_color.color = cmap_dict[new][-1]

        select = Select(title="Colormap:", value=colormap, options=list(cmap_dict.keys()))
        select.on_change("value", select_callback)
        self.select = select

        # ---- colormap auto toggle button
        def auto_toggle_callback(state):
            if state:
                display_min_spinner.disabled = True
                display_max_spinner.disabled = True
            else:
                display_min_spinner.disabled = False
                display_max_spinner.disabled = False

        auto_toggle = Toggle(label="Auto", active=False, button_type="default")
        auto_toggle.on_click(auto_toggle_callback)
        self.auto_toggle = auto_toggle

        # ---- colormap scale radiobutton group
        def scale_radiobuttongroup_callback(selection):
            if selection == 0:  # Linear
                for image_view in image_views:
                    image_view.image_glyph.color_mapper = lin_colormapper
                color_bar.color_mapper = lin_colormapper
                color_bar.ticker = BasicTicker()

            else:  # Logarithmic
                if self.disp_min > 0:
                    for image_view in image_views:
                        image_view.image_glyph.color_mapper = log_colormapper
                    color_bar.color_mapper = log_colormapper
                    color_bar.ticker = LogTicker()
                else:
                    scale_radiobuttongroup.active = 0

        scale_radiobuttongroup = RadioButtonGroup(labels=["Linear", "Logarithmic"], active=0)
        scale_radiobuttongroup.on_click(scale_radiobuttongroup_callback)
        self.scale_radiobuttongroup = scale_radiobuttongroup

        # ---- colormap display max value
        def display_max_spinner_callback(_attr, _old_value, new_value):
            self.display_min_spinner.high = new_value - STEP
            if new_value <= 0:
                scale_radiobuttongroup.active = 0

            lin_colormapper.high = new_value
            log_colormapper.high = new_value

        display_max_spinner = Spinner(
            title="Maximal Display Value:",
            low=disp_min + STEP,
            value=disp_max,
            step=STEP,
            disabled=auto_toggle.active,
        )
        display_max_spinner.on_change("value", display_max_spinner_callback)
        self.display_max_spinner = display_max_spinner

        # ---- colormap display min value
        def display_min_spinner_callback(_attr, _old_value, new_value):
            self.display_max_spinner.low = new_value + STEP
            if new_value <= 0:
                scale_radiobuttongroup.active = 0

            lin_colormapper.low = new_value
            log_colormapper.low = new_value

        display_min_spinner = Spinner(
            title="Minimal Display Value:",
            high=disp_max - STEP,
            value=disp_min,
            step=STEP,
            disabled=auto_toggle.active,
        )
        display_min_spinner.on_change("value", display_min_spinner_callback)
        self.display_min_spinner = display_min_spinner

        # ---- colormap high color
        def display_high_color_callback(_attr, _old_value, new_value):
            lin_colormapper.high_color = new_value
            log_colormapper.high_color = new_value

        display_high_color = ColorPicker(title="High Value Color:", color=cmap_dict[colormap][-1])
        display_high_color.on_change("color", display_high_color_callback)
        self.display_high_color = display_high_color

    @property
    def disp_min(self):
        """Minimal display value (readonly)
        """
        return self.display_min_spinner.value

    @property
    def disp_max(self):
        """Maximal display value (readonly)
        """
        return self.display_max_spinner.value

    def update(self, image):
        """Trigger an update for the colormapper.

        Args:
            image (ndarray): A source image for colormapper.
        """
        if self.auto_toggle.active:
            image_min = int(np.min(image))
            image_max = int(np.max(image))

            if image_min <= 0:  # switch to linear colormap
                self.scale_radiobuttongroup.active = 0

            # force update independently on current display values
            self.display_max_spinner.value = np.inf

            self.display_min_spinner.value = image_min
            self.display_max_spinner.value = image_max
