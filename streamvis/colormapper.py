import colorcet as cc
import numpy as np
from bokeh.models import (
    BasicTicker,
    CheckboxGroup,
    ColorBar,
    ColorPicker,
    LinearColorMapper,
    LogColorMapper,
    LogTicker,
    RadioGroup,
    Select,
    Spinner,
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
STEP = 0.1


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

        # ---- selector
        def select_callback(_attr, _old, new):
            if new in cmap_dict:
                lin_colormapper.palette = cmap_dict[new]
                log_colormapper.palette = cmap_dict[new]
                high_color.color = cmap_dict[new][-1]

        select = Select(
            title="Colormap:", value=colormap, options=list(cmap_dict.keys()), width=100
        )
        select.on_change("value", select_callback)
        self.select = select

        # ---- auto switch
        def auto_switch_callback(_attr, _old, new):
            if 0 in new:
                display_min_spinner.disabled = True
                display_max_spinner.disabled = True
            else:
                display_min_spinner.disabled = False
                display_max_spinner.disabled = False

        auto_switch = CheckboxGroup(labels=["Auto Colormap Range"], width=145)
        auto_switch.on_change("active", auto_switch_callback)
        self.auto_switch = auto_switch

        # ---- scale radiogroup
        def scale_radiogroup_callback(_attr, _old, new):
            if new == 0:  # Linear
                for image_view in image_views:
                    image_view.image_glyph.color_mapper = lin_colormapper
                color_bar.color_mapper = lin_colormapper
                color_bar.ticker = BasicTicker()

            else:  # Logarithmic
                if self.display_min_spinner.value > 0:
                    for image_view in image_views:
                        image_view.image_glyph.color_mapper = log_colormapper
                    color_bar.color_mapper = log_colormapper
                    color_bar.ticker = LogTicker()
                else:
                    scale_radiogroup.active = 0

        scale_radiogroup = RadioGroup(labels=["Linear", "Logarithmic"], active=0, width=145)
        scale_radiogroup.on_change("active", scale_radiogroup_callback)
        self.scale_radiogroup = scale_radiogroup

        # ---- display max value
        def display_max_spinner_callback(_attr, _old_value, new_value):
            self.display_min_spinner.high = new_value - STEP
            if new_value <= 0:
                scale_radiogroup.active = 0

            lin_colormapper.high = new_value
            log_colormapper.high = new_value

        display_max_spinner = Spinner(
            title="Max Display Value:",
            low=disp_min + STEP,
            value=disp_max,
            step=STEP,
            disabled=bool(auto_switch.active),
            width=145,
        )
        display_max_spinner.on_change("value", display_max_spinner_callback)
        self.display_max_spinner = display_max_spinner

        # ---- display min value
        def display_min_spinner_callback(_attr, _old_value, new_value):
            self.display_max_spinner.low = new_value + STEP
            if new_value <= 0:
                scale_radiogroup.active = 0

            lin_colormapper.low = new_value
            log_colormapper.low = new_value

        display_min_spinner = Spinner(
            title="Min Display Value:",
            high=disp_max - STEP,
            value=disp_min,
            step=STEP,
            disabled=bool(auto_switch.active),
            width=145,
        )
        display_min_spinner.on_change("value", display_min_spinner_callback)
        self.display_min_spinner = display_min_spinner

        # ---- high color
        def high_color_callback(_attr, _old_value, new_value):
            lin_colormapper.high_color = new_value
            log_colormapper.high_color = new_value

        high_color = ColorPicker(title="High Color:", color=cmap_dict[colormap][-1], width=90)
        high_color.on_change("color", high_color_callback)
        self.high_color = high_color

        # ---- mask color
        def mask_color_callback(_attr, _old_value, new_value):
            lin_colormapper.nan_color = new_value
            log_colormapper.nan_color = new_value

        mask_color = ColorPicker(title="Mask Color:", color="gray", width=90)
        mask_color.on_change("color", mask_color_callback)
        self.mask_color = mask_color

    def update(self, image):
        """Trigger an update for the colormapper.

        Args:
            image (ndarray): A source image for colormapper.
        """
        if self.auto_switch.active:
            # This is faster than using bottleneck versions of nanmin and nanmax
            image_min = np.floor(np.nanmin(image))
            image_max = np.ceil(np.nanmax(image))

            self.display_min_spinner.value = image_min
            self.display_max_spinner.value = image_max
