import h5py
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, Slider, TextInput

import streamvis as sv

doc = curdoc()

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 1000 + 55
MAIN_CANVAS_HEIGHT = 1000 + 59

# Main plot
sv_main = sv.ImageView(plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH)


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_main])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_main.plot.add_layout(sv_colormapper.color_bar, place="below")


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=400, plot_width=700)


# File path text input
file_path = TextInput(title="File Path:", value="/")

# Dataset path text input
dataset_path = TextInput(title="Dataset Path:", value="/")

# Load button
def _load_image_from_dataset(file, dataset, index):
    with h5py.File(file, "r") as f:
        image = f[dataset][index].astype("float32")
        metadata = dict(shape=list(image.shape))

    return image, metadata


def load_file_button_callback():
    sv_rt.image, sv_rt.metadata = _load_image_from_dataset(
        file_path.value, dataset_path.value, image_index_slider.value
    )

    doc.add_next_tick_callback(update_client)
    image_index_slider.disabled = False


load_file_button = Button(label="Load", button_type="default")
load_file_button.on_click(load_file_button_callback)

# Image index slider
def image_index_slider_callback(_attr, _old, new):
    sv_rt.image, sv_rt.metadata = _load_image_from_dataset(file_path.value, dataset_path.value, new)

    doc.add_next_tick_callback(update_client)


image_index_slider = Slider(
    start=0, end=99, value_throttled=0, step=1, title="Pulse Number", disabled=True,
)
image_index_slider.on_change("value_throttled", image_index_slider_callback)


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=300, datatable_width=400)


# Final layouts
sv_colormapper.select.width = 110
sv_colormapper.high_color.width = 90
sv_colormapper.mask_color.width = 90

layout_controls = column(
    file_path,
    dataset_path,
    load_file_button,
    image_index_slider,
    row(sv_colormapper.select, sv_colormapper.high_color),
    sv_colormapper.scale_radiobuttongroup,
    row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
    sv_colormapper.auto_toggle,
)

final_layout = row(layout_controls, sv_main.plot, column(sv_hist.plots[0], sv_metadata.datatable))

doc.add_root(final_layout)


async def update_client():
    image, metadata = sv_rt.image, sv_rt.metadata

    sv_colormapper.update(image)
    sv_main.update(image)

    # Statistics
    im_block = image[sv_main.y_start : sv_main.y_end, sv_main.x_start : sv_main.x_end]
    sv_hist.update([im_block])

    # Update metadata
    sv_metadata.update(metadata)


async def internal_periodic_callback():
    if sv_rt.image.shape == (1, 1):
        # skip client update if the current image is dummy
        return

    doc.add_next_tick_callback(update_client)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
