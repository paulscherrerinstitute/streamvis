import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColorPicker,
    ColumnDataSource,
    DataTable,
    Div,
    IndexFilter,
    LinearColorMapper,
    Spacer,
    TableColumn,
)

import streamvis as sv

doc = curdoc()

sv_rt = sv.Runtime()

MAIN_CANVAS_WIDTH = 1200
MAIN_CANVAS_HEIGHT = 1200

REASONS_MAP = zip(
    (0, 1, "additional pixel mask"),
    (1, 2, "wrong gain (gain=0)"),
    (2, 4, "wrong gain (gain=1)"),
    (3, 8, "wrong gain (gain=2)"),
    (4, 16, "wrong gain (gain=3)"),
    (5, 32, "wrong gain (gain=0, hg0)"),
    (6, 64, "wrong gain (gain=1, hg0)"),
    (7, 128, "wrong gain (gain=2, hg0)"),
    (8, 256, "wrong gain (gain=3, hg0)"),
    (9, 512, "std dev of gain 0"),
    (10, 1024, "std dev of gain 1"),
    (11, 2048, "std dev of gain 2"),
    (12, 4096, "std dev of gain 3"),
    (20, 1048576, "empty space"),
    (21, 2097152, "double pixel"),
    (22, 4194304, "module edge pixel"),
)

BITS = next(REASONS_MAP)
VALUES = next(REASONS_MAP)
REASONS = next(REASONS_MAP)

# Create streamvis components
sv_streamctrl = sv.StreamControl(sv_rt)
del sv_streamctrl.conv_opts.children[1]  # remove 'Mask' checkbox

sv_metadata = sv.MetadataHandler()
sv_metadata.show_all_switch.active = [0]  # activate 'Show All Metadata' checkbox

color_mapper = LinearColorMapper(
    low_color="#e1e0d9", palette=["#2a78d6"], high_color="#eb6834", low=0.5, high=1.5
)

sv_main = sv.ImageView(
    height=MAIN_CANVAS_HEIGHT, width=MAIN_CANVAS_WIDTH, show_pixel_values_on_zoom=False
)
sv_main.image_renderer.glyph.color_mapper = color_mapper
del sv_main.plot.toolbar.tools[-1]  # remove HoverTool


def drop_selection_button_callback():
    mask_reasons_datatable_source.selected.indices = []


drop_selection_button = Button(label="Drop Selection")
drop_selection_button.on_click(drop_selection_button_callback)

selection_hint_div = Div(
    text=(
        "Hold <b>Ctrl</b> to select multiple rows, <b>Shift</b> to select a range, "
        "and press <b>Drop Selection</b> to clear."
    ),
    styles={"color": "gray"},
)


mask_reasons_datatable_source = ColumnDataSource(
    dict(bit=BITS, value=VALUES, reason=REASONS, number=[0] * len(BITS))
)
mask_reasons_datatable = DataTable(
    source=mask_reasons_datatable_source,
    columns=[
        TableColumn(field="bit", title="Pixel Mask Bit"),
        TableColumn(field="reason", title="Pixel Mask Reason"),
        TableColumn(field="number", title="Number of Pixels"),
    ],
    height=430,
    width=650,
    index_position=None,
)


def good_color_callback(_attr, _old_value, new_value):
    color_mapper.low_color = new_value


good_color = ColorPicker(title="Good Pixels:", color=color_mapper.low_color, width=90)
good_color.on_change("color", good_color_callback)


def masked_color_callback(_attr, _old_value, new_value):
    color_mapper.palette = [new_value]


masked_color = ColorPicker(title="Masked Pixels:", color=color_mapper.palette[0], width=90)
masked_color.on_change("color", masked_color_callback)


def selected_color_callback(_attr, _old_value, new_value):
    color_mapper.high_color = new_value


selected_color = ColorPicker(title="Selected Pixels:", color=color_mapper.high_color, width=90)
selected_color.on_change("color", selected_color_callback)


# Final layouts
layout_controls = column(
    row(mask_reasons_datatable),
    row(drop_selection_button, column(Spacer(height=7), selection_hint_div)),
    row(good_color, masked_color, selected_color, Spacer(width=100), sv_streamctrl.conv_opts),
    row(sv_metadata.datatable),
)

final_layout = row(sv_main.plot, Spacer(width=30), layout_controls)

doc.add_root(final_layout)


_empty_pixel_mask_reasons = np.zeros((1, 1), dtype="int64")
_last_pixel_mask_reasons = None
_last_selected_mask = None
_last_highlight_pixel_mask_reasons = None


async def internal_periodic_callback():
    global _last_pixel_mask_reasons, _last_selected_mask, _last_highlight_pixel_mask_reasons

    if not sv_streamctrl.is_receiving:
        return

    metadata, pixel_mask_reasons = sv_streamctrl.get_pixel_mask_reasons()
    if pixel_mask_reasons is None:
        pixel_mask_reasons = _empty_pixel_mask_reasons

    is_new_image = pixel_mask_reasons is not _last_pixel_mask_reasons
    if is_new_image:  # update Number of Pixels in the DataTable
        numbers = [np.count_nonzero(pixel_mask_reasons & (1 << bit)) for bit in BITS]
        mask_reasons_datatable_source.data.update(number=numbers)
        _last_pixel_mask_reasons = pixel_mask_reasons

        # TODO: Force DataTable refresh (remove after bokeh/3.10)
        mask_reasons_datatable.source = ColumnDataSource()
        mask_reasons_datatable.source = mask_reasons_datatable_source

    selected_mask = 0
    for ind in mask_reasons_datatable_source.selected.indices:
        selected_mask |= 1 << BITS[ind]

    if is_new_image or selected_mask != _last_selected_mask:
        highlight_pixel_mask_reasons = (pixel_mask_reasons != 0).astype("uint8")

        if selected_mask:
            highlight_pixel_mask_reasons[(pixel_mask_reasons & selected_mask) != 0] = 2

        _last_selected_mask = selected_mask
        _last_highlight_pixel_mask_reasons = highlight_pixel_mask_reasons
    else:
        highlight_pixel_mask_reasons = _last_highlight_pixel_mask_reasons

    with sv_main.plot.hold(render=True):
        sv_main.image_renderer.view.filter = IndexFilter(indices=[])
        sv_main.update(highlight_pixel_mask_reasons)
        sv_main.image_renderer.view.filter = IndexFilter(indices=[0])

    sv_metadata.update(metadata)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
