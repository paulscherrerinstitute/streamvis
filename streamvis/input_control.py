from collections import deque

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import CheckboxGroup, CustomJS, Div, RadioGroup, Select, Slider, Toggle

js_backpressure_code = """
if (cb_obj.tags[0]) return;
cb_obj.tags = [true];
"""

DP_LABELS = ["keep", "mask", "interp"]


class StreamControl:
    def __init__(self, sv_rt):
        """Initialize a stream control widget.
        """
        doc = curdoc()
        self.receiver = doc.receiver
        self.stats = doc.stats
        self.jf_adapter = doc.jf_adapter
        self._sv_rt = sv_rt

        # connect toggle button
        def toggle_callback(_active):
            if _active or not self._prev_image_buffer:
                self.prev_image_slider.disabled = True
            else:
                self.prev_image_slider.disabled = False

            self._update_toggle_view()

        toggle = Toggle(label="Connect", button_type="primary", tags=[True], default_size=145)
        toggle.js_on_change("tags", CustomJS(code=js_backpressure_code))
        toggle.on_click(toggle_callback)
        self.toggle = toggle

        # data type select
        datatype_select = Select(
            title="Data type:", value="Image", options=["Image", "Gains"], default_size=145
        )
        self.datatype_select = datatype_select

        # conversion options
        conv_opts_div = Div(text="Conversion options:", margin=(5, 5, 0, 5))
        conv_opts_cbg = CheckboxGroup(
            labels=["Mask", "Gap pixels", "Geometry"], active=[0, 1, 2], default_size=145
        )
        self.conv_opts_cbg = conv_opts_cbg
        self.conv_opts = column(conv_opts_div, conv_opts_cbg)

        # double pixels handling
        double_pixels_div = Div(text="Double pixels:", margin=(5, 5, 0, 5))
        double_pixels_rg = RadioGroup(labels=DP_LABELS, active=0, default_size=145)
        self.double_pixels_rg = double_pixels_rg
        self.double_pixels = column(double_pixels_div, double_pixels_rg)

        # rotate image select
        rotate_values = ["0", "90", "180", "270"]
        rotate_image = Select(
            title="Rotate image (deg):",
            value=rotate_values[0],
            options=rotate_values,
            default_size=145,
        )
        self.rotate_image = rotate_image

        # show only events
        self.show_only_events_toggle = CheckboxGroup(labels=["Show Only Events"], default_size=145)

        # Previous Image slider
        self._prev_image_buffer = deque(maxlen=60)

        def prev_image_slider_callback(_attr, _old, new):
            sv_rt.metadata, sv_rt.image = self._prev_image_buffer[new]
            # TODO: fix this workaround
            sv_rt.aggregated_image = sv_rt.image

        prev_image_slider = Slider(
            start=0, end=59, value_throttled=0, step=1, title="Previous Image", disabled=True,
        )
        prev_image_slider.on_change("value_throttled", prev_image_slider_callback)
        self.prev_image_slider = prev_image_slider

        doc.add_periodic_callback(self._update_toggle_view, 1000)

    @property
    def is_activated(self):
        """Return the stream toggle state (readonly)
        """
        return self.toggle.active

    @property
    def is_receiving(self):
        """Return the stream receiver state (readonly)
        """
        return self.receiver.state == "receiving"

    @property
    def mask_active(self):
        return 0 in list(self.conv_opts_cbg.active)

    @property
    def gap_pixels_active(self):
        return 1 in list(self.conv_opts_cbg.active)

    @property
    def geometry_active(self):
        return 2 in list(self.conv_opts_cbg.active)

    @property
    def double_pixels_active(self):
        return DP_LABELS[self.double_pixels_rg.active]

    @property
    def n_rot90(self):
        return int(self.rotate_image.value) // 90

    @property
    def current_image_shape(self):
        return self._sv_rt.image.shape

    def get_stream_data(self, index):
        """Get data from the stream receiver.

        Args:
            index (int): index into data buffer of receiver

        Returns:
            (dict, ndarray): metadata and image at index
        """
        if not self.toggle.tags[0]:
            return dict(shape=[1, 1]), np.zeros((1, 1), dtype="float32")

        mask = self.mask_active
        gap_pixels = self.gap_pixels_active
        geometry = self.geometry_active
        double_pixels = self.double_pixels_active

        if not gap_pixels and double_pixels == "interp":
            double_pixels = "keep"
            self.double_pixels_rg.active = 0

        if self.show_only_events_toggle.active:
            # Show only events
            metadata, raw_image = self.stats.last_hit
        else:
            # Show image at index
            metadata, raw_image = self.receiver.buffer[index]

        if raw_image.dtype == np.uint16:
            self.conv_opts_cbg.disabled = False
            self.double_pixels_rg.disabled = False
        else:
            self.conv_opts_cbg.disabled = True
            self.double_pixels_rg.disabled = True

        jf_handler = self.jf_adapter.handler
        opts_args = dict(
            mask=mask, gap_pixels=gap_pixels, double_pixels=double_pixels, geometry=geometry
        )

        if self.datatype_select.value == "Image":
            image = self.jf_adapter.process(raw_image, metadata, **opts_args)

            if jf_handler and "saturated_pixels" not in metadata and raw_image.dtype == np.uint16:
                saturated_pixels_y, saturated_pixels_x = jf_handler.get_saturated_pixels(
                    raw_image, **opts_args
                )

                metadata["saturated_pixels_y"] = saturated_pixels_y
                metadata["saturated_pixels_x"] = saturated_pixels_x
                metadata["saturated_pixels"] = len(saturated_pixels_x)

        elif self.datatype_select.value == "Gains":
            image = self.jf_adapter.get_gains(raw_image, metadata, **opts_args)

        n_rot90 = self.n_rot90
        if n_rot90:
            image = np.rot90(image, k=n_rot90)

        image = np.ascontiguousarray(image, dtype=np.float32)

        self.toggle.tags = [False]

        if not self._prev_image_buffer or self._prev_image_buffer[-1][0] is not metadata:
            self._prev_image_buffer.append((metadata, image))

        # Set slider to the right-most position
        if len(self._prev_image_buffer) > 1:
            self.prev_image_slider.end = len(self._prev_image_buffer) - 1
            self.prev_image_slider.value = len(self._prev_image_buffer) - 1

        return metadata, image

    def _update_toggle_view(self):
        """Update label and button type of the toggle
        """
        if self.is_activated:
            if self.is_receiving:
                label = "Receiving"
                button_type = "success"
            else:
                label = "Polling"
                button_type = "warning"
        else:
            label = "Connect"
            button_type = "primary"

        self.toggle.label = label
        self.toggle.button_type = button_type
