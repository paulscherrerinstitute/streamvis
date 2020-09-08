import logging

import numpy as np
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, ImageRGBA, Toggle

placeholder = np.zeros((1, 1), dtype=np.uint32)

logger = logging.getLogger(__name__)


class Mask:
    def __init__(self, image_views):
        """Initialize a mask overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
        """
        doc = curdoc()
        self.receiver = doc.receiver

        self.current_file = ""
        self.current_module_map = None

        # ---- rgba image glyph
        self._source = ColumnDataSource(dict(image=[placeholder], x=[0], y=[0], dw=[1], dh=[1]))

        rgba_glyph = ImageRGBA(image="image", x="x", y="y", dw="dw", dh="dh", global_alpha=0)

        for image_view in image_views:
            image_renderer = image_view.plot.add_glyph(self._source, rgba_glyph)
            image_renderer.view.source = ColumnDataSource()

        # ---- toggle button
        def toggle_callback(state):
            if state:
                rgba_glyph.global_alpha = 1
            else:
                rgba_glyph.global_alpha = 0

        toggle = Toggle(label="Mask", button_type="default", default_size=145)
        toggle.on_click(toggle_callback)
        self.toggle = toggle

    def update(self, sv_metadata):
        """Trigger an update for the mask overlay.

        Args:
            sv_metadata (MetadataHandler): Report update issues to that metadata handler.
        """
        handler = self.receiver.jf_adapter.handler
        if handler and handler.pedestal_file:
            if self.current_file != handler.pedestal_file or np.any(
                self.current_module_map != handler.module_map
            ):
                mask_data = handler.get_pixel_mask(gap_pixels=True, geometry=True)
                dh, dw = mask_data.shape

                mask = np.zeros((dh, dw), dtype=np.uint32)
                mask_view = mask.view(dtype=np.uint8).reshape((dh, dw, 4))
                mask_view[:, :, 1] = 255
                mask_view[:, :, 3] = 255 * mask_data

                self.current_file = handler.pedestal_file
                self.current_module_map = handler.module_map
                self._source.data.update(image=[mask], dh=[dh], dw=[dw])

        else:
            self.current_file = ""
            self.current_module_map = None
            self._source.data.update(image=[placeholder])

        if self.toggle.active and self.current_file == "":
            sv_metadata.add_issue("No pedestal file has been provided")
