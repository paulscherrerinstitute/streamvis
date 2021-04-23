from bokeh.models import ColumnDataSource, Quad, Text, Toggle


class IntensityROI:
    def __init__(self, image_views, sv_metadata):
        """Initialize a intensity ROI overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            sv_metadata (MetadataHandler): A metadata handler to report metadata issues.
        """
        self._sv_metadata = sv_metadata

        # ---- intensity ROIs
        self._source = ColumnDataSource(
            dict(left=[], right=[], bottom=[], top=[], text_x=[], text_y=[], text=[])
        )
        quad_glyph = Quad(
            left="left",
            right="right",
            bottom="bottom",
            top="top",
            fill_alpha=0,
            line_color="white",
        )

        text_glyph = Text(
            x="text_x",
            y="text_y",
            text="text",
            text_align="right",
            text_baseline="top",
            text_color="white",
        )

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, quad_glyph)
            image_view.plot.add_glyph(self._source, text_glyph)

        # ---- toggle button
        toggle = Toggle(label="Intensity ROIs", button_type="default", default_size=145)
        self.toggle = toggle

    def _clear(self):
        if len(self._source.data["left"]):
            self._source.data.update(
                left=[], right=[], bottom=[], top=[], text_x=[], text_y=[], text=[]
            )

    def update(self, metadata):
        """Trigger an update for the intensity ROI overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
        """
        if not self.toggle.active:
            self._clear()
            return

        roi_x1 = metadata.get("roi_x1")
        roi_x2 = metadata.get("roi_x2")
        roi_y1 = metadata.get("roi_y1")
        roi_y2 = metadata.get("roi_y2")

        if roi_x1 is None or roi_x2 is None or roi_y1 is None or roi_y2 is None:
            self._sv_metadata.add_issue("Metadata does not contain data for intensity ROIs")
            self._clear()
            return

        if not (len(roi_x1) == len(roi_x2) == len(roi_y1) == len(roi_y2)):
            self._sv_metadata.add_issue("Metadata for intensity ROIs is inconsistent")
            self._clear()
            return

        self._source.data.update(
            left=roi_x1,
            right=roi_x2,
            bottom=roi_y1,
            top=roi_y2,
            text_x=roi_x2,
            text_y=roi_y2,
            text=[str(i) for i in range(len(roi_x1))],
        )
