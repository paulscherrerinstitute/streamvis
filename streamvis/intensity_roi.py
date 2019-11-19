from bokeh.models import ColumnDataSource, Quad, Text, Toggle


class IntensityROI:
    def __init__(self, image_views):
        """Initialize a intensity ROI overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
        """
        # ---- intensity ROIs
        self._source = ColumnDataSource(
            dict(left=[], right=[], bottom=[], top=[], text_x=[], text_y=[], text=[])
        )
        quad_glyph = Quad(
            left='left',
            right='right',
            bottom='bottom',
            top='top',
            fill_alpha=0,
            line_color='white',
            line_alpha=0,
        )

        text_glyph = Text(
            x='text_x',
            y='text_y',
            text='text',
            text_align='right',
            text_baseline='top',
            text_color='white',
            text_alpha=0,
        )

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, quad_glyph)
            image_view.plot.add_glyph(self._source, text_glyph)

        # ---- toggle button
        def toggle_callback(state):
            if state:
                quad_glyph.line_alpha = 1
                text_glyph.text_alpha = 1
            else:
                quad_glyph.line_alpha = 0
                text_glyph.text_alpha = 0

        toggle = Toggle(label="Intensity ROIs", button_type='default')
        toggle.on_click(toggle_callback)
        self.toggle = toggle

    def update(self, metadata, sv_metadata):
        """Trigger an update for the intensity ROI overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
            sv_metadata (MetadataHandler): Report update issues to that metadata handler.
        """
        roi_x1 = metadata.get('roi_x1')
        roi_x2 = metadata.get('roi_x2')
        roi_y1 = metadata.get('roi_y1')
        roi_y2 = metadata.get('roi_y2')

        if roi_x1 and roi_x2 and roi_y1 and roi_y2:
            if not len(roi_x1) == len(roi_x2) == len(roi_y1) == len(roi_y2):
                sv_metadata.add_issue("Metadata for intensity ROIs is inconsistent")

            else:
                self._source.data.update(
                    left=roi_x1,
                    right=roi_x2,
                    bottom=roi_y1,
                    top=roi_y2,
                    text_x=roi_x2,
                    text_y=roi_y2,
                    text=[str(i) for i in range(len(roi_x1))],
                )

        else:
            self._source.data.update(
                left=[], right=[], bottom=[], top=[], text_x=[], text_y=[], text=[]
            )

            if self.toggle.active:
                sv_metadata.add_issue("Metadata does not contain data for intensity ROIs")
