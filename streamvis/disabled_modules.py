import jungfrau_utils as ju
import numpy as np
from bokeh.models import ColumnDataSource, Quad


class DisabledModules:
    def __init__(self, image_views):
        """Initialize a disabled pixels overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
        """
        self._detector_name = ""
        self._ju_handler = None
        self._source = ColumnDataSource(dict(left=[], right=[], top=[], bottom=[]))

        glyph = Quad(
            left="left",
            right="right",
            top="top",
            bottom="bottom",
            line_alpha=0,
            fill_alpha=0,
            hatch_pattern="/",
            hatch_color="white",
        )

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, glyph)

    def update(self, metadata, geometry, gap_pixels, n_rot90):
        """Trigger an update for the disabled modules overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
            geometry (bool): Geometry corrections are applied.
            gap_pixels (bool): Gap pixels are added.
            n_rot90 (int): Number of image 90 deg rotations.
        """
        detector_name = metadata.get("detector_name")
        disabled_modules = metadata.get("disabled_modules")

        if not disabled_modules or not detector_name:
            self._source.data.update(left=[], right=[], top=[], bottom=[])
            return

        if self._detector_name != detector_name:
            try:
                self._ju_handler = ju.JFDataHandler(detector_name)
                self._detector_name = detector_name
            except KeyError:
                self._source.data.update(left=[], right=[], top=[], bottom=[])
                return

        if self._ju_handler.is_stripsel():
            self._source.data.update(left=[], right=[], top=[], bottom=[])
            return

        detector_geometry = self._ju_handler.detector_geometry
        im_shape = self._ju_handler._get_shape_out(gap_pixels=gap_pixels, geometry=geometry)

        if geometry:
            left = np.take(detector_geometry.origin_x, disabled_modules)
            bottom = np.take(detector_geometry.origin_y, disabled_modules)
        else:
            left = np.zeros(len(disabled_modules))
            bottom = np.array(disabled_modules) * (512 + 2 * gap_pixels)

        right = left + 1024 + 6 * gap_pixels
        top = bottom + 512 + 2 * gap_pixels

        # a total number of rotations
        n_rot90 += detector_geometry.rotate90

        if n_rot90 == 1:  # (x, y) -> (y, -x)
            left, right, bottom, top = bottom, top, im_shape[1] - right, im_shape[1] - left
        elif n_rot90 == 2:  # (x, y) -> (-x, -y)
            left, right = im_shape[1] - right, im_shape[1] - left
            bottom, top = im_shape[0] - top, im_shape[0] - bottom
        elif n_rot90 == 3:  # (x, y) -> (-y, x)
            left, right, bottom, top = im_shape[0] - top, im_shape[0] - bottom, left, right

        self._source.data.update(left=left, right=right, top=top, bottom=bottom)
