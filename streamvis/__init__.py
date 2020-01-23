"""
Streamvis project is a webserver and a collection of apps for visualization of data streams.
It is based on bokeh and generally works with zmq streams.
"""

from .colormapper import ColorMapper
from .histogram import Histogram
from .image_view import ImageView
from .mask import Mask
from .metadata import MetadataHandler
from .resolution_rings import ResolutionRings
from .runtime import Runtime
from .stream_graph import StreamGraph
from .intensity_roi import IntensityROI
from .projection import Projection
from .input_control import StreamControl
from .image_processor import ImageProcessor
from .saturated_pixels import SaturatedPixels
from .spots import Spots

__version__ = "0.8.0"
