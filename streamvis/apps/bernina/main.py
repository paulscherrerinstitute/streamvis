from datetime import datetime
from functools import partial

import h5py
import jungfrau_utils as ju
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import BasicTicker, BasicTickFormatter, Button, \
    ColumnDataSource, DataRange1d, DatetimeAxis, Grid, Line, LinearAxis, Panel, \
    PanTool, Plot, ResetTool, Spacer, Tabs, Title, Toggle, WheelZoomTool
from PIL import Image as PIL_Image
from tornado import gen

import receiver
import streamvis as sv

doc = curdoc()
doc.title = receiver.args.page_title

# Expected image sizes for the detector
IMAGE_SIZE_X = 1030
IMAGE_SIZE_Y = 1554

# initial image size to organize placeholders for actual data
image_size_x = IMAGE_SIZE_X
image_size_y = IMAGE_SIZE_Y

current_gain_file = ''
current_pedestal_file = ''
jf_calib = None

sv_rt = sv.Runtime()

connected = False

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = image_size_x//2 + 55 + 40
MAIN_CANVAS_HEIGHT = image_size_y//2 + 86 + 60

ZOOM_CANVAS_WIDTH = 388 + 55
ZOOM_CANVAS_HEIGHT = 388 + 62

DEBUG_INTENSITY_WIDTH = 700

APP_FPS = 1
STREAM_ROLLOVER = 36000

util_plot_size = 160

ZOOM_INIT_WIDTH = 500
ZOOM_INIT_HEIGHT = 500
ZOOM1_INIT_X = 265
ZOOM1_INIT_Y = 800
ZOOM2_INIT_X = 265
ZOOM2_INIT_Y = 200

# Custom tick formatter for displaying large numbers
tick_formatter = BasicTickFormatter(precision=1)

# Create colormapper
sv_colormapper = sv.ColorMapper()


# Main plot
sv_mainplot = sv.ImagePlot(
    sv_colormapper,
    plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH,
    image_height=image_size_y, image_width=image_size_x,
)
sv_mainplot.plot.title = Title(text=' ')

# ---- add colorbar
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.height = 10
sv_colormapper.color_bar.location = (0, -5)
sv_mainplot.plot.add_layout(sv_colormapper.color_bar, place='below')

# ---- add zoom plot 1
sv_zoomplot1 = sv.ImagePlot(
    sv_colormapper,
    plot_height=ZOOM_CANVAS_HEIGHT, plot_width=ZOOM_CANVAS_WIDTH,
    image_height=image_size_y, image_width=image_size_x,
)
sv_zoomplot1.plot.title = Title(text='Signal roi', text_color='red')

sv_mainplot.add_as_zoom(
    sv_zoomplot1, line_color='red',
    init_x=ZOOM1_INIT_X, init_width=ZOOM_INIT_WIDTH,
    init_y=ZOOM1_INIT_Y, init_height=ZOOM_INIT_HEIGHT,
)

# ---- add zoom plot 2
sv_zoomplot2 = sv.ImagePlot(
    sv_colormapper,
    plot_height=ZOOM_CANVAS_HEIGHT, plot_width=ZOOM_CANVAS_WIDTH,
    image_height=image_size_y, image_width=image_size_x,
)
sv_zoomplot2.plot.title = Title(text='Background roi', text_color='green')

sv_mainplot.add_as_zoom(
    sv_zoomplot2, line_color='green',
    init_x=ZOOM2_INIT_X, init_width=ZOOM_INIT_WIDTH,
    init_y=ZOOM2_INIT_Y, init_height=ZOOM_INIT_HEIGHT,
)


# Total intensity plot
total_intensity_plot = Plot(
    title=Title(text="Total Image Intensity"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=util_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

# ---- tools
total_intensity_plot.add_tools(PanTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
total_intensity_plot.add_layout(
    LinearAxis(axis_label="Total intensity", formatter=tick_formatter), place='left')
total_intensity_plot.add_layout(DatetimeAxis(), place='below')

# ---- grid lines
total_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
total_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
total_sum_source = ColumnDataSource(dict(x=[], y=[]))
total_intensity_plot.add_glyph(total_sum_source, Line(x='x', y='y'))


# Zoom1 intensity plot
zoom1_intensity_plot = Plot(
    title=Title(text="Normalized signal−background Intensity"),
    x_range=total_intensity_plot.x_range,
    y_range=DataRange1d(),
    plot_height=util_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

# ---- tools
zoom1_intensity_plot.add_tools(PanTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
zoom1_intensity_plot.add_layout(LinearAxis(
    axis_label="Intensity", formatter=tick_formatter), place='left')
zoom1_intensity_plot.add_layout(DatetimeAxis(), place='below')

# ---- grid lines
zoom1_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom1_sum_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_intensity_plot.add_glyph(zoom1_sum_source, Line(x='x', y='y', line_color='red'))


# Histogram plots
sv_hist = sv.Histogram(nplots=3, plot_height=300, plot_width=600)
sv_hist.plots[0].title = Title(text="Full image")
sv_hist.plots[1].title = Title(text="Signal roi", text_color='red')
sv_hist.plots[2].title = Title(text="Background roi", text_color='green')
sv_hist.auto_toggle.width = 300

# Intensity stream reset button
def intensity_stream_reset_button_callback():
    stream_t = datetime.now()  # keep the latest point in order to prevent full axis reset
    total_sum_source.data.update(x=[stream_t], y=[total_sum_source.data['y'][-1]])
    zoom1_sum_source.data.update(x=[stream_t], y=[zoom1_sum_source.data['y'][-1]])

intensity_stream_reset_button = Button(label="Reset", button_type='default')
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)


# Stream panel
# ---- connect toggle button
def stream_button_callback(state):
    global connected
    if state:
        connected = True
        stream_button.label = 'Connecting'
        stream_button.button_type = 'default'

    else:
        connected = False
        stream_button.label = 'Connect'
        stream_button.button_type = 'default'


stream_button = Toggle(label="Connect", button_type='default')
stream_button.on_click(stream_button_callback)

# assemble
tab_stream = Panel(child=column(stream_button), title="Stream")
data_source_tabs = Tabs(tabs=[tab_stream])


# Colormapper panel
colormap_panel = column(
    sv_colormapper.select,
    Spacer(height=10),
    sv_colormapper.scale_radiobuttongroup,
    Spacer(height=10),
    sv_colormapper.auto_toggle,
    sv_colormapper.display_max_spinner,
    sv_colormapper.display_min_spinner,
)


# Metadata datatable
sv_metadata = sv.MetadataHandler(
    datatable_height=130, datatable_width=700, check_shape=(IMAGE_SIZE_Y, IMAGE_SIZE_X),
)


# Final layouts
layout_main = column(Spacer(), sv_mainplot.plot)

layout_zoom = column(sv_zoomplot1.plot, sv_zoomplot2.plot, Spacer())

hist_layout = row(sv_hist.plots[0], sv_hist.plots[1], sv_hist.plots[2])

hist_controls = row(
    Spacer(width=20), column(Spacer(height=19), sv_hist.auto_toggle),
    sv_hist.lower_spinner, sv_hist.upper_spinner, sv_hist.nbins_spinner,
    column(Spacer(height=19), sv_hist.log10counts_toggle))

layout_utility = column(
    gridplot([total_intensity_plot, zoom1_intensity_plot],
             ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)),
    row(Spacer(width=400), intensity_stream_reset_button))

layout_controls = row(Spacer(width=45), colormap_panel, Spacer(width=45), data_source_tabs)

layout_metadata = column(
    sv_metadata.datatable,
    row(sv_metadata.show_all_toggle, sv_metadata.issues_dropdown),
)

final_layout = column(
    row(layout_main, Spacer(width=15), column(layout_zoom), Spacer(width=15),
        column(layout_metadata, layout_utility, layout_controls)),
    column(hist_layout, hist_controls))

doc.add_root(final_layout)


@gen.coroutine
def update_client(image, metadata):
    sv_colormapper.update(image)

    pil_im = PIL_Image.fromarray(image)
    sv_mainplot.update(pil_im)

    # Signal roi and intensity
    sig_y_start = int(np.floor(sv_zoomplot1.y_start))
    sig_y_end = int(np.ceil(sv_zoomplot1.y_end))
    sig_x_start = int(np.floor(sv_zoomplot1.x_start))
    sig_x_end = int(np.ceil(sv_zoomplot1.x_end))

    im_block1 = image[sig_y_start:sig_y_end, sig_x_start:sig_x_end]
    sig_sum = np.sum(im_block1, dtype=np.float)
    sig_area = (sig_y_end - sig_y_start) * (sig_x_end - sig_x_start)

    # Background roi and intensity
    bkg_y_start = int(np.floor(sv_zoomplot2.y_start))
    bkg_y_end = int(np.ceil(sv_zoomplot2.y_end))
    bkg_x_start = int(np.floor(sv_zoomplot2.x_start))
    bkg_x_end = int(np.ceil(sv_zoomplot2.x_end))

    im_block2 = image[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end]
    bkg_sum = np.sum(im_block2, dtype=np.float)
    bkg_area = (bkg_y_end - bkg_y_start) * (bkg_x_end - bkg_x_start)

    # Update histogram
    sv_hist.update([image, im_block1, im_block2])

    # correct the backgroud roi sum by subtracting overlap area sum
    overlap_y_start = max(sig_y_start, bkg_y_start)
    overlap_y_end = min(sig_y_end, bkg_y_end)
    overlap_x_start = max(sig_x_start, bkg_x_start)
    overlap_x_end = min(sig_x_end, bkg_x_end)
    if (overlap_y_end - overlap_y_start > 0) and (overlap_x_end - overlap_x_start > 0):
        # else no overlap
        bkg_sum -= np.sum(image[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end],
                          dtype=np.float)
        bkg_area -= (overlap_y_end - overlap_y_start) * (overlap_x_end - overlap_x_start)

    if bkg_area == 0:
        # background area is fully surrounded by signal area
        bkg_int = 0
    else:
        bkg_int = bkg_sum / bkg_area

    # Corrected signal intensity
    sig_sum -= bkg_int * sig_area

    stream_t = datetime.now()
    total_sum_source.stream(
        new_data=dict(x=[stream_t], y=[np.sum(image, dtype=np.float)]), rollover=STREAM_ROLLOVER)
    zoom1_sum_source.stream(new_data=dict(x=[stream_t], y=[sig_sum]), rollover=STREAM_ROLLOVER)

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)
    sv_metadata.update(metadata_toshow)


@gen.coroutine
def internal_periodic_callback():
    global current_gain_file, current_pedestal_file, jf_calib
    if sv_mainplot.plot.inner_width is None:
        # wait for the initialization to finish, thus skip this periodic callback
        return

    if connected:
        if receiver.state == 'polling':
            stream_button.label = 'Polling'
            stream_button.button_type = 'warning'

        elif receiver.state == 'receiving':
            stream_button.label = 'Receiving'
            stream_button.button_type = 'success'

            sv_rt.current_metadata, sv_rt.current_image = receiver.data_buffer[-1]

            if sv_rt.current_image.dtype != np.float16 and sv_rt.current_image.dtype != np.float32:
                gain_file = sv_rt.current_metadata.get('gain_file')
                pedestal_file = sv_rt.current_metadata.get('pedestal_file')
                detector_name = sv_rt.current_metadata.get('detector_name')
                is_correction_data_present = gain_file and pedestal_file and detector_name

                if is_correction_data_present:
                    if current_gain_file != gain_file or current_pedestal_file != pedestal_file:
                        # Update gain/pedestal filenames and JungfrauCalibration
                        current_gain_file = gain_file
                        current_pedestal_file = pedestal_file

                        with h5py.File(current_gain_file, 'r') as h5gain:
                            gain = h5gain['/gains'][:]

                        with h5py.File(current_pedestal_file, 'r') as h5pedestal:
                            pedestal = h5pedestal['/gains'][:]
                            pixel_mask = h5pedestal['/pixel_mask'][:].astype(np.int32)

                        jf_calib = ju.JungfrauCalibration(gain, pedestal, pixel_mask)

                    sv_rt.current_image = jf_calib.apply_gain_pede(sv_rt.current_image)
                    sv_rt.current_image = ju.apply_geometry(sv_rt.current_image, detector_name)
            else:
                sv_rt.current_image = sv_rt.current_image.astype('float32', copy=True)

    if sv_rt.current_image.shape != (1, 1):
        doc.add_next_tick_callback(partial(
            update_client, image=sv_rt.current_image, metadata=sv_rt.current_metadata))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
