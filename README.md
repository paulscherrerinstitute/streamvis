[![GitHub version](https://badge.fury.io/gh/ivan-usov%2Fstreamvis.svg)](https://badge.fury.io/gh/ivan-usov%2Fstreamvis)
[![Deployment](https://github.com/paulscherrerinstitute/streamvis/actions/workflows/deployment.yaml/badge.svg)](https://github.com/paulscherrerinstitute/streamvis/actions/workflows/deployment.yaml)
[![GitHub license](https://img.shields.io/github/license/paulscherrerinstitute/streamvis)](https://github.com/paulscherrerinstitute/streamvis/blob/master/LICENSE)

# Stream visualization server
**Streamvis** project is a webserver and a collection of apps for visualization of data streams. It is based on [bokeh](https://github.com/bokeh/bokeh) and generally works with [zmq](https://github.com/zeromq/libzmq) streams.

An example of user application:
![streamvis_1](https://user-images.githubusercontent.com/13196195/50630977-0275a280-0f43-11e9-8734-17257dd1fb1d.gif)

## Build
The build is triggered upon pushing a tag into a `master` branch and involves running the Github Actions script, which builds a package and uploads it to `paulscherrerinstitute` anaconda channel. A tagged release commit can be created with `make_release.py` script.

To build a local conda package without uploading it to the anaconda channel:
```bash
$ conda build ./conda-recipe
```

A docker image can be built from the latest version on `paulscherrerinstitute` anaconda channel:
```bash
$ docker build ./docker
```

## Install
The default installation procedure:
```bash
$ conda install -c paulscherrerinstitute streamvis
```

If a package was built locally:
```bash
$ conda install --use-local streamvis
```

## Running the server
```bash
$ streamvis <app_name> [--opt-params]
```
Navigate to http://localhost:5006/ in your browser to start using the **streamvis** application.

To get a general help, a list of available applications and optional parameters:
```bash
$ streamvis -h
```

## Message metadata entries
Required metadata entries:
* `type: str` - image data type, e.g. "uint16", or "float32"
* `shape: Iterable[int]` - image shape in pixels

Optional jungfrau-related metadata entries:
* `detector_name: str` - detector name (e.g. "JF01T03V01"), required for adc->keV conversion and geometry
* `gain_file: str` - path to a gain file, required for adc->keV conversion
* `pedestal_file: str` - path to a pedestal file, required for adc->keV conversion and mask
* `daq_rec: int` - a last bit is used to determine whether detector is in a highgain mode
* `module_map: Iterable[int]` - a mapping between data regions and detector module positions (e.g. [0, -1, 1] - a second module is switched off)
* `mask: bool` - superseed a user GUI selection for mask
* `gap_pixels: bool` - superseed a user GUI selection for gap_pixels
* `geometry: bool` - superseed a user GUI selection for geometry
* `double_pixels: str` - superseed a user GUI selection for double_pixels, can be "keep", "mask", or "interp"

Statistics tab:
* `pulse_id: int` - is required for statistics to be collected, data is grouped into runs based on this value
* `is_good_frame: bool` - causes a metadata issue if not True, increments a number in "Bad Frames" column
* `saturated_pixels: int` - causes a metadata issue if not 0, increments a number in "Sat pix frames" column
* `laser_on: bool` - is used to split statistics between laser_on/laser_off columns, those columns are hidden if this key is not present
* `sfx_hit: bool` - increments a number in "Laser ON/OFF frames" columns (those columns are hidden if laser_on is not present)

Hitrate Plot tab:
* `pulse_id: int` - is required for statistics to be collected, determines which point along x-axis is updated
* `number_of_spots: int` - if larger than a value of --hit-threshold cli argument, then sets sfx_hit to True (ignored, if sfx_hit is present)
* `sfx_hit: bool` - determines whether it's a hit (also whether the data may be displayed when "Show Only Hits" button is toggled)

Radial Profile tab:
* `pulse_id: int` - is required for statistics to be collected, the corresponding data is ignored if values is not within "Pulse ID Window" from the most recent pulse_id received
* `radint_q: Iterable[float]` - a vector of q values (x-axis)
* `radint_I: Iterable[float]` - a vector of intensity values (y-axis)
* `laser_on: bool` - determines which graph from "Frames laser on/off" is being updated

ROI intensities tab / Intensity ROIs overlay:
* `roi_x1: Iterable[float]` - a vector of intensity ROI left borders
* `roi_x2: Iterable[float]` - a vector of intensity ROI right borders
* `roi_y1: Iterable[float]` - a vector of intensity ROI bottom borders
* `roi_y2: Iterable[float]` - a vector of intensity ROI top borders
* `roi_intensities_normalised: Iterable[float]` - a vector of corresponding ROI intensities

Saturated Pixels overlay:
* `saturated_pixels_x: Iterable[float]` - x-coordinates of saturated pixels (attempt to derive if not present in case of raw data (uint16))
* `saturated_pixels_y: Iterable[float]` - y-coordinates of saturated pixels (attempt to derive if not present in case of raw data (uint16))
* `saturated_pixels: int` - a number of saturated pixels (attempt to derive if not present in case of raw data (uint16))

Spots overlay:
* `spot_x: Iterable[float]` - x-coordinates of spots
* `spot_y: Iterable[float]` - y-coordinates of spots
* `number_of_spots: int` - should be equal to a length of both, spot_x and spot_y

Resolution Rings overlay:
* `detector_distance: float` - distance to detector in meters
* `beam_energy: float` - beam energy in eV
* `beam_center_x: float` - beam x-coordinate
* `beam_center_y: float` - beam y-coordinate

Disabled Modules overlay (requires a valid `detector_name`):
* `disabled_modules: Iterable[int]` - indexes of modules to display as disabled

Trajectory Plot:
* `number_of_spots: int` - a value increases a color saturation of trajectory glyphs and is shown on mouse hover
* `swissmx_x: float` - trajectory x-coordinate
* `swissmx_y: float` - trajectory y-coordinate
* `frame: int` - a value is shown on mouse hover over a trajectory glyph

Aggregated images:
* `aggregated_images: int` - in case of aggregation, treat a received image as a sum of that number of images
