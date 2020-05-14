[![GitHub version](https://badge.fury.io/gh/ivan-usov%2Fstreamvis.svg)](https://badge.fury.io/gh/ivan-usov%2Fstreamvis)
[![Build Status](https://travis-ci.com/ivan-usov/streamvis.svg?branch=master)](https://travis-ci.com/ivan-usov/streamvis)
[![GitHub license](https://img.shields.io/github/license/ivan-usov/streamvis.svg)](https://github.com/ivan-usov/streamvis/blob/master/LICENSE)

# Stream visualization server
**Streamvis** project is a webserver and a collection of apps for visualization of data streams. It is based on [bokeh](https://github.com/bokeh/bokeh) and generally works with [zmq](https://github.com/zeromq/libzmq) streams.

An example of user application:
![streamvis_1](https://user-images.githubusercontent.com/13196195/50630977-0275a280-0f43-11e9-8734-17257dd1fb1d.gif)

## Build
The build is triggered upon pushing a tag into a `master` branch and involves running the [Travis CI](https://travis-ci.com/ivan-usov/streamvis) script, which builds a package and uploads it to `paulscherrerinstitute` anaconda channel. A tagged release commit can be created with `make_release.py` script.

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
