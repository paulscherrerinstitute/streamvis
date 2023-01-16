import numpy as np
import pytest

import streamvis as sv


@pytest.fixture(name="sv_hist_single_plot", scope="function")
def _sv_hist_single_plot():
    sv_hist = sv.Histogram(nplots=1)
    yield sv_hist


@pytest.mark.parametrize("nplots", [1, 3, 10])
def test_nplots(nplots):
    sv_hist = sv.Histogram(nplots=nplots)

    assert len(sv_hist.plots) == nplots
    assert len(sv_hist._plot_sources) == nplots


@pytest.mark.parametrize("plot_height", [10, 200, 1000])
@pytest.mark.parametrize("plot_width", [20, 350, 800])
def test_plot_sizes(plot_height, plot_width):
    sv_hist = sv.Histogram(nplots=1, plot_height=plot_height, plot_width=plot_width)
    plot = sv_hist.plots[0]

    assert plot.plot_height == plot_height
    assert plot.plot_width == plot_width


@pytest.mark.parametrize("value", [0, -1, 1, 350, 800, -1000])
def test_update_single_value(sv_hist_single_plot, value):
    data = [np.array(value)]
    default_nbins = sv_hist_single_plot.nbins
    sv_hist_single_plot.update(data)

    assert len(sv_hist_single_plot._plot_sources[0].data["left"]) == default_nbins
    assert len(sv_hist_single_plot._plot_sources[0].data["right"]) == default_nbins
    assert len(sv_hist_single_plot._plot_sources[0].data["top"]) == default_nbins


def test_update_array(sv_hist_single_plot):
    data = [np.random.randint(-100, 100, 1000)]
    default_nbins = sv_hist_single_plot.nbins
    sv_hist_single_plot.update(data)

    assert len(sv_hist_single_plot._plot_sources[0].data["left"]) == default_nbins
    assert len(sv_hist_single_plot._plot_sources[0].data["right"]) == default_nbins
    assert len(sv_hist_single_plot._plot_sources[0].data["top"]) == default_nbins


@pytest.mark.parametrize("nbins", [1, 300, 1000])
def test_update_fixed(nbins):
    sv_hist = sv.Histogram(nplots=1, nbins=nbins)
    sv_hist.auto_toggle.active = [0]  # True
    data = [np.random.randint(-100, 100, 1000)]
    sv_hist.update(data)

    assert len(sv_hist._plot_sources[0].data["left"]) == nbins
    assert len(sv_hist._plot_sources[0].data["right"]) == nbins
    assert len(sv_hist._plot_sources[0].data["top"]) == nbins
