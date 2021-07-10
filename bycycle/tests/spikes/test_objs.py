"""Tests for spike objects."""

from copy import deepcopy

import pytest
import numpy as np
import pandas as pd

from neurodsp.tests.tutils import plot_test
from bycycle import Spikes, SpikesMEA

###################################################################################################
###################################################################################################


@pytest.mark.parametrize('find_extrema_kwargs', [None, {'filter_kwargs': {'n_cycles': 3}}])
def test_spikes(find_extrema_kwargs):
    """Test Spike object initialization."""

    spikes = Spikes(find_extrema_kwargs=find_extrema_kwargs)

    assert spikes.sig == None
    assert spikes.fs == None
    assert spikes.f_range == None
    assert spikes.std == None

    assert spikes.df_features == None
    assert spikes.spikes == []
    assert spikes.params == None
    assert spikes.spikes_gen == None


@pytest.mark.parametrize('center_extrema', ['trough', 'peak'])
@pytest.mark.parametrize('n_gaussians', [0, 2, 3])
def test_spikes_fit(sim_spikes, center_extrema, n_gaussians):

    sig = sim_spikes['sig']
    fs = sim_spikes['fs']
    f_range = sim_spikes['f_range']

    spikes = Spikes(center_extrema)

    spikes.fit(sig, fs, f_range, n_gaussians=n_gaussians, tol=1e-3)

    if n_gaussians != 0:

        assert len(spikes.params) == len(spikes.df_features) == len(spikes.r_squared)
        assert spikes.params.shape[1] == (n_gaussians * 4) + 3

        # Magic methods
        assert len(spikes) == len(spikes.params)
        assert isinstance(spikes[0], np.ndarray)
        for spike in spikes:
            assert isinstance(spike, np.ndarray)

    if center_extrema == 'peak':
        assert 'sample_peak' in spikes.df_features.columns
        assert 'sample_last_trough' in spikes.df_features.columns
        assert 'sample_next_trough' in spikes.df_features.columns
    else:
        assert 'sample_trough' in spikes.df_features.columns
        assert 'sample_last_peak' in spikes.df_features.columns
        assert 'sample_next_peak' in spikes.df_features.columns


@pytest.mark.parametrize('inplace', [True, False])
def test_spikes_normalize(sim_spikes_fit, inplace):

    spikes = deepcopy(sim_spikes_fit['spikes'])

    pre_norm = spikes.spikes.copy()

    if inplace:
        spikes.normalize(inplace)
        post_norm = spikes.spikes.copy()
    else:
        post_norm = spikes.normalize(inplace)

    assert (pre_norm != post_norm).any()

    spikes.spikes = None

    with pytest.raises(ValueError):
        spikes.normalize()


def test_spikes_generate_spikes(sim_spikes_fit):

    spikes = deepcopy(sim_spikes_fit['spikes'])

    spikes_gen_pre = spikes.spikes_gen.copy()

    spikes.spikes_gen = None

    spikes.generate_spikes()

    for gen, regen in zip(spikes_gen_pre, spikes.spikes_gen):
        assert (gen == regen).all()

    # Null case
    spikes.params[:, 0] = np.nan

    spikes.generate_spikes()

    for spike in spikes.spikes_gen:
        assert np.isnan(spike)

    # Single gaussian case
    spikes_gen_gauss = spikes.params[:, :-3]

    spikes_gen_gauss = spikes_gen_gauss[:, [0, 3, 6]]

    spikes_gen_sigmoid = spikes.params[:, -3:]

    spikes.params = np.concatenate((spikes_gen_gauss, spikes_gen_sigmoid), axis=1)

    spikes.generate_spikes()

    # Error case
    spikes.df_features = None

    with pytest.raises(ValueError):
        spikes.generate_spikes()


@plot_test
@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
def test_spikes_plot(sim_spikes_fit, stack, normalize):

    spikes = deepcopy(sim_spikes_fit['spikes'])

    if not stack:
        spikes.plot(stack=stack, normalize=normalize, index=0)
    else:
        spikes.plot(stack=stack, normalize=normalize)

    spikes.df_features = None
    with pytest.raises(ValueError):
        spikes.plot()


@plot_test
def test_plot_gaussian_param(sim_spikes_fit):

    spikes = deepcopy(sim_spikes_fit['spikes'])

    spikes.plot_gaussian_params()

    spikes.df_features = None

    with pytest.raises(ValueError):
        spikes.plot_gaussian_params()


@pytest.mark.parametrize('find_extrema', [None, {'filter_kwargs': {'n_cycles': 3}}])
@pytest.mark.parametrize('center_extrema', ['peak', 'trough'])
def test_mea(find_extrema, center_extrema):
    """Test SpikesMEA object initialization."""

    mea = SpikesMEA(center_extrema, find_extrema)

    assert mea.center_extrema == center_extrema
    assert isinstance(mea.find_extrema_kwargs, dict)
    assert mea.find_extrema_kwargs['filter_kwargs']['n_cycles'] == 3
    assert mea.volts is None
    assert mea.components is None


def test_mea_fit(sim_spikes):
    """Test SpikesMEA fitting."""

    sig = sim_spikes['sig']
    sigs = np.vstack((sig, sig))

    f_range = sim_spikes['f_range']
    fs = sim_spikes['fs']

    mea = SpikesMEA()
    mea.fit(sigs, fs, f_range)

    assert (mea.sigs == sigs).all()
    assert (mea.sig == sig).all()

    assert isinstance(mea.df_features, pd.DataFrame)
    assert isinstance(mea.volts, np.ndarray)

    assert mea.volts.shape == (len(mea.df_features), (5 * len(sigs)))


def test_mea_pca(sim_spikes):
    """Test SpikesMEA PCA."""

    sig = sim_spikes['sig']
    sigs = np.vstack((sig, sig))

    f_range = sim_spikes['f_range']
    fs = sim_spikes['fs']

    mea = SpikesMEA()
    mea.fit(sigs, fs, f_range)

    mea.pca(10, 1)

    # Optional sklearn dependency not required
    try:
        import sklearn
        assert isinstance(mea.components, np.ndarray)
        assert mea.components.shape == (len(mea.df_features), 1)
    except ImportError:
        assert mea.components is None
