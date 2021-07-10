"""Test functions for the Bycycle class."""

from inspect import ismethod

from pytest import raises

import pandas as pd
import numpy as np

from bycycle import Bycycle, BycycleGroup
from bycycle.tests.tutils import plot_test

###################################################################################################
###################################################################################################

def test_bycycle():
    """Test initializing a Bycycle object."""

    bm = Bycycle()

    assert bm.center_extrema == 'peak'
    assert bm.burst_method == 'cycles'
    assert isinstance(bm.thresholds, dict)
    assert isinstance(bm.find_extrema_kwargs, dict)
    assert bm.burst_kwargs == {}
    assert bm.return_samples
    assert bm.consistency_mode == 'min'

    defaults = [bm.df_features, bm.sig, bm.fs, bm.f_range]
    assert defaults == [None] * len(defaults)


def test_bycycle_fit(sim_args):
    """Test the fit method of a Bycycle object."""

    bm = Bycycle()

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    with raises(ValueError):
        bm.fit(np.array([sig, sig]), fs, f_range)

    bm.fit(sig, fs, f_range)

    assert isinstance(bm.df_features, pd.DataFrame)
    assert bm.fs == fs
    assert bm.f_range == f_range
    assert (bm.sig == sig).all()


@plot_test
def test_bycycle_plot(sim_args):
    """Test the plot method of a Bycycle object."""

    bm = Bycycle()

    sig = sim_args['sig']
    fs = sim_args['fs']
    f_range = sim_args['f_range']

    with raises(ValueError):
        bm.plot()

    bm.fit(sig, fs, f_range)
    bm.plot()


def test_bycyclegroup():
    """Test initializing a BycycleGroup object."""

    bg = BycycleGroup()

    assert isinstance(bg.dfs_features, list) and len(bg.dfs_features) == 0
    assert bg.center_extrema == 'peak'
    assert bg.burst_method == 'cycles'
    assert isinstance(bg.thresholds, dict)
    assert isinstance(bg.find_extrema_kwargs, dict)
    assert bg.burst_kwargs == {}
    assert bg.return_samples
    assert bg.consistency_mode == 'min'

    defaults = [bg.sigs, bg.fs, bg.f_range]
    assert defaults == [None] * len(defaults)


def test_bycyclegroup_fit(sim_args):
    """Test the fit method of a BycycleGroup object."""

    # 2d case
    sigs = np.array([sim_args['sig']] * 2)
    fs = sim_args['fs']
    f_range = sim_args['f_range']
    thresholds = {
        'amp_fraction_threshold': 0.,
        'amp_consistency_threshold': .5,
        'period_consistency_threshold': .5,
        'monotonicity_threshold': .8,
        'min_n_cycles': 3
    }

    bg = BycycleGroup(thresholds=thresholds)

    with raises(ValueError):
        bg.fit(sigs[0], fs, f_range)

    bg.fit(sigs, fs, f_range)

    assert isinstance(bg.dfs_features, list)
    for bm in bg:
        assert isinstance(bm, Bycycle)
    assert isinstance(bg.dfs_features[0].df_features, pd.DataFrame)
    assert ismethod(bg.dfs_features[0].fit)
    assert ismethod(bg.dfs_features[0].plot)
    assert len(bg) == 2
    assert bg.fs == fs
    assert bg.f_range == f_range
    assert (bg.sigs == sigs).all()

    # 3d case
    sigs = np.array([sigs, sigs])
    bg = BycycleGroup()

    with raises(ValueError):
        bg.fit(sigs[0][0], fs, f_range)

    bg.fit(sigs, fs, f_range)

    assert isinstance(bg.dfs_features, list)
    assert isinstance(bg.dfs_features[0], list)
    assert isinstance(bg.dfs_features[0][0], Bycycle)
    assert isinstance(bg.dfs_features[0][0].df_features, pd.DataFrame)
    assert ismethod(bg.dfs_features[0][0].fit)
    assert ismethod(bg.dfs_features[0][0].plot)
    assert len(bg) == 2
    assert len(bg[0]) == 2
    assert bg.fs == fs
    assert bg.f_range == f_range
    assert (bg.sigs == sigs).all()