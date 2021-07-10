"""Tests for features.burst."""

import pytest

import numpy as np

from bycycle.features.burst import *

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("dual_thresh", [True, False])
@pytest.mark.parametrize("center_e", ['peak', 'trough'])
def test_compute_burst_features(sim_args, dual_thresh, center_e):

    sig = sim_args['sig']
    df_shape_features = sim_args['df_shapes']

    if center_e == 'trough':

        # Swap column names (fixture is peak centered)
        rename_dict = {'sample_peak': 'sample_trough',
                       'sample_zerox_decay': 'sample_zerox_rise',
                       'sample_zerox_rise': 'sample_zerox_decay',
                       'sample_last_trough': 'sample_last_peak',
                       'sample_next_trough': 'sample_next_peak'}

        df_shape_features.rename(columns=rename_dict, inplace=True)

    if dual_thresh:

        # Use dual threshold burst detecion
        burst_detection_kwargs = {'fs': sim_args['fs'], 'f_range': sim_args['f_range']}

        df_burst_features = compute_burst_features(df_shape_features, sig, burst_method='amp',
                                                   burst_kwargs=burst_detection_kwargs)

        burst_fraction = df_burst_features['burst_fraction']

        assert np.nan not in burst_fraction
        assert np.all((burst_fraction >= 0) & (burst_fraction <= 1))

        # Excpeted error
        try:
            df_burst_features = compute_burst_features(df_shape_features, sig, burst_method='amp',
                                                       burst_kwargs={})
            assert False
        except ValueError as e:
            pass


    else:

        # Use consistency burst detection
        df_burst_features = compute_burst_features(df_shape_features, sig)

        amp_fraction = df_burst_features['amp_fraction'].values[1:-1]
        amp_consistency = df_burst_features['amp_consistency'].values[1:-1]
        period_consistency = df_burst_features['period_consistency'].values[1:-1]
        monotonicity = df_burst_features['monotonicity'].values[1:-1]

        assert np.all((amp_fraction >= 0) & (amp_fraction <= 1))
        assert np.all((amp_consistency >= 0) & (amp_consistency <= 1))
        assert np.all((period_consistency >= 0) & (period_consistency <= 1))
        assert np.all((monotonicity >= 0) & (monotonicity <= 1))

    assert len(df_shape_features) == len(df_burst_features)

    # Expected error
    try:
        df_burst_features = compute_burst_features(df_shape_features, sig, burst_method=None)
        assert False
    except ValueError as e:
        pass


@pytest.mark.parametrize("direction", ['both', 'next', 'last'])
def test_compute_amp_consistency(sim_args, direction):

    df_shape_features = sim_args['df_shapes']
    df_burst = sim_args['df_burst']
    print(df_burst['amp_consistency'])
    amp_consist_min = compute_amp_consistency(df_shape_features, direction, 'min')
    amp_consist_mean = compute_amp_consistency(df_shape_features, direction, 'mean')
    amp_consist_max = compute_amp_consistency(df_shape_features, direction, 'max')

    assert amp_consist_min[1:-1].mean() <= amp_consist_mean[1:-1].mean() \
        <= amp_consist_max[1:-1].mean()


@pytest.mark.parametrize("direction", ['both', 'next', 'last'])
def test_compute_period_consistency(sim_args, direction):

    df_shape_features = sim_args['df_shapes']

    period_consist_min = compute_period_consistency(df_shape_features, direction, 'min')
    period_consist_mean = compute_period_consistency(df_shape_features, direction, 'mean')
    period_consist_max = compute_period_consistency(df_shape_features, direction, 'max')

    assert period_consist_min[1:-1].mean() <= period_consist_mean[1:-1].mean() \
        <= period_consist_max[1:-1].mean()
