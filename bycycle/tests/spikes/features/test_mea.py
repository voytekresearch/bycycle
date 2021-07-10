"""Tests for MEA features."""

import pytest
import numpy as np

from bycycle.spikes.features.mea import compute_pca_features, compute_voltage_features

###################################################################################################
###################################################################################################


def test_compute_pca_features(sim_spikes, sim_spikes_df):

    sig = sim_spikes['sig']
    sigs = np.vstack((sig, sig))
    df_samples = sim_spikes_df['df_samples']

    components = compute_pca_features(df_samples, sigs, 10, 1)

    # Optional sklearn dependency not required
    try:
        import sklearn
        assert isinstance(components, np.ndarray)
        assert components.shape == (len(df_samples), 1)
    except ImportError:
        assert components is None


def test_compute_voltage_features(sim_spikes, sim_spikes_df):

    sig = sim_spikes['sig']
    sigs = np.vstack((sig, sig))
    df_samples = sim_spikes_df['df_samples']

    volts = compute_voltage_features(df_samples, sigs)

    assert isinstance(volts, np.ndarray)
    assert volts.shape == (len(df_samples), (5 * len(sigs)))
