"""Compute cycle-by-cycle features."""

import warnings
import numpy as np
import pandas as pd

from bycycle.utils.checks import check_param_range
from bycycle.utils.dataframes import drop_samples_df
from bycycle.features.shape import compute_shape_features
from bycycle.features.burst import compute_burst_features
from bycycle.burst import detect_bursts_cycles, detect_bursts_amp

###################################################################################################
###################################################################################################

def compute_features(sig, fs, f_range, center_extrema='peak', burst_method='cycles',
                     burst_kwargs=None, threshold_kwargs=None, find_extrema_kwargs=None,
                     consistency_mode='min', return_samples=True):
    """Compute shape and burst features for each cycle.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    center_extrema : {'peak', 'trough'}
        The center extrema in the cycle.

        - 'peak' : cycles are defined trough-to-trough
        - 'trough' : cycles are defined peak-to-peak

    burst_method : {'cycles', 'amp'}
        Method for detecting bursts.

        - 'cycles': detect bursts based on the consistency of consecutive periods & amplitudes
        - 'amp': detect bursts using an amplitude threshold

    burst_kwargs : dict, optional, default: None
        Additional keyword arguments defined in :func:`~.compute_burst_fraction` for dual
        amplitude threshold burst detection (i.e. when burst_method='amp').
    threshold_kwargs : dict, optional, default: None
        Feature thresholds for cycles to be considered bursts, matching keyword arguments for:

        - :func:`~.detect_bursts_cycles` for consistency burst detection
          (i.e. when burst_method='cycles')
        - :func:`~.detect_bursts_amp` for  amplitude threshold burst detection
          (i.e. when burst_method='amp').

    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (:func:`~.find_extrema`)
        to change filter parameters or boundary. By default, the filter length is set to three
        cycles of the low cutoff frequency (``f_range[0]``).
    consistency_mode : {'min', 'mean', 'max'}
        Adjacent amplitude and period comparison method.
    return_samples : bool, optional, default: True
        Returns samples indices of cyclepoints used for determining features if True.

    Returns
    -------
    df_features : pandas.DataFrame
        A dataframe containing shape and burst features for each cycle. Columns:

        - ``period`` : period of the cycle
        - ``time_decay`` : time between peak and next trough
        - ``time_rise`` : time between peak and previous trough
        - ``time_peak`` : time between rise and decay zero-crosses
        - ``time_trough`` : duration of previous trough estimated by zero-crossings
        - ``volt_decay`` : voltage change between peak and next trough
        - ``volt_rise`` : voltage change between peak and previous trough
        - ``volt_amp`` : average of rise and decay voltage
        - ``volt_peak`` : voltage at the peak
        - ``volt_trough`` : voltage at the last trough
        - ``time_rdsym`` : fraction of cycle in the rise period
        - ``time_ptsym`` : fraction of cycle in the peak period
        - ``band_amp`` : average analytic amplitude of the oscillation

        When consistency burst detection is used (i.e. burst_method='cycles'):

        - ``amp_fraction`` : normalized amplitude
        - ``amp_consistency`` : difference in the rise and decay voltage within a cycle
        - ``period_consistency`` : difference between a cycle’s period and the period of the
          adjacent cycles
        - ``monotonicity`` : fraction of monotonic voltage changes in rise and decay phases
          (positive going in rise and negative going in decay)

        When dual threshold burst detection is used (i.e. burst_method='amp'):

        - ``burst_fraction`` : fraction of a cycle that is bursting

        When cyclepoints are returned (i.e. default, return_samples=True)

        - ``sample_peak`` : sample at which the peak occurs
        - ``sample_zerox_decay`` : sample of the decaying zero-crossing
        - ``sample_zerox_rise`` : sample of the rising zero-crossing
        - ``sample_last_trough`` : sample of the last trough
        - ``sample_next_trough`` : sample of the next trough

    Examples
    --------
    Compute shape and burst features:

    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sig = sim_bursty_oscillation(10, fs, freq=10)
    >>> df_features = compute_features(sig, fs, f_range=(8, 12))
    """

    # Ensure arguments are within valid range
    check_param_range(fs, 'fs', (0, np.inf))

    # Compute shape features for each cycle
    df_shape_features = compute_shape_features(sig, fs, f_range, center_extrema=center_extrema,
                                               find_extrema_kwargs=find_extrema_kwargs)

    # Ensure kwargs are a dictionaries
    if burst_method == 'amp' and not isinstance(burst_kwargs, dict):
        burst_kwargs = {}

    if not isinstance(threshold_kwargs, dict):
        threshold_kwargs = {}
        warnings.warn("""
            No burst detection thresholds are provided. This is not recommended. Please
            inspect your data and choose appropriate parameters for 'threshold_kwargs'.
            Default burst detection parameters are likely not well suited for your
            desired application.
            """)

    # Ensure required kwargs are set for amplitude burst detection
    if burst_method == 'amp':
        burst_kwargs['fs'] = fs
        burst_kwargs['f_range'] = f_range

    # Compute burst features for each cycle
    df_burst_features = compute_burst_features(df_shape_features, sig, burst_method=burst_method,
                                               burst_kwargs=burst_kwargs,
                                               consistency_mode=consistency_mode)

    # Concatenate shape and burst features
    df_features = pd.concat((df_burst_features, df_shape_features), axis=1)

    # Define whether or not each cycle is part of a burst
    if burst_method == 'cycles':
        df_features = detect_bursts_cycles(df_features, **threshold_kwargs)
    elif burst_method == 'amp':
        df_features = detect_bursts_amp(df_features, **threshold_kwargs)
    else:
        raise ValueError('Invalid argument for "burst_method".'
                         'Either "cycles" or "amp" must be specified."')

    df_features = drop_samples_df(df_features) if return_samples is False else df_features

    return df_features
