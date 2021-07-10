"""Spike plotting functions."""

import numpy as np

from neurodsp.plts import plot_time_series, plot_bursts
from neurodsp.plts.utils import check_ax

from bycycle.utils.dataframes import get_extrema_df
from bycycle.utils.timeseries import limit_signal
from bycycle.spikes.utils import split_signal

###################################################################################################
###################################################################################################

def plot_spikes(df_features, sig, fs, spikes=None, index=None, xlim=None, ax=None):
    """Plot a group of spikes or the cyclepoints for an individual spike.

    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
    sig : 1d or 2d array
        Voltage timeseries. May be 2d if spikes are split.
    fs : float
        Sampling rate, in Hz.
    spikes : 1d array, optional, default: None
        Spikes that have been split into a 2d array. Ignored if ``index`` is passed.
    index : int, optional, default: None
        The index in ``df_features`` to plot. If None, plot all spikes.
    xlim : tuple
        Upper and lower time limits. Ignored if spikes or index is passed.
    ax : matplotlib.Axes, optional, default: None
        Figure axes upon which to plot.
    """

    ax = check_ax(ax, (10, 4))

    center_e, _ = get_extrema_df(df_features)

    # Plot a single spike
    if index is not None:

        times = np.arange(0, len(sig)/fs, 1/fs)

        # Get where spike starts/ends
        start = df_features.iloc[index]['sample_start'].astype(int)
        end = df_features.iloc[index]['sample_end'].astype(int)

        sig_lim = sig[start:end+1]
        times_lim = times[start:end+1]

        # Plot the spike waveform
        plot_time_series(times_lim, sig_lim, ax=ax)

        # Plot cyclespoints
        labels, keys = _infer_labels(center_e)
        colors = ['C0', 'C1', 'C2', 'C3']

        for idx, key in enumerate(keys):

            sample = df_features.iloc[index][key].astype('int')

            plot_time_series(np.array([times[sample]]), np.array([sig[sample]]),
                             colors=colors[idx], labels=labels[idx], ls='', marker='o', ax=ax)

    # Plot as stack of spikes
    elif index is None and spikes is not None:

        times = np.arange(0, len(spikes[0])/fs, 1/fs)

        plot_time_series(times, spikes, ax=ax)

    # Plot as continuous timeseries
    elif index is None and spikes is None:

        ax = check_ax(ax, (15, 3))

        times = np.arange(0, len(sig)/fs, 1/fs)

        plot_time_series(times, sig, ax=ax, xlim=xlim)

        if xlim is None:
            sig_lim = sig
            df_lim = df_features
            times_lim = times
            starts = df_lim['sample_start']
        else:
            cyc_idxs = (df_features['sample_start'].values >= xlim[0] * fs) & \
                    (df_features['sample_end'].values <= xlim[1] * fs)

            df_lim = df_features.iloc[cyc_idxs].copy()

            sig_lim, times_lim = limit_signal(times, sig, start=xlim[0], stop=xlim[1])

            starts = df_lim['sample_start'] - int(fs * xlim[0])

        ends = starts + (df_lim['sample_end'].values - df_lim['sample_start'].values)

        is_spike = np.zeros(len(sig_lim), dtype='bool')

        for start, end in zip(starts, ends):
            is_spike[start:end] = True

        plot_bursts(times_lim, sig_lim, is_spike, ax=ax)


def _infer_labels(center_e):
    """Create labels based on center extrema."""

    # Infer labels
    if center_e == 'trough':
        labels = ['Trough', 'Peak', 'Inflection']
        keys = ['sample_trough', 'sample_next_peak', 'sample_end']
    elif center_e == 'peak':
        labels = ['Peak', 'Trough', 'Inflection']
        keys = ['sample_peak', 'sample_next_trough', 'sample_end']

    return labels, keys
