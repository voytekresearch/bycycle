"""Compute spike cyclespoints functions."""

import numpy as np

from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.spikes.utils import create_cyclepoints_df

###################################################################################################
###################################################################################################

def compute_spike_cyclepoints(sig, fs, f_range, std=2):
    """Find spike cyclepoints.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    std : float or int, optional, default: 2
        The standard deviation threshold used to identify spikes.

    Returns
    -------
    df_samples : pd.DataFrame
        Contains cycle points locations, in samples:

        - sample_start : spike start location.
        - sample_decay : spike zero-crossing decay location.
        - sample_trough : spike minima location.
        - sample_rise : spike zero-crossing rise location.
        - sample_next_peak : spike maxima location.
        - sample_end : spike end location.

    """

    # Find troughs
    _peaks, _troughs = find_extrema(sig, fs, f_range, first_extrema='trough', pass_type='bandpass')

    # Threshold troughs
    thresh = np.mean(sig) - np.std(sig) * std

    volt_troughs = sig[_troughs]

    idxs = np.where(volt_troughs < thresh)[0]

    if len(idxs) == 0:
        raise ValueError('No spikes found outside of std. Try reducing std.')

    idxs = idxs[1:] if idxs[0] == 0 else idxs
    idxs = idxs[:-1] if idxs[-1] == len(sig) else idxs

    # Index cyclepoints
    troughs = _troughs[idxs]
    volt_troughs = sig[troughs]

    last_peaks = _peaks[idxs-1]
    next_peaks = _peaks[idxs]

    last_troughs = _troughs[idxs-1]
    next_troughs = _troughs[idxs+1]

    # Determine spike starting points
    starts = np.zeros_like(troughs)
    ends = np.zeros_like(troughs)

    std_sig = np.std(sig)

    volt_troughs = sig[troughs]
    volt_last_peaks = sig[last_peaks]
    volt_last_troughs = sig[last_troughs]
    volt_next_troughs = sig[next_troughs]
    volt_next_peaks = sig[next_peaks]
    volts = zip(volt_last_troughs, volt_last_peaks, volt_troughs,
                volt_next_peaks, volt_next_troughs)

    for idx, (last_trough, last_peak, trough, next_peak, next_trough) in enumerate(volts):

        # Criteria to defining start/end inflections:
        #   a) Height from side extrema to inflections > 0.5 stdev
        #   b) Inflection voltage must be closer to side peak than trough

        pre_norm_diff = abs(last_peak - last_trough) / std_sig
        post_norm_diff = abs(next_trough - next_peak) / std_sig

        pre_criteria_b = (last_trough - trough) > (last_peak - last_trough)
        post_criteria_b = (next_trough - trough) > (next_peak - next_trough)

        starts[idx] = last_troughs[idx] if pre_norm_diff > 0.5 and pre_criteria_b \
            else last_peaks[idx]
        ends[idx] =  next_troughs[idx] if post_norm_diff > 0.5 and post_criteria_b \
            else next_peaks[idx]

    decays, rises = _compute_spike_zerox(-sig, last_peaks, troughs, next_peaks)

    # Oraganize points into a dataframe
    df_samples = create_cyclepoints_df(sig, starts, decays, troughs, rises,
                                       last_peaks, next_peaks, ends)

    # Drop overlapping spikes, favoring larger voltage
    drop_spikes = np.ones(len(df_samples), dtype=bool)

    for idx in range(len(df_samples)-1):

        if not drop_spikes[idx]:
            continue

        curr_end = ends[idx]
        curr_volt_trough = volt_troughs[idx]

        next_start = starts[idx+1]
        next_volt_trough = volt_troughs[idx+1]

        if curr_end > next_start and curr_volt_trough > next_volt_trough:
            drop_spikes[idx] = False
        elif curr_end > next_start and curr_volt_trough < next_volt_trough:
            drop_spikes[idx+1] = False

    df_samples = df_samples[drop_spikes]
    df_samples.reset_index(inplace=True, drop=True)

    return df_samples


def _compute_spike_zerox(sig, last_peaks, troughs, next_peaks):
    """Find spike zero-crossings."""

    decays = np.round((troughs - last_peaks) / 2).astype(int) + last_peaks
    rises = np.round((next_peaks - troughs) / 2).astype(int) + troughs

    return decays, rises
