"""Compute multi-electrode array features."""

import warnings
import numpy as np

###################################################################################################
###################################################################################################


def compute_pca_features(df_samples, sigs, pad, n_components, norm_mean=False, norm_std=False):
    """Compute principal component scores for each spike.

    Parameters
    ----------
    df_samples : pandas.DataFrame
        Contains cyclepoint locations for each spike.
    sigs : 2d array
        Voltage time series.
    pad : int
        Number of samples to include around trough (one-sided).
    n_components : int or float
        The number of components to include (int) or the number of components required to reach
        a minimum variance explained (float).
    norm_mean : bool, optional, default: False
        Normalize the mean of each MEA of each electrode.
    norm_std : bool, optional, default: False
        Normalize the standard deviation of each electrode.

    Returns
    -------
    components : 2d array
        Principal component scores for each spike location.
    """

    # Trough indices
    troughs = df_samples['sample_trough'].values

    # Epoch spikes
    starts = troughs - pad
    ends = troughs + pad
    sigs_split = np.array([sigs[:, s:e] for s, e in zip(starts, ends)])
    sigs_split = sigs_split.reshape(sigs_split.shape[0], -1)

    # PCA
    try:

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        ys = StandardScaler(with_mean=norm_mean, with_std=norm_std).fit_transform(sigs_split)
        components = PCA(n_components=n_components).fit_transform(ys)

        return components

    except ImportError:

        warnings.warn('Optional dependency, sklearn, is required for PCA and is not installed.')

        return None


def compute_voltage_features(df_samples, sigs):
    """Compute cyclepoints voltages across electrodes.

    Parameters
    ----------
    df_samples : pandas.DataFrame
        Contains cyclepoint locations for each spike.
    sigs : 2d array
        Voltage time series.

    Returns
    -------
    volts : 2d array
        Voltages for each spike at the mean cycle's start, decay, trough, rise and end points.
    """

    # Cyclepoints indices
    starts = df_samples['sample_start'].values
    decays = df_samples['sample_decay'].values
    troughs = df_samples['sample_trough'].values
    rises = df_samples['sample_rise'].values
    ends = df_samples['sample_end'].values

    # Index volts
    volt_starts = sigs[:, starts]
    volt_decays = sigs[:, decays]
    volt_troughs = sigs[:, troughs]
    volt_rises = sigs[:, rises]
    volt_ends = sigs[:, ends]

    # Combine
    volts = (volt_starts, volt_decays, volt_troughs, volt_rises, volt_ends)
    volts = np.vstack(volts).T

    return volts

