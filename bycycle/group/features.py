"""Functions to compute features across 2 dimensional arrays of data."""

import warnings
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from bycycle.features import compute_features
from bycycle.burst import detect_bursts_cycles, detect_bursts_amp
from bycycle.group.utils import progress_bar, check_kwargs_shape
from bycycle.utils.dataframes import epoch_df

###################################################################################################
###################################################################################################

def compute_features_2d(sigs, fs, f_range, compute_features_kwargs=None, axis=0,
                        return_samples=True, n_jobs=-1, progress=None):
    """Compute shape and burst features for a 2 dimensional array of signals.

    Parameters
    ----------
    sigs : 2d array
        Voltage time series, i.e. (n_channels, n_samples) or (n_epochs, n_samples).
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest, in Hz.
    compute_features_kwargs : dict or list of dict
        Keyword arguments used in :func:`~.compute_features`.
    axis : {0, None}
        Which axes to calculate features across:

        - ``axis=0`` : Iterates over each row/signal in an array independently (i.e. for each
          channel in (n_channels, n_timepoints)).
        - ``axis=None`` : Flattens rows/signals prior to computing features (i.e. across flatten
          epochs in (n_epochs, n_timepoints)).

    return_samples : bool, optional, default: True
        Whether to return a dataframe of cyclepoint sample indices.
    n_jobs : int, optional, default: -1
        The number of jobs to compute features in parallel.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.

    Returns
    -------
    dfs_features : list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
        Each dataframe is computed using the :func:`~.compute_features` function.

    Notes
    -----

    - The order of ``dfs_features`` corresponds to the order of ``sigs``. This list of dataframes
      may be reorganized into a single dataframe using :func:`~.flatten_dfs`.
    - When ``axis=None`` parallel computation may not be performed due to the requirement of
      flattening the array into one dimension.
    - If ``compute_features_kwargs`` is a dictionary, the same kwargs are applied applied across
      the first axis of ``sigs``. Otherwise, a list of dictionaries equal in length to the
      first axis of ``sigs`` is required to apply unique kwargs to each signal.
    - ``return_samples`` is controlled from the kwargs passed in this function. If
      ``return_samples`` is a key in ``compute_features_kwargs``, it's value will be ignored.

    Examples
    --------
    Compute the features of a 2d array (n_epochs=10, n_samples=5000) containing epoched data:

    >>> import numpy as np
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sigs = np.array([sim_bursty_oscillation(10, fs, 10) for i in range(10)])
    >>> compute_kwargs = {'burst_method': 'amp', 'threshold_kwargs':{'burst_fraction_threshold': 1}}
    >>> dfs_features = compute_features_2d(sigs, fs, f_range=(8, 12), axis=None,
    ...                                   compute_features_kwargs=compute_kwargs)

    Compute the features of a 2d array in parallel using the same compute_features kwargs. Note each
    signal features are computed separately in this case, recommended for (n_channels, n_samples):

    >>> compute_kwargs = {'burst_method': 'amp', 'threshold_kwargs':{'burst_fraction_threshold': 1}}
    >>> dfs_features = compute_features_2d(sigs, fs, f_range=(8, 12), n_jobs=2, axis=0,
    ...                                   compute_features_kwargs=compute_kwargs)

    Compute the features of a 2d array in parallel using using individualized settings per signal to
    examine the effect of various amplitude consistency thresholds:

    >>> sigs =  np.array([sim_bursty_oscillation(10, fs, freq=10)] * 10)
    >>> compute_kwargs = [{'threshold_kwargs': {'amp_consistency_threshold': thresh*.1}}
    ...                   for thresh in range(1, 11)]
    >>> dfs_features = compute_features_2d(sigs, fs, f_range=(8, 12), return_samples=False,
    ...                                   n_jobs=2, compute_features_kwargs=compute_kwargs, axis=0)
    """

    # Check compute_features_kwargs
    kwargs = deepcopy(compute_features_kwargs)
    kwargs = np.array(kwargs) if isinstance(kwargs, list) else kwargs

    check_kwargs_shape(sigs, kwargs, axis)

    kwargs = {} if kwargs is None else kwargs
    kwargs = [kwargs] if isinstance(kwargs, dict) else list(kwargs)

    # Drop return_samples argument, as it is set directly in the function call
    for kwarg in kwargs:
        kwarg.pop('return_samples', None)

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    if axis == 0:
        # Compute each signal independently and in paralllel
        with Pool(processes=n_jobs) as pool:

            if len(kwargs) > 1:
                # Map iterable sigs and kwargs together
                mapping = pool.imap(partial(_proxy_2d, fs=fs, f_range=f_range,
                                            return_samples=return_samples),
                                    zip(sigs, kwargs))

            else:
                # Only map sigs, kwargs are the same for each mapping
                mapping = pool.imap(partial(compute_features, fs=fs, f_range=f_range,
                                            return_samples=return_samples,
                                            **kwargs[0]),
                                    sigs)

            dfs_features = list(progress_bar(mapping, progress, len(sigs)))

    elif axis is None:
        # Compute features after flattening the 2d array (i.e. calculated across a 1d signal)
        sig_flat = sigs.flatten()

        center_extrema = kwargs[0].pop('center_extrema', 'peak')

        df_flat = compute_features(sig_flat, fs=fs, f_range=f_range, return_samples=True,
                                   center_extrema=center_extrema, **kwargs[0])

        dfs_features = epoch_df(df_flat, len(sig_flat), len(sigs[0]))

        if not return_samples:

            drop_columns = [col for col in dfs_features[0] if col.startswith('sample_')]

            for idx in range(len(dfs_features)):

                dfs_features[idx] = dfs_features[idx].drop(drop_columns, axis=1)

        # Apply different thresholds if specified
        if len(kwargs) > 0:

            for idx, compute_kwargs in enumerate(kwargs):

                burst_method = compute_kwargs.pop('burst_method', 'cycles')
                thresholds = compute_kwargs.pop('threshold_kwargs', {})
                center_extrema_next = compute_kwargs.pop('center_extrema', None)

                if idx > 0 and center_extrema_next is not None \
                    and center_extrema_next != center_extrema:

                    warnings.warn('''
                        The same center extrema must be used when axis is None and
                        compute_features_kwargs is a list. Proceeding using the first
                        center_extrema: {extrema}.'''.format(extrema=center_extrema))

                if burst_method == 'cycles':
                    dfs_features[idx] = detect_bursts_cycles(dfs_features[idx], **thresholds)

                elif burst_method == 'amp':
                    dfs_features[idx] = detect_bursts_amp(dfs_features[idx], **thresholds)

    else:
        raise ValueError("The axis kwarg must be either 0 or None.")

    return dfs_features


def compute_features_3d(sigs, fs, f_range, compute_features_kwargs=None, axis=0,
                        return_samples=True, n_jobs=-1, progress=None):
    """Compute shape and burst features for a 3 dimensional array of signals.

    Parameters
    ----------
    sigs : 3d array
        Voltage time series, with 3d shape, i.e. (n_channels, n_epochs, n_samples).
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest, in Hz.
    compute_features_kwargs : dict or 1d list of dict or 2d list of dict
        Keyword arguments used in :func:`~.compute_features`.
    axis : {0, 1, (0, 1)}
        Which axes to calculate features across:

        - ``axis=0`` : Iterates over 2D slices along the zeroth dimension, (i.e. for each channel in
          (n_channels, n_epochs, n_timepoints)).
        - ``axis=1`` : Iterates over 2D slices along the first dimension (i.e. across flatten epochs
          in (n_epochs, n_channels, n_timepoints)).
        - ``axis=(0, 1)`` : Iterates over 1D slices along the zeroth and first dimensions (i.e across
          each signal independently in (n_participants, n_channels, n_timepoints)).

    return_samples : bool, optional, default: True
        Whether to return a dataframe of cyclepoint sample indices.
    n_jobs : int, optional, default: -1
        The number of jobs to compute features in parallel.
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm' if installed.

    Returns
    -------
    dfs_features : list of pandas.DataFrame
        Dataframes containing shape and burst features for each cycle.
        Each dataframe is computed using the :func:`~.compute_features` function.

    Notes
    -----

    - The order of ``dfs_features`` corresponds to the order of ``sigs``. This list of dataframes
      may be reorganized into a single dataframe using :func:`~.flatten_dfs`.
    - If ``compute_features_kwargs`` is a dictionary, the same kwargs are applied applied across
      all signals. A 1d list, equal in length to the first dimensions of sigs, may be applied to
      each set of signals along the first dimensions. A 2d list, the same shape as the first two
      dimensions of ``sigs`` may also be used to applied unique parameters to each signal.
    - ``return_samples`` is controlled from the kwargs passed in this function. The
      ``return_samples`` value in ``compute_features_kwargs`` will be ignored.

    Examples
    --------
    Compute the features of a 3d array, in parallel, with a shape of
    (n_channels=2, n_epochs=3, n_signals=5000) using the same compute_features kwargs:

    >>> import numpy as np
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> fs = 500
    >>> sigs = np.array([[sim_bursty_oscillation(10, fs, freq=10) for epoch in range(3)]
    ...                 for ch in range(2)])
    >>> threshold_kwargs = {'amp_consistency_threshold': .5, 'period_consistency_threshold': .5,
    ...                     'monotonicity_threshold': .8, 'min_n_cycles': 3}
    >>> compute_feature_kwargs = {'threshold_kwargs': threshold_kwargs, 'center_extrema': 'trough'}
    >>> features = compute_features_3d(sigs, fs, f_range= (8, 12),
    ...                                compute_features_kwargs=compute_feature_kwargs, axis=0,
    ...                                n_jobs=2)

    Compute the features of a 3d array, in parallel, with a shape of
    (n_channels=2, n_epochs=3, n_signals=5000) using channel-specific compute_features kwargs:

    >>> threshold_kwargs_ch1 = {'amp_consistency_threshold': .25, 'monotonicity_threshold': .25,
    ...                         'period_consistency_threshold': .25, 'min_n_cycles': 3}
    >>> threshold_kwargs_ch2 = {'amp_consistency_threshold': .5, 'monotonicity_threshold': .5,
    ...                         'period_consistency_threshold': .5, 'min_n_cycles': 3}
    >>> compute_kwargs = [{'threshold_kwargs': threshold_kwargs_ch1, 'center_extrema': 'trough'},
    ...                   {'threshold_kwargs': threshold_kwargs_ch2, 'center_extrema': 'trough'}]
    >>> features = compute_features_3d(sigs, fs, f_range= (8, 12),
    ...                                compute_features_kwargs=compute_kwargs, axis=0, n_jobs=2)
    """

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    # Convert list of kwargs to array to check dimensions
    kwargs = deepcopy(compute_features_kwargs)
    kwargs = np.array(kwargs) if isinstance(kwargs, list) else kwargs

    check_kwargs_shape(sigs, kwargs, axis)
    kwargs = list(kwargs.flatten()) if isinstance(kwargs, np.ndarray) else [kwargs]

    if axis in [0, 1]:
        # Independently across 2d slices along either the zeroth or first axis
        sigs = np.swapaxes(sigs, 0, 1) if axis == 1 else sigs
        kwargs = kwargs * np.shape(sigs)[0] if len(kwargs) == 1 else kwargs

        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(partial(_proxy_3d, fs=fs, f_range=f_range,
                                        return_samples=return_samples),
                                zip(sigs, kwargs))

            dfs_features = list(progress_bar(mapping, progress, len(sigs)))

        # Swap the first two axes to return original shape
        dfs_features = [list(dfs) for dfs in zip(*dfs_features)] if axis == 1 else dfs_features

    elif axis == (0, 1):
        # Independently across the first two axes (i.e. for each signal)
        sigs_2d = sigs.reshape(np.shape(sigs)[0]*np.shape(sigs)[1], np.shape(sigs)[2])
        kwargs = kwargs[0] if len(kwargs) == 1 else kwargs

        df_2d = compute_features_2d(sigs_2d, fs, f_range, compute_features_kwargs=kwargs,
                                    return_samples=return_samples, n_jobs=n_jobs,
                                    progress=progress, axis=0)

    else:

        raise ValueError("The axis kwarg must be either 0, 1, or (0, 1).")

    if axis == (0, 1):

        dfs_features = np.zeros((np.shape(sigs)[0], np.shape(sigs)[1])).tolist()

        # Reshape
        for dim0_idx in range(np.shape(sigs)[0]):
            for dim1_idx in range(np.shape(sigs)[1]):
                dfs_features[dim0_idx][dim1_idx] = df_2d[dim0_idx + dim1_idx]

    return dfs_features


def _proxy_2d(args, fs=None, f_range=None, return_samples=None):
    """Proxy function to map kwargs and 2d sigs together."""

    sig, kwargs = args[0], args[1:]

    return compute_features(sig, fs=fs, f_range=f_range,
                            return_samples=return_samples, **kwargs[0])

def _proxy_3d(args, fs=None, f_range=None, return_samples=None):
    """Proxy function to map kwargs and 3d sigs together."""

    sigs, kwargs = args[0], args[1]

    return compute_features_2d(sigs, fs, f_range, compute_features_kwargs=kwargs, axis=None,
                               return_samples=return_samples)
