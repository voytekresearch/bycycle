"""Compute gaussian features."""

import warnings
from functools import partial
from itertools import repeat

from multiprocessing import Pool, cpu_count
from bycycle.group.utils import progress_bar

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats as st
from bycycle.cyclepoints import find_extrema, find_zerox

###################################################################################################
###################################################################################################


def compute_gaussian_features(df_samples, sig, fs, maxfev=2000, tol=1.49e-6, n_jobs=-1, chunksize=1,
                              progress=None, z_thresh_K=0.5, z_thresh_cond=0.5):
    """Compute gaussian features.

    Parameters
    ----------
    df_samples : pandas.DataFrame
        Contains cyclepoint locations for each spike.
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    maxfev : int, optional, default: 2000
        The maximum number of calls in curve_fit.
    tol : float, optional, default: 10e-6
        Relative error desired.
    n_jobs : int, optional, default: -1
        The number of jobs to compute features in parallel.
    chunksize : int, optional, default: 1
        Number of chunks to split spikes into. Each chunk is submitted as a separate job.
        With a large number of spikes, using a larger chunksize will drastically speed up
        runtime. An optimal chunksize is typically np.ceil(n_spikes/n_jobs).
    progress : {None, 'tqdm', 'tqdm.notebook'}
        Specify whether to display a progress bar. Uses 'tqdm', if installed.
    z_thresh_K : float, optional, default: 0.5
        Potassium (k) current z-score threshold.
    z_thresh_cond : float, optional, default: 0.5
        Conductive current z-score threshold.

    Returns
    -------
    params : dict
        Fit parameter values.
    """

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    indices = [*range(len(df_samples))]

    # Compute features in parallel
    with Pool(processes=n_jobs) as pool:

        mapping = pool.imap(partial(_compute_gaussian_features_cycle, df_samples=df_samples,
                                    sig=sig, fs=fs, maxfev=maxfev, tol=tol,
                                    z_thresh_K=0.5, z_thresh_cond=0.5),
                            indices, chunksize=chunksize)

        params = list(progress_bar(mapping, progress, len(df_samples)))

    return np.array(params)


def _compute_gaussian_features_cycle(index, df_samples=None, sig=None, fs=None,
                                     f_ranges=(300, 2000), maxfev=None, tol=None,
                                     z_thresh_K=0.5, z_thresh_cond=0.5):
    """Compute gaussian features for one cycle."""

    start = df_samples.iloc[index]['sample_start'].astype(int)
    end = df_samples.iloc[index]['sample_end'].astype(int)
    sample_trough = df_samples.iloc[index]['sample_trough'].astype(int)

    # Adjust samples to start at zero
    sample_trough -= start

    # Get signal and time
    sig_cyc = sig[start:end+1]
    cyc_len = len(sig_cyc)
    times_cyc = np.arange(0, cyc_len/fs, 1/fs)

    # Fit single skewed gaussian to Na current
    Na_params, Na_gaus = _single_gaus_fit(index, sample_trough, sig_cyc, cyc_len, times_cyc, fs,
                                          extrema_type="trough", maxfev=None, tol=None)

    if not np.isnan(np.sum(Na_gaus)):

        # Get Na center and std
        Na_center = int(Na_params[3]*Na_params[0])
        Na_std = int(Na_params[4]*Na_params[0])

        # Determine Na current region
        upper_std = Na_center + Na_std
        lower_std = Na_center - Na_std

        # Calculate Na current r-squared
        Na_rsq = calculate_r_squared(sig_cyc[lower_std:upper_std], Na_gaus[lower_std:upper_std])

        # Check if Na r-squared is above threshold
        if Na_rsq < 0.5:
            Na_rsq = np.nan
            Na_params = np.append(Na_params, Na_rsq)

            K_params = np.array([np.nan] * len(Na_params))
            cond_params = np.array([np.nan] * len(Na_params))
            print("Failed fits for index = " + str(index))

        else:

            Na_params = np.append(Na_params, Na_rsq)

            # Substract Na current gaussian fit
            rem_sig = sig_cyc - Na_gaus

            # Split remaining signal into left of Na current (K current)
            #   and right (conductive current)
            rem_sig_K, times_K, z_score_K, rem_sig_cond, times_cond, z_score_cond = \
                calculate_k_cond_regions(Na_center, rem_sig, times_cyc,
                                         fs, z_thresh_K, z_thresh_cond)

            # Evaluate remaining signal in K current region
            #   if there is signal over the noise cutoff, fit a second gaussian
            if any(i >= z_thresh_K for i in z_score_K):
                # Get peak of remaining signal
                K_peak = get_current_peak(rem_sig_K, fs, f_ranges, z_thresh_K, z_score_K)

                if K_peak == None:
                    K_params = np.array([np.nan] * len(Na_params))
                    K_gaus = np.array([np.nan] * len(times_K))

                else:
                    # Fit single skewed gaussian to K current
                    K_params, K_gaus = _single_gaus_fit(index, K_peak, rem_sig_K, len(rem_sig_K),
                                                        times_K, fs, extrema_type="peak",
                                                        maxfev=None, tol=None)

                    # Calculate r-squared
                    K_rsq = calculate_r_squared(rem_sig_K, K_gaus)
                    K_params = np.append(K_params, K_rsq)

            else:
                K_params = np.array([np.nan] * len(Na_params))
                K_gaus =  np.array([np.nan] * len(times_K))

            # Evaluate remaining signal in conductive current region
            #   if there is signal over the noise cutoff, fit a second gaussian
            if any(i >= z_thresh_cond for i in z_score_cond):
                # Get peak of remaining signal
                cond_peak = get_current_peak(rem_sig_cond, fs, f_ranges, z_thresh_cond, z_score_cond)

                if cond_peak == None:
                    cond_params = np.array([np.nan] * len(Na_params))
                    cond_gaus = np.array([np.nan] * len(times_cond))

                else:
                    # Fit single skewed gaussian to K current
                    cond_params, cond_gaus = _single_gaus_fit(index, cond_peak, rem_sig_cond,
                                                              len(rem_sig_cond), times_cond, fs,
                                                              extrema_type="peak",  maxfev=None,
                                                              tol=None)

                    # Calculate r-squared
                    cond_rsq = calculate_r_squared(rem_sig_cond, cond_gaus)
                    cond_params = np.append(cond_params, cond_rsq)

            else:
                cond_params = np.array([np.nan] * len(Na_params))
                cond_gaus =  np.array([np.nan] * len(times_cond))

    else:
        Na_rsq = np.nan
        Na_params = np.append(Na_params, Na_rsq)

        K_params = np.array([np.nan] * len(Na_params))
        cond_params = np.array([np.nan] * len(Na_params))
        print("Failed fits for index = " + str(index))

    # Get only center, std, height and alpha parameters
    cond_params_gaus = cond_params[3:]
    Na_params_gaus = Na_params[3:]
    K_params_gaus = K_params[3:]

    all_params = [*cond_params_gaus, *Na_params_gaus, *K_params_gaus]

    return all_params


def estimate_params(extrema, sig_cyc, fs, extrema_type="trough", n_decimals=2):
    """Initial gaussian parameter estimates.

    Parameters
    ----------
    df_samples : pandas.DataFrame
        Contains cycle points locations for each spike.
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    index : int
        The spike index in the 2d array (i.e. the spikes attribute of a Spikes class.
    n_decimals : int, optional, default: 2
        Number of decimals to round parameters to.

    Returns
    -------
    params : 1d array
        Estimated centers, stds, alphas, heights, and sigmoid max, growth, midpoint.
        The number of centers, stds, alphas, heights varies from one to three.
    bounds : list of list
        Lower and upper bounds for each parameters.
    """

    cyc_len = len(sig_cyc)

    centers = []
    stds = []
    heights = []

    # Define parameters
    if extrema_type == "trough":
        height0 =  sig_cyc[extrema] - np.mean(sig_cyc)
    else:
        height0 =  sig_cyc[extrema]

    center0 = extrema / cyc_len

    std0 = estimate_std(sig_cyc, extrema_type=extrema_type, plot=False)

    centers.append(center0.round(n_decimals))
    stds.append(std0.round(n_decimals))
    heights.append(height0.round(n_decimals))

    # Assume no skew
    alphas = [0] * len(centers)
    params = [*centers, *stds, *alphas, *heights]

    return np.array(params)


def _estimate_bounds(sig_cyc, centers, stds, heights):
    """Estimate parameter's lower and upper bounds."""

    # Define bounds
    lower_heights = [height * .5 if height > 0 else height * 1.5 for height in heights]
    upper_heights = [height * 1.5 if height > 0 else height * .5 for height in heights]

    lower_stds = [std * .5 for std in stds]
    upper_stds = [std * 1.5 for std in stds]

    lower_alphas = [-3 for std in stds]
    upper_alphas = [3 for std in stds]

    lower_centers = [center * .5 for center in centers]
    upper_centers = [center * 1.5 for center in centers]

    upper_max = np.max(sig_cyc) - np.min((sig_cyc[0], sig_cyc[-1]))

    bounds = [
        [*lower_centers, *lower_stds, *lower_alphas, *lower_heights, 0, -1, 0],
        [*upper_centers, *upper_stds, *upper_alphas, *upper_heights, upper_max, 1, 1]
    ]

    return bounds


def _fit_gaussians(xs, ys, guess, tol, maxfev, index, bounds=None):
    """Fit gaussians with scipy's curve_fit."""

    try:
        # Fit gaussians
        warnings.filterwarnings("ignore")
        params, _ = curve_fit(sim_gaussian_cycle, xs, ys,
                              p0=guess)

    except:
        # Raise warning for failed fits
        warn_str = "Failed fit for index {idx}.".format(idx=index)
        warnings.warn(warn_str, RuntimeWarning)
        params = np.array([np.nan] * len(guess))

    return params


###################################################################################################
###################################################################################################


def sim_gaussian_cycle(times, *params):
    """Proxy function for compatibility between _sim_ap_cycle and curve_fit.

    Parameters
    ----------
    times : 1d array
        Time definition of the cycle.
    params : floats
        Variable number of centers, stds, alphas, and heights arguments, respectively. The number
        of these variable parameters determines the number of gaussians simulated. An additional
        three trailing arguments to define a sigmoid baseline as maximum, growth, midpoint.

    Returns
    -------
    sig_cycle : 1d array
        Simulated action potential.
    """

    gaussian_params = params
    sing_gaus = _sim_skewed_gaussian(*params)

    return sing_gaus


def _single_gaus_fit(index, extrema, sig_cyc, cyc_len, times_cyc,
                     fs, extrema_type="trough", maxfev=None, tol=None):
    """Calculate guassian fits for single current  """

    # Get cyc time/freq parameters
    _cyc_params = [cyc_len, cyc_len/fs, fs]

    # Initial parameter estimation
    _gaus_params = estimate_params(extrema, sig_cyc, fs, extrema_type=extrema_type, n_decimals=2)

    # Initial bound estimation for Na current
    _bounds = _estimate_bounds(sig_cyc, *_gaus_params.reshape(4, -1)[[0, 1, 3]])

    # Append cycle params to Gaussian params
    _params = np.insert(_gaus_params, 0, _cyc_params , axis=0)

    # Fit single skewed gaussian
    _params_fit = _fit_gaussians(times_cyc, sig_cyc, _params, tol, maxfev, index, bounds=None)

    if np.isnan(np.sum(_params_fit)):
        _gaus = np.array([np.nan] * len(times_cyc))

    else:
        _gaus = _sim_skewed_gaussian(*_params_fit)

    return [_params_fit, _gaus]


def calculate_k_cond_regions(Na_center, rem_sig, times_cyc, fs, z_thresh_K, z_thresh_cond):
    """Calculate K current and conductive current regions
       of the signal based on the center of the Na current.
    """

    rem_sig_K = rem_sig[Na_center:,]
    rem_sig_cond = rem_sig[:Na_center,]

    times_K = times_cyc[Na_center:,]
    times_cond = times_cyc[:Na_center,]

    # Calculate z scores
    z_score_K = st.zscore(rem_sig_K)
    z_score_cond = st.zscore(rem_sig_cond)

    return [rem_sig_K, times_K, z_score_K, rem_sig_cond, times_cond, z_score_cond]


def _sim_gaussian(samples, n_seconds, fs, std, center=.5, height=1.):
    """Simulate a gaussian cycle."""

    xs = np.linspace(0, 1, int(samples))
    cycle = np.exp(-(xs-center)**2 / (2*std**2))

    return cycle


def _sim_skewed_gaussian(samples, n_seconds, fs, center, std, alpha, height=1):
    """Simulate a skewed gaussian cycle."""

    from scipy.stats import norm

    #n_samples = int(np.ceil(n_seconds * fs))
    n_samples = int(samples)

    # Gaussian distribution
    cycle = _sim_gaussian(n_samples, n_seconds, fs, std, center, height)

    # Skewed cumulative distribution function.
    #   Assumes time are centered around 0. Adjust to center around 0.5.
    times = np.linspace(-1, 1, n_samples)
    cdf = norm.cdf(alpha * ((times - ((center * 2) - 1)) / std))

    # Skew the gaussian
    cycle = cycle * cdf

    # Rescale height
    cycle = (cycle / np.max(cycle)) * height

    return cycle

###################################################################################################
###################################################################################################

def estimate_std(spike, extrema_type='trough', plot=False):

    spike = -spike if extrema_type == 'peak' else spike

    height, height_idx = np.min(spike), np.argmin(spike)
    half_height = height / 2

    right = spike[height_idx:]
    left = np.flip(spike[:height_idx+1])

    if plot:
        plt.plot(-spike if extrema_type=='peak' else spike)
        plt.axvline(height_idx, color='r')

    right_idx = _get_closest(right, spike, half_height)
    left_idx = _get_closest(left, spike, half_height)

    if right_idx == None:
        right_idx = left_idx

    if left_idx == None:
        left_idx = right_idx

    fwhm = (right_idx + left_idx + 1)

    std = fwhm / (2 * len(spike) * np.sqrt(2 * np.log(2)))

    return std


def _get_closest(flank, spike, half_height):

    for idx, volt in enumerate(flank):

        if volt > half_height:

            # Get closest sample left or right of half max location
            closest = np.argmin([volt - half_height,
                                 half_height - flank[idx-1]])

            idx = [idx, idx-1][closest]

            return idx


def get_current_peak(sig, fs, f_ranges, z_thresh, z_score):

    peaks, troughs = find_extrema(sig, fs, f_ranges, first_extrema=None, pass_type='bandpass')

    if len(peaks) != 0:
        if len(peaks) > 1:
            # Select highest peak
            max_volt = max( (v, i) for i, v in enumerate(sig[peaks]) )[1]
            peak = peaks[max_volt]
        else:
            peak = peaks[0]

        # Check if peak is over z score threshold
        if z_score[peak] > z_thresh:

            return peak

    return None

def calculate_r_squared(sig_cyc, sig_cyc_est):

    residuals = sig_cyc - sig_cyc_est
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((sig_cyc - np.mean(sig_cyc))**2)

    r_squared = 1 - (ss_res / ss_tot)

    return r_squared
