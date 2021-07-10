"""Class objects to compute features for spiking data."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from neurodsp.plts.utils import savefig

from bycycle import Bycycle

from bycycle.spikes.features import compute_shape_features, compute_gaussian_features
from bycycle.spikes.features.gaussians import sim_action_potential
from bycycle.spikes.features.mea import compute_voltage_features, compute_pca_features

from bycycle.spikes.cyclepoints import compute_spike_cyclepoints
from bycycle.spikes.plts import plot_spikes
from bycycle.spikes.utils import split_signal, rename_df

###################################################################################################
###################################################################################################


class Spikes:
    """Compute spikes features.

    Attributes
    ----------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
    spikes : 2d array
        The signal associated with each spike (row in the ``df_features``).
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    std : float or int, optional, default: 1.5
        The standard deviation used to identify spikes.
    spikes_gen : 2d list
        Spikes generated from fit parameters.
    center_extrema : {'peak', 'trough'}
        The center extrema in the cycle.

        - 'peak' : cycles are defined trough-to-trough
        - 'trough' : cycles are defined peak-to-peak

    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (:func:`~.find_extrema`)
        to change filter parameters or boundary. By default, the filter length is set to three
        cycles of the low cutoff frequency (``f_range[0]``).

    """

    def __init__(self, center_extrema='trough', find_extrema_kwargs=None):
        """Initialize object."""

        # Settings
        self.center_extrema = center_extrema

        if find_extrema_kwargs is None:
            self.find_extrema_kwargs = {'filter_kwargs': {'n_cycles': 3}}
        else:
            self.find_extrema_kwargs = find_extrema_kwargs

        # Fit params
        self.sig = None
        self.fs = None
        self.f_range = None
        self.std = None

        # Results
        self.df_features = None
        self.spikes = []
        self.params = None
        self.spikes_gen = None


    def __len__(self):
        """Define the length of the object."""

        return len(self.spikes)


    def __iter__(self):
        """Allow for iterating across the object."""

        for spike in self.spikes:
            yield spike[~np.isnan(spike)]


    def __getitem__(self, index):
        """Allow for indexing into the object."""

        return self._spikes[index]


    def fit(self, sig, fs, f_range, std=2, n_gaussians=0,
            maxfev=2000, tol=1.49e-6, n_jobs=-1, chunksize=1, progress=None):
        """Compute features for each spike.

        Parameters
        ----------
        sig : 1d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        f_range : tuple of (float, float)
            Frequency range for narrowband signal of interest (Hz).
        std : float or int, optional, default: 2
            Standard deviation used to identify spikes.
        n_gaussians : {0, 2, 3}
            Fit a n number of gaussians to each spike. If zero, no gaussian fitting occurs.
        maxfev : int, optional, default: 2000
            Maximum number of calls in curve_fit.
            Only used when n_gaussians is {1, 2}.
        tol : float, optional, default: 1.49e-6
            Relative error desired.
            Only used when n_gaussians is {1, 2}.
        n_jobs : int, optional, default: -1
            Number of jobs to compute features in parallel.
            Only used when n_gaussians is {1, 2}.
        chunksize : int, optional, default: 1
            Number of chunks to split spikes into. Each chunk is submitted as a separate job.
            With a large number of spikes, using a larger chunk size will drastically speed up
            runtime. An optimal chunksize is typically np.ceil(n_spikes/n_jobs).
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
            Only used when n_gaussians is {1, 2}.
        """

        # Set attibutes
        self.sig = sig
        self.fs = fs
        self.f_range = f_range
        self.std = std

        # Cyclepoints
        if self.center_extrema == 'trough':
            df_features = compute_spike_cyclepoints(self.sig, self.fs, self.f_range, self.std)
        else:
            df_features = compute_spike_cyclepoints(-self.sig, self.fs, self.f_range, self.std)

        # Isolate spikes
        spikes = split_signal(df_features, self.sig)

        self.spikes = spikes
        self._spikes = [sp[~np.isnan(sp)] for sp in self.spikes]

        # Compute shape features
        df_shape_features = compute_shape_features(df_features, self.sig)

        self.df_features = pd.concat((df_features, df_shape_features), axis=1)

        # Compute gaussian features
        if n_gaussians != 0 and self.center_extrema == 'trough':
            params = compute_gaussian_features(self.df_features, self.sig, self.fs,
                                               n_gaussians, maxfev, tol, n_jobs, chunksize, progress)
        elif n_gaussians != 0:
            params = compute_gaussian_features(self.df_features, -self.sig, self.fs,
                                               n_gaussians, maxfev, tol, n_jobs, chunksize, progress)

        if n_gaussians != 0:

            self.params = params

            if len(params[0][:-3]) % 3 == 0:
                param_labels = ['center0', 'center1', 'center2', 'std0', 'std1', 'std2', 'alpha0',
                                'alpha1', 'alpha2', 'height0', 'height1', 'height2', 'sigmoid_max',
                                'sigmoid_growth', 'sigmoid_mid']
            elif len(params[0][:-3]) % 2 == 0:
                param_labels = ['center0', 'center1', 'std0', 'std1', 'alpha0', 'alpha1',
                                'height0', 'height1', 'sigmoid_max', 'sigmoid_growth',
                                'sigmoid_mid']

            self._param_labels = param_labels

            # Invert parameters
            if self.center_extrema == 'peak':

                invert_inds = [9, 10, 11, 12] if len(params[0][:-3]) % 3 == 0 else [6, 7, 10]

                for ind in invert_inds:
                    self.params[:, ind] = self.params[:, ind] * -1

            param_dict = {k: v for k, v in zip(param_labels, self.params.transpose())}
            df_gaussian_features = pd.DataFrame.from_dict(param_dict)

            # Calculate r-squared of the fits
            self.generate_spikes()

            r_squared = np.zeros(len(self.spikes))

            for idx, (sig_cyc, sig_cyc_est) in enumerate(zip(self._spikes, self.spikes_gen)):

                if np.any(np.isnan(sig_cyc_est)):
                    r_squared[idx] = np.nan
                else:
                    # Calculate r-squared
                    residuals = sig_cyc - sig_cyc_est
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((sig_cyc - np.mean(sig_cyc))**2)

                    r_squared[idx] = 1 - (ss_res / ss_tot)

            self.r_squared = r_squared

            df_gaussian_features['r_squared'] = r_squared

            # Merge dataframes
            self.df_features = pd.concat((self.df_features, df_gaussian_features), axis=1)

        # Rename dataframe
        if self.center_extrema == 'peak':

            # Rename columns
            self.df_features = rename_df(self.df_features)


    def normalize(self, inplace=True):
        """Mean and variance normalize spikes.

        Parameters
        ----------
        inplace : bool, optional, default: True
            Modifies the sig attibute in place if True. If false, return a normalized signal.
        """

        if self.spikes is None:
            raise ValueError('The fit method must be successfully called prior to normalization.')

        # Mean and variance normalize
        mean_ = np.nanmean(self.spikes, axis=1)
        scale_ = np.nanstd(self.spikes, axis=1)
        scale_[np.where(scale_ == 0)[0]] = 1

        sig_roll = np.rollaxis(self.spikes.copy(), axis=1)

        sig_roll -= mean_
        sig_roll /= scale_

        if inplace:
            self.spikes = sig_roll.transpose()
        else:
            return sig_roll.transpose()


    def generate_spikes(self):
        """Generate spikes from fit parameters."""

        if self.df_features is None or self.params is None:
            raise ValueError('The fit method must be successfully called prior to generating, '
                             'using either n_gaussians = {2, 3}.')

        self.spikes_gen = []
        for idx, param in enumerate(self.params):

            if np.isnan(param[0]):
                self.spikes_gen.append(np.nan)
                continue

            times_spike = np.arange(0, len(self._spikes[idx])/self.fs, 1/self.fs)

            param = param[~np.isnan(param)]

            spike_gen = sim_action_potential(times_spike, *param)

            # Translate up y-axis for single gaussian fits
            if len(param) == 7:
                spike_gen += self._spikes[idx].max()

            self.spikes_gen.append(spike_gen)


    @savefig
    def plot(self, stack=True, index=None, normalize=False, xlim=None, ax=None):
        """Plot spike results.

        Parameters
        ----------
        stack : bool, optional, default: True
            Plots spikes as 2d arrays ontop of one another. Ignored if index is not None.
        index : int, optional, default: None
            The index in ``df_features`` to plot. If None, plot all spikes.
        normalize : bool, optional, default: False
            Mean centers and variance normalizes spikes attribute when True.
        xlim : tuple
            Upper and lower time limits. Ignored if ``stack`` is True or ``index`` is passed.
        ax : matplotlib.Axes, optional, default: None
            Figure axes upon which to plot.
        """

        if self.df_features is None or self.sig is None or self.fs is None:
            raise ValueError('The fit method must be successfully called prior to plotting.')

        # Plot an individual spike or a spike summary
        if normalize:
            spikes = self.normalize(inplace=False)
        else:
            spikes = self.spikes.copy()

        if stack:
            plot_spikes(self.df_features, self.sig, self.fs, spikes, index, xlim, ax)
        else:
            plot_spikes(self.df_features, self.sig, self.fs, None, index, xlim, ax)


    @savefig
    def plot_gaussian_params(self):
        """Plot gaussian parameters distributions."""

        if self.df_features is None or 'center0' not in self.df_features:
            raise ValueError('The fit method must be successfully called, '
                             'with gaussian=True, prior to plotting.')

        # Setup figure and axes
        fig = plt.figure(figsize=(10, 18))

        gs = GridSpec(6, 2, figure=fig)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        ax6 = fig.add_subplot(gs[3, 0])
        ax7 = fig.add_subplot(gs[3, 1])
        ax8 = fig.add_subplot(gs[4, 0])
        ax9 = fig.add_subplot(gs[4, 1])
        ax10 = fig.add_subplot(gs[5, :])

        # Labels
        titles = self._param_labels

        xlabels = ['mv (normalized)', 'mv (normalized)',
                   'Skew Coefficient', 'mV', 'mV']

        if len(titles) == 15:
            xlabels = [l for lab in xlabels for l in [lab, lab, lab]]
        else:
            xlabels = [l for lab in xlabels for l in [lab, lab]]

        # Plot
        axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        for idx in range(10):

            axes[idx].hist(self.df_features[titles[idx]].values)
            axes[idx].set_title(titles[idx])
            axes[idx].set_xlabel(xlabels[idx])

        ax10.hist(self.df_features['r_squared'].values)
        ax10.set_title('R-Squared')

        # Increase spacing
        fig.subplots_adjust(hspace=.5)


class SpikesMEA(Spikes):
    """Compute multi-electrode array spike features.

    Attributes
    ----------
    df_features : pandas.DataFrame
        Dataframe containing shape and burst features for each spike.
    volts : 2d array
        Voltages for each electrode at the mean spike's start, decay, trough, rise and end points.
    components : 2d array
        Principal component scores for each spike location.
    spikes : 2d array
        The signal associated with each spike (row in the ``df_features``).
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range for narrowband signal of interest (Hz).
    std : float or int, optional, default: 1.5
        The standard deviation used to identify spikes.
    spikes_gen : 2d list
        Spikes generated from fit parameters.
    center_extrema : {'peak', 'trough'}
        The center extrema in the cycle.

        - 'peak' : cycles are defined trough-to-trough
        - 'trough' : cycles are defined peak-to-peak

    find_extrema_kwargs : dict, optional, default: None
        Keyword arguments for function to find peaks an troughs (:func:`~.find_extrema`)
        to change filter parameters or boundary. By default, the filter length is set to three
        cycles of the low cutoff frequency (``f_range[0]``).
    """

    def __init__(self, center_extrema='trough', find_extrema_kwargs=None):
        """Initialize object."""

        super(SpikesMEA, self).__init__(center_extrema=center_extrema,
                                        find_extrema_kwargs=find_extrema_kwargs)

        self.volts = None
        self.components = None


    def fit(self, sigs, fs, f_range, std=2, n_gaussians=0, maxfev=2000,
            tol=1.49e-6, n_jobs=-1, chunksize=1, progress=None):
        """Compute features for each spike.

        Parameters
        ----------
        sig : 1d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        f_range : tuple of (float, float)
            Frequency range for narrowband signal of interest (Hz).
        std : float or int, optional, default: 2
            Standard deviation used to identify spikes.
        n_gaussians : {0, 2, 3}
            Fit a n number of gaussians to each spike. If zero, no gaussian fitting occurs.
        maxfev : int, optional, default: 2000
            Maximum number of calls in curve_fit.
            Only used when n_gaussians is {1, 2}.
        tol : float, optional, default: 1.49e-6
            Relative error desired.
            Only used when n_gaussians is {1, 2}.
        n_jobs : int, optional, default: -1
            Number of jobs to compute features in parallel.
            Only used when n_gaussians is {1, 2}.
        chunksize : int, optional, default: 1
            Number of chunks to split spikes into. Each chunk is submitted as a separate job.
            With a large number of spikes, using a larger chunk size will drastically speed up
            runtime. An optimal chunksize is typically np.ceil(n_spikes/n_jobs).
        progress : {None, 'tqdm', 'tqdm.notebook'}
            Specify whether to display a progress bar. Uses 'tqdm', if installed.
            Only used when n_gaussians is {1, 2}.
        """

        self.sigs = sigs
        self.sig = sigs.mean(axis=0)

        super(SpikesMEA, self).fit(self.sig, fs, f_range, std, n_gaussians,
                                   maxfev, tol, n_jobs, chunksize, progress)

        self.volts = compute_voltage_features(self.df_features, self.sigs)


    def pca(self, pad, n_components, norm_mean=True, norm_std=False):
        """Compute principal component scores for each spike.

        Parameters
        ----------
        pad : int
            Number of samples to include to the left and right of each trough (one-sided).
        n_components : int or float
            The number of components to include (int) or the number of components required to reach
            a minimum variance explained (float).
        norm_mean : bool, optional, default: True
            Normalize the mean of each MEA of each electrode.
        norm_std : bool, optional, default: False
            Normalize the standard deviation of each electrode.
        """

        self.components = compute_pca_features(self.df_features, self.sigs, pad, n_components)

