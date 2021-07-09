"""Class objects to compute features for spiking data."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from neurodsp.plts.utils import savefig

from bycycle import Bycycle

from bycycle.spikes.features import compute_shape_features, compute_gaussian_features
from bycycle.spikes.features.gaussians import sim_gaussian_cycle
from bycycle.spikes.cyclepoints import compute_spike_cyclepoints
from bycycle.spikes.plts import plot_spikes
from bycycle.spikes.plts import plot_gaussian_fit
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
        self.z_thresh_K=0.5
        self.z_thresh_cond=0.5

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


    def fit(self, sig, fs, f_range, std=2,
            maxfev=2000, gaussian_fit=True, tol=1.49e-6, n_jobs=-1, chunksize=1, progress=None, z_thresh_K=0.5, z_thresh_cond=0.5):
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
        gaussian_fit : {True, False}
            Fit gaussians to each spike. If False no gaussian fitting occurs.
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
        if gaussian_fit:
            if self.center_extrema == 'trough':
                params = compute_gaussian_features(self.df_features, self.sig, self.fs,
                                                    maxfev, tol, n_jobs, chunksize, progress, z_thresh_K, z_thresh_cond)
            else:
                params = compute_gaussian_features(self.df_features, -self.sig, self.fs,
                                                    maxfev, tol, n_jobs, chunksize, progress, z_thresh_K, z_thresh_cond)

            self.params = params

            param_labels = ['Cond_center', 'Cond_std', 'Cond_alpha',  'Cond_height',
                            'Cond_r_squared', 'Na_center', 'Na_std', 'Na_alpha', 'Na_height',
                            'Na_r_squared','K_center','K_std', 'K_alpha', 'K_height', 'K_r_squared']


            self._param_labels = param_labels

            param_dict = {k: v for k, v in zip(param_labels, self.params.transpose())}
            df_gaussian_features = pd.DataFrame.from_dict(param_dict)

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
            raise ValueError('The fit method must be successfully called prior to generating ')

        self.spikes_gen = []
        for idx, param in enumerate(self.params):
            print("idx", idx)
            print("param1", param)
            if np.isnan(param[0]):
                self.spikes_gen.append(np.nan)
                continue

            times_spike = np.arange(0, len(self._spikes[idx])/self.fs, 1/self.fs)

            param = param[~np.isnan(param)]
            print("param",param)
            spike_gen = sim_gaussian_cycle(times_spike, *param)

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

        if self.df_features is None or 'Na_center' not in self.df_features:
            raise ValueError('The fit method must be successfully called, '
                             'with gaussian_fit=True, prior to plotting.')

        # Setup figure and axes
        fig = plt.figure(figsize=(10, 18))

        gs = GridSpec(8, 2, figure=fig)

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
        ax10 = fig.add_subplot(gs[5, 0])
        ax11 = fig.add_subplot(gs[5, 1])
        ax12 = fig.add_subplot(gs[6, 0])
        ax13 = fig.add_subplot(gs[6, 1])
        ax14 = fig.add_subplot(gs[7, 0])

        # Labels
        titles = self._param_labels

        xlabels = ['Relative position in fit window', 'mv (normalized)',
                   'Skew Coefficient', 'mV', '']

        # Set labels for all 3 gaussians
        xlabels = xlabels*3

        # Set colors for gaussian plots
        hist_colors = 5*["green"] + 5*["royalblue"] +  5*["darkorange"]

        # Plot
        axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14]
        for idx in range(15):

            axes[idx].hist(self.df_features[titles[idx]].values, color = hist_colors[idx])
            axes[idx].set_title(titles[idx])
            axes[idx].set_xlabel(xlabels[idx])

        # Increase spacing
        fig.subplots_adjust(hspace=.8)


    @savefig
    def plot_gaussian_fit_steps(self, index=None):
        """Plot gaussian fit steps for a given spike."""

        if self.df_features is None and np.isnan(self.df_features['Na_center']):
            raise ValueError('No successful gaussian fit found for spike.')

        else:
            if index:
                plot_gaussian_fit(self.df_features.iloc[index], self.sig, self.fs,
                                  self.z_thresh_cond, self.z_thresh_K)

            else:
                # Loop through all spikes
                for spk in range(len(self.df_features)):
                    print("Gaussian fit for spike with index = " + str(spk))
                    plot_gaussian_fit(self.df_features.iloc[spk], self.sig, self.fs,
                                      self.z_thresh_cond, self.z_thresh_K)
