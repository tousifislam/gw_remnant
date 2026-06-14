#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : gw_plotter.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Provides quick diagnostic plotting utilities for gravitational wave remnant
#   properties including mass, energy, linear momentum, angular momentum, and
#   kick velocity profiles as functions of time.
#
#====================================================================================

from __future__ import annotations

import numpy as np

from ..remnant_calculators.peak_luminosity_calculator import PeakLuminosityCalculator
from ..remnant_calculators.kick_velocity_calculator import LinearMomentumCalculator
from ..remnant_calculators.remnant_mass_calculator import RemnantMassCalculator

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
matplotlib.rcParams['mathtext.fontset'] ='stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.linewidth'] = 1 #set the value globally
plt.rcParams["figure.figsize"] = (14,10)
plt.rcParams['font.size'] = '18'

# Time boundary separating pre-merger and post-merger regions in plots
_T_SPLIT = -500


class GWPlotter(PeakLuminosityCalculator, LinearMomentumCalculator,
                RemnantMassCalculator):
    """
    Diagnostic plotting utilities for gravitational wave remnant properties.

    This class provides methods to visualize the time evolution of various
    physical quantities extracted from gravitational waveforms, including
    mass, energy, linear momentum, angular momentum, and kick velocity.

    Inherits From:
        PeakLuminosityCalculator: Provides energy flux calculations
        LinearMomentumCalculator: Provides linear momentum calculations
        RemnantMassCalculator: Provides mass and energy calculations
    """

    def _get_title_string(self):
        """Build title string showing mass ratio and primary spin z-component."""
        chi1z = self.chi1[2]
        return f'$[q,\\chi_{{1z}}]=[{self.q:.2f},{chi1z:.2f}]$'

    def _plot_split_timeseries(self, data_list, ylabel_list, figsize=(11, 8),
                               height=0.28, v_gap=0.03, bottom_start=0.08,
                               save_path=None):
        """Plot time series split into pre-merger and post-merger panels.

        Args:
            data_list (list): List of arrays to plot (one per row, bottom to top).
            ylabel_list (list): Corresponding y-axis labels.
            figsize (tuple): Figure size.
            height (float): Height of each subplot row.
            v_gap (float): Vertical gap between rows.
            bottom_start (float): Bottom margin of the lowest row.
            save_path (str or None): If provided, save figure to this path instead
                of displaying it.

        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        n_rows = len(data_list)
        fig = plt.figure(figsize=figsize)

        pre_mask = self.time <= _T_SPLIT
        post_mask = self.time > _T_SPLIT

        left_width = 0.52
        right_width = 0.35
        gap = 0.08

        row_bottoms = [bottom_start + i * (height + v_gap) for i in range(n_rows)]

        for i, (ylabel, data, bottom) in enumerate(zip(ylabel_list, data_list, row_bottoms)):
            # Left subplot (pre-merger)
            ax_left = fig.add_axes([0.1, bottom, left_width, height])
            ax_left.plot(self.time[pre_mask], np.real(data[pre_mask]))
            ax_left.set_ylabel(ylabel, fontsize=16)
            ax_left.tick_params(labelsize=14)
            ax_left.grid(True)
            ax_left.set_xlim(xmin=self.time[pre_mask][0], xmax=_T_SPLIT)

            # Title on top row only
            if i == n_rows - 1:
                ax_left.set_title(self._get_title_string(), fontsize=16, pad=10)

            # x-label on bottom row only
            if i == 0:
                ax_left.set_xlabel('$t$ $[M]$', fontsize=16)
            else:
                ax_left.tick_params(labelbottom=False)

            # Right subplot (post-merger)
            ax_right = fig.add_axes([0.1 + left_width + gap, bottom, right_width, height])
            ax_right.plot(self.time[post_mask], np.real(data[post_mask]))
            ax_right.tick_params(labelsize=14)
            ax_right.grid(True)
            ax_right.set_xlim(xmin=_T_SPLIT, xmax=self.time[post_mask][-1])

            if i == 0:
                ax_right.set_xlabel('$t$ $[M]$', fontsize=16)
            else:
                ax_right.tick_params(labelbottom=False)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        return fig

    def plot_mass_energy(self, save_path: str | None = None) -> matplotlib.figure.Figure:
        """
        Plot mass and energy evolution as a function of time.

        Creates a 4-row, 2-column figure showing:
        - Top row: Real part of (2,2) mode waveform derivative
        - Second row: Energy flux (time derivative of energy)
        - Third row: Total energy as a function of time
        - Bottom row: Mass as a function of time

        Each row is split into pre-merger (t <= -500M, 60% width) and post-merger
        (t > -500M, 40% width) regions for better visualization.

        All quantities are plotted in geometric units where G=c=1.
        Time is in units of total mass M.

        Args:
            save_path (str or None): If provided, save figure to this path.

        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        data_list = [self.Moft, self.Eoft, self.E_dot, np.real(self.h_dot[(2, 2)])]
        ylabel_list = ['$M(t)$ $[M]$', '$E(t)$ $[M]$', '$\\dot{E}$ $[M]$', '$rh_{22}/M$']
        return self._plot_split_timeseries(data_list, ylabel_list, figsize=(11, 10),
                                           height=0.20, v_gap=0.02, save_path=save_path)

    def plot_linear_momentum(self, save_path: str | None = None) -> matplotlib.figure.Figure:
        """
        Plot linear momentum components as a function of time.

        Creates a 3-row, 2-column figure showing the x, y, and z components of the
        linear momentum vector. Each row is split into pre-merger (t <= -500M, 60% width)
        and post-merger (t > -500M, 40% width) regions for better visualization of
        different phases of the evolution.

        The momentum components are in units of total mass M.

        Args:
            save_path (str or None): If provided, save figure to this path.

        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        data_list = [self.Poft[0], self.Poft[1], self.Poft[2]]
        ylabel_list = ['$P_x(t)$', '$P_y(t)$', '$P_z(t)$']
        return self._plot_split_timeseries(data_list, ylabel_list, save_path=save_path)

    def plot_angular_momentum(self, save_path: str | None = None) -> matplotlib.figure.Figure:
        """
        Plot angular momentum components as a function of time.

        Creates a 3-row, 2-column figure showing the x, y, and z components of the
        angular momentum vector. Each row is split into pre-merger (t <= -500M, 60% width)
        and post-merger (t > -500M, 40% width) regions for better visualization of
        different phases of the evolution.

        The angular momentum components are in units of total mass squared M^2.

        Args:
            save_path (str or None): If provided, save figure to this path.

        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        data_list = [self.Joft[0], self.Joft[1], self.Joft[2]]
        ylabel_list = ['$J_x(t)$', '$J_y(t)$', '$J_z(t)$']
        return self._plot_split_timeseries(data_list, ylabel_list, save_path=save_path)

    def plot_kick_velocity(self, save_path: str | None = None) -> matplotlib.figure.Figure:
        """
        Plot the magnitude of kick velocity as a function of time.

        Creates a single-row, 2-column figure showing the time evolution of the recoil
        (kick) velocity magnitude imparted to the remnant black hole due to asymmetric
        gravitational wave emission. The plot is split into pre-merger (t <= -500M, 60% width)
        and post-merger (t > -500M, 40% width) regions.

        Kick velocity is given in units of the speed of light c.

        Args:
            save_path (str or None): If provided, save figure to this path.

        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        data_list = [self.kickoft]
        ylabel_list = ['$|v(t)|$ $[c]$']
        return self._plot_split_timeseries(data_list, ylabel_list, figsize=(11, 4),
                                           height=0.65, v_gap=0.03, bottom_start=0.20,
                                           save_path=save_path)

    def plot_spin_vector(self, save_path: str | None = None) -> matplotlib.figure.Figure:
        """
        Plot dimensionless spin vector components as a function of time.

        Creates a 3-row, 2-column figure showing the x, y, and z components of the
        dimensionless spin vector. Each row is split into pre-merger (t <= -500M, 60% width)
        and post-merger (t > -500M, 40% width) regions for better visualization of
        different phases of the evolution.

        The spin components are dimensionless (chi = S/M^2).

        Args:
            save_path (str or None): If provided, save figure to this path.

        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        data_list = [self.spin_vector_oft[0], self.spin_vector_oft[1],
                     self.spin_vector_oft[2]]
        ylabel_list = ['$\\chi_x(t)$', '$\\chi_y(t)$', '$\\chi_z(t)$']
        return self._plot_split_timeseries(data_list, ylabel_list, save_path=save_path)
