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
    def _mk_subplot(self, X, Y, Ylabel, Xlabel=None, title=None):
        """
        Create a formatted subplot for time series data.
        
        Args:
            X (np.ndarray): Time array or x-axis data
            Y (np.ndarray): Physical quantity to plot on y-axis
            Ylabel (str): Label for y-axis
            Xlabel (str): Label for x-axis. If None, no label is added
            title (str): Plot title. If None, no title is added
        """
        if title is not None:
            plt.title('q=%.2f' % self.qinput, fontsize=16)
        plt.plot(X, Y)
        plt.ylabel('%s' % Ylabel, fontsize=14)
        if Xlabel is not None:
            plt.xlabel('%s' % Xlabel, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.grid()
    
    
    def plot_mass_energy(self):
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
        """
        fig = plt.figure(figsize=(11, 10))
        
        # Define time split
        t_split = -500
        pre_merger_mask = self.time <= t_split
        post_merger_mask = self.time > t_split
        
        # Define subplot positions [left, bottom, width, height]
        left_width = 0.52
        right_width = 0.35
        gap = 0.08
        height = 0.20  # Smaller height since we have 4 rows
        v_gap = 0.02   # Smaller vertical gap
        
        # Row positions (bottom to top)
        row_bottoms = [0.08, 0.08 + height + v_gap, 0.08 + 2*(height + v_gap), 
                       0.08 + 3*(height + v_gap)]
        
        # Y-axis labels and data
        ylabel_list = ['$M(t)$ $[M]$', '$E(t)$ $[M]$', '$\\dot{E}$ $[M]$', '$rh_{22}/M$']
        data_list = [self.Moft, self.Eoft, self.E_dot, np.real(self.h_dot[(2, 2)])]
        
        for i, (ylabel, data, bottom) in enumerate(zip(ylabel_list, data_list, row_bottoms)):
            # Left subplot (pre-merger, t <= -500)
            ax_left = fig.add_axes([0.1, bottom, left_width, height])
            ax_left.plot(self.time[pre_merger_mask], data[pre_merger_mask])
            ax_left.set_ylabel(ylabel, fontsize=16)
            ax_left.tick_params(labelsize=14)
            ax_left.grid(True)
            ax_left.set_xlim(xmin=self.time[pre_merger_mask][0], xmax=-500)
            
            # Add title only to top left subplot
            if i == 3:  # Top row (4th in the list)
                try:
                    ax_left.set_title('$[q,\\chi_1z]=[%.2f,%.2f]$' % (self.qinput, self.spin1_input), 
                                    fontsize=16, pad=10)
                except:
                    ax_left.set_title('$[q,\\chi_1z]=[%.2f,%.2f]$' % (self.qinput, self.spin1_input[-1]), 
                                    fontsize=16, pad=10)
            
            # Only show x-label and x-tick labels for bottom row
            if i == 0:
                ax_left.set_xlabel('$t$ $[M]$', fontsize=16)
            else:
                ax_left.tick_params(labelbottom=False)
            
            # Right subplot (post-merger, t > -500)
            ax_right = fig.add_axes([0.1 + left_width + gap, bottom, right_width, height])
            ax_right.plot(self.time[post_merger_mask], data[post_merger_mask])
            ax_right.tick_params(labelsize=14)
            ax_right.grid(True)
            ax_right.set_xlim(xmin=-500, xmax=self.time[post_merger_mask][-1])
            
            # Only show x-label and x-tick labels for bottom row
            if i == 0:
                ax_right.set_xlabel('$t$ $[M]$', fontsize=16)
            else:
                ax_right.tick_params(labelbottom=False)
        
        plt.show()

        
    def plot_linear_momentum(self):
        """
        Plot linear momentum components as a function of time.
        
        Creates a 3-row, 2-column figure showing the x, y, and z components of the
        linear momentum vector. Each row is split into pre-merger (t <= -500M, 60% width)
        and post-merger (t > -500M, 40% width) regions for better visualization of
        different phases of the evolution.
        
        The momentum components are in units of total mass M.
        """
        fig = plt.figure(figsize=(11, 8))
        
        # Define time split
        t_split = -500
        pre_merger_mask = self.time <= t_split
        post_merger_mask = self.time > t_split
        
        # Define subplot positions [left, bottom, width, height]
        # 60% width for pre-merger, 40% for post-merger, small vertical gaps
        left_width = 0.52
        right_width = 0.35
        gap = 0.08
        height = 0.28
        v_gap = 0.03
        
        # Row positions (bottom to top)
        row_bottoms = [0.08, 0.08 + height + v_gap, 0.08 + 2*(height + v_gap)]
        
        # Y-axis labels
        ylabel_list = ['$P_x(t)$', '$P_y(t)$', '$P_z(t)$']
        P_components = [self.Poft[0], self.Poft[1], self.Poft[2]]
        
        for i, (ylabel, P_comp, bottom) in enumerate(zip(ylabel_list, P_components, row_bottoms)):
            # Left subplot (pre-merger, t <= -500)
            ax_left = fig.add_axes([0.1, bottom, left_width, height])
            ax_left.plot(self.time[pre_merger_mask], np.real(P_comp[pre_merger_mask]))
            ax_left.set_ylabel(ylabel, fontsize=16)
            ax_left.tick_params(labelsize=14)
            ax_left.grid(True)
            ax_left.set_xlim(xmin=self.time[pre_merger_mask][0], xmax=-500)
            
            # Add title only to top left subplot
            if i == 2:
                try:
                    ax_left.set_title('$[q,\\chi_1z]=[%.2f,%.2f]$' % (self.qinput, self.spin1_input), 
                                    fontsize=16, pad=10)
                except:
                    ax_left.set_title('$[q,\\chi_1z]=[%.2f,%.2f]$' % (self.qinput, self.spin1_input[-1]), 
                                    fontsize=16, pad=10)
            
            # Only show x-label and x-tick labels for bottom row
            if i == 0:
                ax_left.set_xlabel('$t$ $[M]$', fontsize=16)
            else:
                ax_left.tick_params(labelbottom=False)
            
            # Right subplot (post-merger, t > -500)
            ax_right = fig.add_axes([0.1 + left_width + gap, bottom, right_width, height])
            ax_right.plot(self.time[post_merger_mask], np.real(P_comp[post_merger_mask]))
            ax_right.tick_params(labelsize=14)
            ax_right.grid(True)
            ax_right.set_xlim(xmin=-500, xmax=self.time[post_merger_mask][-1])
            
            # Only show x-label and x-tick labels for bottom row
            if i == 0:
                ax_right.set_xlabel('$t$ $[M]$', fontsize=16)
            else:
                ax_right.tick_params(labelbottom=False)
        
        plt.show()
    
        
    def plot_angular_momentum(self):
        """
        Plot angular momentum components as a function of time.
        
        Creates a 3-row, 2-column figure showing the x, y, and z components of the
        angular momentum vector. Each row is split into pre-merger (t <= -500M, 60% width)
        and post-merger (t > -500M, 40% width) regions for better visualization of
        different phases of the evolution.
        
        The angular momentum components are in units of total mass squared M^2.
        """
        fig = plt.figure(figsize=(11, 8))
        
        # Define time split
        t_split = -500
        pre_merger_mask = self.time <= t_split
        post_merger_mask = self.time > t_split
        
        # Define subplot positions [left, bottom, width, height]
        # 60% width for pre-merger, 40% for post-merger, small vertical gaps
        left_width = 0.52  # Reduced slightly
        right_width = 0.35
        gap = 0.08  # Increased gap to prevent overlap
        height = 0.28
        v_gap = 0.03
        
        # Row positions (bottom to top)
        row_bottoms = [0.08, 0.08 + height + v_gap, 0.08 + 2*(height + v_gap)]
        
        # Y-axis labels
        ylabel_list = ['$J_x(t)$', '$J_y(t)$', '$J_z(t)$']
        J_components = [self.Joft[0], self.Joft[1], self.Joft[2]]
        
        for i, (ylabel, J_comp, bottom) in enumerate(zip(ylabel_list, J_components, row_bottoms)):
            # Left subplot (pre-merger, t <= -500)
            ax_left = fig.add_axes([0.1, bottom, left_width, height])
            ax_left.plot(self.time[pre_merger_mask], np.real(J_comp[pre_merger_mask]))
            ax_left.set_ylabel(ylabel, fontsize=16)
            ax_left.tick_params(labelsize=14)
            ax_left.grid(True)
            ax_left.set_xlim(xmin=self.time[pre_merger_mask][0],xmax=-500)
            
            # Add title only to top left subplot
            if i == 2:
                try:
                    ax_left.set_title('$[q,\\chi_1z]=[%.2f,%.2f]$' % (self.qinput, self.spin1_input), fontsize=16, pad=10)
                except:
                    ax_left.set_title('$[q,\\chi_1z]=[%.2f,%.2f]$' % (self.qinput, self.spin1_input[-1]), fontsize=16, pad=10)
            
            # Only show x-label and x-tick labels for bottom row
            if i == 0:
                ax_left.set_xlabel('$t$ $[M]$', fontsize=16)
            else:
                ax_left.tick_params(labelbottom=False)
            
            # Right subplot (post-merger, t > -500)
            ax_right = fig.add_axes([0.1 + left_width + gap, bottom, right_width, height])
            ax_right.plot(self.time[post_merger_mask], np.real(J_comp[post_merger_mask]))
            ax_right.tick_params(labelsize=14)
            ax_right.grid(True)
            ax_right.set_xlim(xmin=-500,xmax=self.time[post_merger_mask][-1])
            
            # Only show x-label and x-tick labels for bottom row
            if i == 0:
                ax_right.set_xlabel('$t$ $[M]$', fontsize=16)
            else:
                ax_right.tick_params(labelbottom=False)
        
        plt.show()

    
    def plot_kick_velocity(self):
        """
        Plot the magnitude of kick velocity as a function of time.
        
        Creates a single-row, 2-column figure showing the time evolution of the recoil
        (kick) velocity magnitude imparted to the remnant black hole due to asymmetric
        gravitational wave emission. The plot is split into pre-merger (t <= -500M, 60% width)
        and post-merger (t > -500M, 40% width) regions.
        
        Kick velocity is given in units of the speed of light c.
        """
        fig = plt.figure(figsize=(11, 4))
        
        # Define time split
        t_split = -500
        pre_merger_mask = self.time <= t_split
        post_merger_mask = self.time > t_split
        
        # Define subplot positions [left, bottom, width, height]
        left_width = 0.52
        right_width = 0.35
        gap = 0.08
        height = 0.65
        bottom = 0.20
        
        ylabel = '$|v(t)|$ $[c]$'
        
        # Left subplot (pre-merger, t <= -500)
        ax_left = fig.add_axes([0.1, bottom, left_width, height])
        ax_left.plot(self.time[pre_merger_mask], self.kickoft[pre_merger_mask])
        ax_left.set_ylabel(ylabel, fontsize=16)
        ax_left.set_xlabel('$t$ $[M]$', fontsize=16)
        ax_left.tick_params(labelsize=14)
        ax_left.grid(True)
        ax_left.set_xlim(xmin=self.time[pre_merger_mask][0], xmax=-500)
        
        # Add title
        try:
            ax_left.set_title('$[q,\\chi_1z]=[%.2f,%.2f]$' % (self.qinput, self.spin1_input), 
                            fontsize=16, pad=10)
        except:
            ax_left.set_title('$[q,\\chi_1z]=[%.2f,%.2f]$' % (self.qinput, self.spin1_input[-1]), 
                            fontsize=16, pad=10)
        
        # Right subplot (post-merger, t > -500)
        ax_right = fig.add_axes([0.1 + left_width + gap, bottom, right_width, height])
        ax_right.plot(self.time[post_merger_mask], self.kickoft[post_merger_mask])
        ax_right.set_xlabel('$t$ $[M]$', fontsize=16)
        ax_right.tick_params(labelsize=14)
        ax_right.grid(True)
        ax_right.set_xlim(xmin=-500, xmax=self.time[post_merger_mask][-1])
        
        plt.show()