#############################################################################
##
##      Filename: gw_plotter.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: Makes quick plots for energy and momenta profiles
##
##      Modified:
##
#############################################################################

import numpy as np

from gw_remnant.remnant_calculators.peak_luminosity_calculator import PeakLuminosityCalculator
from gw_remnant.remnant_calculators.kick_velocity_calculator import LinearMomentumCalculator
from gw_remnant.remnant_calculators.remnant_mass_calculator import RemnantMassCalculator

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
matplotlib.rcParams['mathtext.fontset'] ='stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral' 
matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally
plt.rcParams["figure.figsize"] = (14,10)
plt.rcParams['font.size'] = '18'


class GWPlotter(PeakLuminosityCalculator, LinearMomentumCalculator, 
                RemnantMassCalculator):
    """
    Class to make quick diagnostic plots for the mass, energy and momentum
    as a function of time
    """
    def _mk_subplot(self, X, Y, Ylabel, Xlabel=None, title=None):
        """
        plot subplots for each kinds of attributes
        """       
        if title is not None:
            plt.title('q=%.2f'%self.qinput, fontsize=14)
        plt.plot(X, Y)
        plt.ylabel('%s'%Ylabel, fontsize=12)
        if Xlabel is not None:
            plt.xlabel('%s'%Xlabel, fontsize=12)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.grid()
    
    def plot_mass_energy(self):
        """
        plot masses and energy as a function of time
        """
        plt.figure(figsize=(8,6))
        plt.subplot(411)
        self._mk_subplot(self.time, np.real(self.h_dot[(2,2)]), Ylabel='rh_22/M', 
                         title='q=%.2f'%self.qinput)
        plt.subplot(412)
        self._mk_subplot(self.time, self.E_dot, Ylabel='E_dot [M]')
        plt.subplot(413)
        self._mk_subplot(self.time, self.Eoft, Ylabel='E(t) [M]')
        plt.subplot(414)
        self._mk_subplot(self.time, self.Moft, Ylabel='M(t) [M]', Xlabel='time [M]')
        plt.tight_layout()
        plt.show()
        
    def plot_linear_momentum(self):
        """
        plot linear momentum vector as a function of time
        """
        plt.figure(figsize=(12,8))
        plt.subplot(311)
        self._mk_subplot(self.time, np.real(self.Poft[0]), Ylabel='Px(t)', 
                         title='q=%.2f'%self.qinput)
        plt.subplot(312)
        self._mk_subplot(self.time, self.Poft[1], Ylabel='Py(t)')
        plt.subplot(313)
        self._mk_subplot(self.time, self.Poft[2], Ylabel='Pz(t)', Xlabel='time [M]')
        plt.tight_layout()
        plt.show()
        
    def plot_angular_momentum(self):
        """
        plot angular momentum as a function of time
        """
        plt.figure(figsize=(12,8))
        plt.subplot(311)
        self._mk_subplot(self.time, np.real(self.Joft[0]), Ylabel='Jx(t)', 
                         title='q=%.2f'%self.qinput)
        plt.subplot(312)
        self._mk_subplot(self.time, self.Joft[1], Ylabel='Jy(t)')
        plt.subplot(313)
        self._mk_subplot(self.time, self.Joft[2], Ylabel='Jz(t)', Xlabel='time [M]')
        plt.tight_layout()
        plt.show()
    
    def plot_kick_velocity(self):
        """
        plot the magnitude of the kick velocity as a function of time
        """
        plt.figure(figsize=(6,4))
        self._mk_subplot(self.time, self.kickoft, Ylabel='|v(t) [c]|', Xlabel='time [M]',
                         title='q=%.2f'%self.qinput)
        plt.tight_layout()
        plt.show()