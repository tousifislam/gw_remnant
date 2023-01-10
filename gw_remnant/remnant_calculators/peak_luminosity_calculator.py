#############################################################################
##
##      Filename: peak_luminosity_calculator.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: Estimates peak GW luminosity of a BBH merger
##
##      Modified:
##
#############################################################################

import numpy as np
from .remnant_mass_calculator import RemnantMassCalculator
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class PeakLuminosityCalculator(RemnantMassCalculator):
    """
    Class to compute the peak luminosity of the GW radiation using a spline
    fit to the data
    """
    def __init__(self, time, hdict, qinput, M_initial=1, use_filter=False):
        
        super().__init__(time, hdict, qinput, M_initial, use_filter)
        self.L_peak = self._compute_peak_luminosity()
    
    def _get_peaks_via_spline_fit(self, t, func):
        """
        Finds the peak time of a function quadratically
        Fits the function to a quadratic over the 5 points closest to the argmax func.
        t : an array of times
        func : array of function values
        Returns: tpeak, fpeak
        """
        # Use a 4th degree spline for interpolation, so that the roots of its derivative can be found easily.
        spl = spline(t, func, k=4)
        # find the critical points
        cr_pts = spl.derivative().roots()
        # also check the endpoints of the interval
        cr_pts = np.append(cr_pts, (t[0], func[-1]))  
        # critial values
        cr_vals = spl(cr_pts)
        # we only care about the maximas
        max_index = np.argmax(cr_vals)
        return cr_pts[max_index], cr_vals[max_index]
    
    def _compute_peak_luminosity(self):
        """
        computes the peak luminosity;
        Eq(1) of https://arxiv.org/pdf/2010.00120.pdf
        """
        # find the max value of the discrete series
        discrete_peak_index = np.argmax(self.E_dot)
        # use 10 points in each side of the max point
        indx_begin = discrete_peak_index - 10
        indx_end = discrete_peak_index + 10
        time_cut = self.time[indx_begin:indx_end]
        L_cut = self.E_dot[indx_begin:indx_end]
        # find the continuous peak using spline fit
        L_peak = self._get_peaks_via_spline_fit(time_cut, L_cut)[1]
        return L_peak
        

