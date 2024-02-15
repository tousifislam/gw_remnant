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
    def __init__(self, time, hdict, qinput, spin1_input=None, spin2_input=None, 
                 ecc_input=None, E_initial=None, L_initial=None, 
                 M_initial=1, use_filter=False):
        """
        Inputs:
        
        time (float): array of geometric times
        hdict (float): dictionary of geometric waveform with modes as keys.
                       keys should be as '(2,2)', '(3,3)' and so on
        qinput (float): mass ratio value
        spin1_input (array of floats): spin vector for the primary black hole at the start of the waveform. 
                                       Example- [0,0,0.1]
                                       default: None
        spin2_input (array of floats): spin vector for the secondary black hole at the start of the waveform. 
                                       Example- [0,0,0.1]
                                       default: None
        ecc_input (float): eccentricity estimate at the start of the waveform; 
                           gw_remnant does not change whether this estimate is correct;
                           user is supposed to know the eccentricity of the binary at the reference time;
                           default: None
        E_initial (float): initial energy of the binary
                           default: None - in that case, we compute it using PN expression;
                           set it to zero if you want to inspect change of energy/momenta;
                           set it to a given value if you know the initial energy e.g. from NR simulaiton;
        L_initial (float): initial angular momentum of the binary
                           default: None - in that case, we compute it using PN expression;
                           set it to zero if you want to inspect change of energy/momenta;
                           set it to a given value if you know the initial energy e.g. from NR simulaiton;
                           
        M_initial (float): initial total mass of the binary;
                           default: 1M.
        use_filter (binary): if true, smooths the data while computing the flux; 
                             default: False
                           
        Outputs:
        
        L_peak: peak luminosity in geometric units
                Eq(27) of https://arxiv.org/pdf/2301.07215.pdf
        """
        super().__init__(time, hdict, qinput, spin1_input, spin2_input, 
                 ecc_input, E_initial, L_initial, M_initial, use_filter)
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
        Computes the peak luminosity;
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
        

