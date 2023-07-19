#############################################################################
##
##      Filename: waveform_generator.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: Generates surrogate waveforms and surrogate remnant properties
##
##      Modified:
##
#############################################################################

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

PATH_TO_BHPTNRSur = "/home/UMDAR.UMASSD.EDU/tislam/work/BHPTNRSurrogate"
import sys
sys.path.append(PATH_TO_BHPTNRSur+"/surrogates")
import BHPTNRSur1dq1e4 as bhptsur

import gwtools

import lal
import lalsimulation as lalsim
from lal import MSUN_SI, MTSUN_SI, PC_SI, C_SI

import gwsurrogate
sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

import surfinBH
fit_name = 'NRSur3dq8Remnant'
fit = surfinBH.LoadFits(fit_name)


class WaveformGenerator():
    """
    Class to generate gravitational waveforms for different models;
    At this moment, the class supports three models:
        (i) BHPTNRSur1dq1e4;
        (ii) NRHybSur3dq8;
    It further computes the remnant quantities using the following remnant
    surrogates:
        (i) NRHybSur3dq8Remnant;
    """
    def __init__(self, mass_ratio, modes=None, common_times=None, f_low=None, 
                 get_NRSur=True, get_BHPT=True):
        
        # mass ratio
        self.qinput = mass_ratio
        
        # f_low for NRHybSur3dq8
        if f_low is None:
            self.f_low = 3e-3
        else:
            self.f_low = f_low
            
        # common time grid for both waveforms
        if common_times is None:
            self.common_times = np.arange(-5000.0, 50.0, 0.1)
        else:
            self.common_times = common_times

        # modes
        if modes is None:
            self.modes = [(2,2),(2,1),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4)]
        else:
            self.modes = modes
            
        # generate waveforms
        if get_NRSur:
            self.hnr = self._generate_nrsur()
        if get_BHPT:
            self.hbhpt = self._generate_bhptsur()
        print('final common time grid : [%.2f,%.2f]'%(self.common_times[0],self.common_times[-1]))
        
        # generate NRSur remnant outputs
        if self.qinput<=10:
            self.rem_sur = self._NRSurRemnant_predictions()
        
    def _generate_nrsur(self):
        """
        generate NRHybSur3dq8 waveform
        """
        chiA = [0, 0, 0.0]
        chiB = [0, 0, 0.0]
        # step size, Units of M
        dt = 0.1        
        # initial frequency, Units of cycles/M
        # dyn stands for dynamics and is always None for this model
        t, h, dyn = sur(self.qinput, chiA, chiB, dt=dt, f_low=self.f_low) 
        # make sure t=0 is the peak of 22 amplitude
        t_peak = self._peak_time(t, h[(2,2)])
        t = t - t_peak
        print('NRSur original time grid : [%.2f,%.2f]'%(t[0],t[-1]))
        # only select modes that matches self.modes
        h_out = {}
        for mode in self.modes:
            # cast waveforms in the common time grid
            h_out[mode] = gwtools.gwtools.interpolate_h(t, h[mode], self.common_times)
            # get negative m modes
            h_out[(mode[0],-mode[-1])] = ((-1)**mode[0]) * np.conjugate(h_out[mode])
        return h_out
    
    def _generate_bhptsur(self):
        """
        generate BHPTNRSur1dq1e4 waveform
        """
        t, h = bhptsur.generate_surrogate(q=self.qinput, modes=self.modes)
        # make sure t=0 is the peak of 22 amplitude
        t_peak = self._peak_time(t, h[(2,2)])
        t = t - t_peak
        print('BHPTSur original time grid : [%.2f,%.2f]'%(t[0],t[-1]))
        for mode in h.keys():
            # cast waveforms in the common time grid
            h[mode] = gwtools.gwtools.interpolate_h(t, h[mode],self.common_times)
        return h
    
    def _NRSurRemnant_predictions(self):
        """
        NRSur3dq8 remnant output
        """
        chiA = [0,0,0.0]   
        chiB = [0,0,0.0]
        # All of these together
        mf, chif, vf, mf_err, chif_err, vf_err = fit.all(self.qinput, chiA, chiB)
        return mf, mf_err, chif[-1], chif_err[-1], np.linalg.norm(vf), np.linalg.norm(vf_err)

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

    def _peak_time(self, t, modes):
        """
        wrapper to find the peak time using 4th order spline, using 22 mode of the waveform
        """
        normSqrVsT = abs(modes)**2
        return self._get_peaks_via_spline_fit(t, normSqrVsT)[0]
