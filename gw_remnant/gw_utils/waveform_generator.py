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
        (iii) SEOBNRv4HM;
    It further computes the remnant quantities using the following remnant
    surrogates:
        (i) NRHybSur3dq8Remnant;
    """
    def __init__(self, mass_ratio, modes=None, common_times=None, f_low=None, 
                 get_NRSur=True, get_SEOB=True, get_BHPT=True):
        
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
        if get_NRSur:#self.qinput<=10:
            self.hnr = self._generate_nrsur()
        if get_SEOB:#self.qinput<=99:
            self.hseob = self._generate_SEOBNRv4HM()
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
        return mf, mf_err, np.linalg.norm(vf), np.linalg.norm(vf_err)
    
    
    def _SEOBNRv4HM(self, q, chi1z, chi2z, deltaTOverM, omega0):
        """ 
        base function to generate waveform for 'SEOBNRv4HM' using LALSuite.
        Returns dimless time and dict containing dimless complex strain of all
        available modes.
        """
        
        ## use M=10 and distance=1 Mpc, but will scale these out before outputting h
        M = 10      # dimless mass
        distance = 1.0e6 * PC_SI

        MT = M*MTSUN_SI
        f_low = omega0/np.pi/MT
        f_ref = f_low

        # component masses of the binary
        m1_kg =  M*MSUN_SI*q/(1.+q)
        m2_kg =  M*MSUN_SI/(1.+q)

        nqcCoeffsInput=lal.CreateREAL8Vector(10)

        sphtseries, dyn, dynHI = lalsim.SimIMRSpinAlignedEOBModes( \
            deltaTOverM * MT, m1_kg, m2_kg, f_low, distance, chi1z, chi2z, 41, \
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., nqcCoeffsInput, 0)

        h_dict = {}
        type_struct = type(sphtseries)
        while type(sphtseries) is type_struct:
            l = sphtseries.l
            m = sphtseries.m
            hlm = sphtseries.mode.data.data
            hlm *= distance/MT/C_SI              # rescale to rhOverM
            h_dict['h_l%dm%d'%(l,m)] = hlm
            # go to the next mode because stupid EOBNR
            sphtseries = sphtseries.next

        ##HACK HACK HACK     #FIXME FIXME
        ## SEOBNRv4HM has a different tetrad convention (to be fixed soon
        # apparently), that shows up as a minus sign for all modes
        for key in h_dict:
            h_dict[key] *= -1

        t = deltaTOverM *np.arange(len(h_dict['h_l2m2']))      # dimensionless time
        return t, h_dict


    def _generate_SEOBNRv4HM(self):
        """
        wrapper to generate SEOB waveform
        """
        # generate SEOB waveform
        t, h = self._SEOBNRv4HM(q=self.qinput, chi1z=0.0, chi2z=0.0, deltaTOverM=0.1, omega0=1.5e-2)
        # make sure t=0 is the peak of 22 amplitude
        t_peak = self._peak_time(t, h['h_l2m2'])
        t = t - t_peak
        
        print('SEOB original time grid : [%.2f,%.2f]'%(t[0],t[-1]))
        # name modes according to our needs
        hSEOB={}
        # have dictionary keys as (l,m)
        for mode in h.keys():
            # cast waveforms in the common time grid
            hSEOB[(int(mode[3]),int(mode[5]))] = gwtools.gwtools.interpolate_h(t, h[mode], self.common_times)
            # get negative m modes
            hSEOB[(int(mode[3]),-int(mode[5]))] = ((-1)**int(mode[3])) * np.conjugate(hSEOB[(int(mode[3]),int(mode[5]))])
        return hSEOB


    def _get_peak_via_quadratic_fit(self, t, func):
        """
        Finds the peak time of a function quadratically
        Fits the function to a quadratic over the 5 points closest to the argmax func.
        t : an array of times
        func : array of function values
        Returns: tpeak, fpeak
        """

        # Find the time closest to the peak, making sure we have room on either side
        index = np.argmax(func)
        index = max(2, min(len(t) - 3, index))

        # Do a quadratic fit to 5 points,
        # subtracting t[index] to make the matrix inversion nice
        testTimes = t[index-2:index+3] - t[index]
        testFuncs = func[index-2:index+3]
        xVecs = np.array([np.ones(5),testTimes,testTimes**2.])
        invMat = np.linalg.inv(np.array([[v1.dot(v2) for v1 in xVecs] \
            for v2 in xVecs]))

        yVec = np.array([testFuncs.dot(v1) for v1 in xVecs])
        coefs = np.array([yVec.dot(v1) for v1 in invMat])
        return t[index] - coefs[1]/(2.*coefs[2]), coefs[0] - coefs[1]**2./4/coefs[2]
    
    def _peak_time(self, t, modes):
        """
        wrapper to find the peak time quadratically, using 22 mode of the waveform
        """
        normSqrVsT = abs(modes)**2
        return self._get_peak_via_quadratic_fit(t, normSqrVsT)[0]