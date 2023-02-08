#############################################################################
##
##      Filename: kick_velocity_calculator.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: Estimates kick velocity of a BBH merger
##
##      Modified:
##
#############################################################################

import numpy as np
import scipy.integrate as integrate
from .remnant_mass_calculator import RemnantMassCalculator


class LinearMomentumCalculator(RemnantMassCalculator):
    """
    class to compute the final kick of the remnant black hole
    """
    def __init__(self, time, hdict, qinput, M_initial=1, use_filter=False):
        
        super().__init__(time, hdict, qinput, M_initial, use_filter)
        
        self.lmax = self._get_lmax()
        self.P_dot = np.array([self._compute_dPxdt(), self._compute_dPydt(), 
                               self._compute_dPzdt()])
        self.Poft = self._compute_Poft()
        self.voft = self._compute_voft()
        self.kickoft = self._compute_kickoft()
        self.remnant_kick = self._compute_remnant_kick()
        self.peak_kick = self. _compute_peak_kick()
    
    def _read_dhdt_dict(self, l, m):
        """
        computes time derivative of each mode in a waveform
        pass hdict having both positive and negative modes;
        For unphysical modes e.g. (3,4) etc as well as for modes
        that do not exist in the waveform dictionary, it returns
        zero;
        """
        if l<2 or l>self.lmax:
            return np.zeros(len(self.time),dtype=complex)
        elif m<-l or m>l:
            return np.zeros(len(self.time),dtype=complex)
        else:
            if (l,m) in self.hdict.keys():
                return self.h_dot[l,m]
            else:
                return np.zeros(len(self.time),dtype=complex)
    
    def _get_lmax(self):
        """
        maximum value of \ell modes available in the input waveform data
        """
        return max([mode[0] for mode in self.hdict.keys()])
        
    def _coeffs_a(self, l,m):
        """
        Eq.(3.16) of arXiv:0707.4654
        """
        return ((l-m)*(l+m+1))**0.5 / (l*(l+1) )

    def _coeffs_b(self, l,m):
        """
        Eq.(3.17) of arXiv:0707.4654
        """
        return  (1/(2*l))* (((l-2)*(l+2)*(l+m)*(l+m-1)) / ((2*l-1)*(2*l+1)))**0.5

    def _coeffs_c(self, l,m):
        """
        Eq.(3.18) of arXiv:0707.4654
        """
        return  2*m / (l*(l+1) )

    def _coeffs_d(self, l,m):
        """
        Eq.(3.19) of arXiv:0707.4654
        """
        return  (1/l) * (((l-2)*(l+2)*(l-m)*(l+m))/((2*l-1)*(2*l+1)))**0.5

    def _compute_dPxdt(self):
        """
        derivative of the emitted linear momentum in the x-direction;
        Eq(6) of https://arxiv.org/pdf/1802.04276.pdf
        """
        dPxdt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l,m) = mode
            dPxdt += (1/(8*np.pi)) * np.real( self.h_dot[(l,m)] * (self._coeffs_a(l,m) * np.conj(self._read_dhdt_dict(l,m+1)) 
                                                        + self._coeffs_b(l,-m) * np.conj(self._read_dhdt_dict(l-1,m+1)) 
                                                        - self._coeffs_b(l+1,m+1) * np.conj(self._read_dhdt_dict(l+1,m+1))))
        return dPxdt
    
    def _compute_dPydt(self):
        """
        derivative of the emitted linear momentum in the y-direction;
        Eq(7) of https://arxiv.org/pdf/1802.04276.pdf
        """
        dPydt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l,m) = mode
            dPydt += (1/(8*np.pi)) * np.imag( self.h_dot[(l,m)] * (self._coeffs_a(l,m) * np.conj(self._read_dhdt_dict(l,m+1)) 
                                                        + self._coeffs_b(l,-m) * np.conj(self._read_dhdt_dict(l-1,m+1)) 
                                                        - self._coeffs_b(l+1,m+1) * np.conj(self._read_dhdt_dict(l+1,m+1))))
        return dPydt
    
    def _compute_dPzdt(self):
        """
        derivative of the emitted linear momentum in the z-direction;
        Eq(8) of https://arxiv.org/pdf/1802.04276.pdf
        """
        dPzdt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l,m) = mode
            dPzdt += (1/(16*np.pi)) * np.real(self.h_dot[(l,m)] * ( self._coeffs_c(l,m) * np.conj(self._read_dhdt_dict(l,m)) 
                                                         + self._coeffs_d(l,m) * np.conj(self._read_dhdt_dict(l-1,m)) 
                                                         + self._coeffs_d(l+1,m) * np.conj(self._read_dhdt_dict(l+1,m))))
        return dPzdt
    
    def _compute_Poft(self):
        """
        Radiated linear momentum vector as a function of time;
        Obtained by integrating Eq(6),(7) and (8) of https://arxiv.org/pdf/1802.04276.pdf
        """
        Pxoft = integrate.cumtrapz(self.P_dot[0], self.time, initial=0.0)
        Pyoft = integrate.cumtrapz(self.P_dot[1], self.time, initial=0.0)
        Pzoft = integrate.cumtrapz(self.P_dot[2], self.time, initial=0.0)
        return np.array([Pxoft, Pyoft, Pzoft])
    
    def _compute_voft(self):
        """
        Velocity vector of the center of mass of the system as a function of time;
        Eq(13) of https://arxiv.org/pdf/1802.04276.pdf
        """
        return np.transpose(self.Poft / self.Moft)
    
    def _compute_kickoft(self):
        """
        Time profile of the kick magnitude imparted to the system;
        """
        return np.array([np.linalg.norm(self.voft[i]) for i in range(len(self.time))])

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
    
    def _compute_peak_kick(self):
        """
        Computes the peak of kick velocity profile
        """
        return self._get_peak_via_quadratic_fit(self.time, self.kickoft)[1]
    
    def _compute_remnant_kick(self):
        """
        Final kick velocity of the remnant;
        Eq(14) of Time profile of the kick imparted to the system;
        """
        return self.kickoft[-1]
    
    
