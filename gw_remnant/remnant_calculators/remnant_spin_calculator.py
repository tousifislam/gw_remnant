#############################################################################
##
##      Filename: remnant_spin_calculator.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: Estimates spin of the final black hole in a BBH merger
##
##      Modified:
##
#############################################################################

import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import splev, splrep
import gwtools

from .remnant_mass_calculator import RemnantMassCalculator
from .kick_velocity_calculator import LinearMomentumCalculator


class AngularMomentumCalculator(LinearMomentumCalculator, RemnantMassCalculator):
    """
    Class to compute the final spin of the binary black hole;
    """
    def __init__(self, time, hdict, qinput, M_initial=1, use_filter=False):
        
        super().__init__(time, hdict, qinput, M_initial, use_filter)
        
        self.J_dot = np.array([self._compute_dJxdt(), self._compute_dJydt(), 
                               self._compute_dJzdt()])
        self.Joft = self._compute_Joft()
        self.L_initial = self._compute_initial_angular_momentum()
        self.spinoft = self._compute_spin_evolution()
        self.remnant_spin = self._compute_remnant_spin()
        
    def _coeff_f(self,l,m):
        """
        Eq. (3.25) of arXiv:0707.4654
        """
        return  (l*(l+1)-m*(m+1))**0.5
    
    def _compute_dJxdt(self):
        """
        derivative of the emitted angular momentum in the x-direction;
        Eq(15) of https://arxiv.org/pdf/1802.04276.pdf
        """
        dJxdt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l,m) = mode
            dJxdt += (1/(32*np.pi)) * np.imag( self.hdict[(l,m)] \
                                    * (self._coeff_f(l,m) * np.conj(self._read_dhdt_dict(l,m+1)) \
                                    + self._coeff_f(l,-m) * np.conj(self._read_dhdt_dict(l,m-1))))
            
        return dJxdt
    
    def _compute_dJydt(self):
        """
        derivative of the emitted angular momentum in the y-direction;
        Eq(16) of https://arxiv.org/pdf/1802.04276.pdf
        """
        dJydt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l,m) = mode
            dJydt += (1/(32*np.pi)) * np.real( self.hdict[(l,m)] \
                                    * (self._coeff_f(l,m) * np.conj(self._read_dhdt_dict(l,m+1)) 
                                    + self._coeff_f(l,-m) * np.conj(self._read_dhdt_dict(l,m-1))))
        return dJydt
    
    def _compute_dJzdt(self):
        """
        derivative of the emitted angular momentum in the z-direction;
        Eq(17) of https://arxiv.org/pdf/1802.04276.pdf
        """
        dJzdt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l,m) = mode
            dJzdt += (1/(16*np.pi)) * m * np.imag( self.hdict[(l,m)] * np.conj(self.h_dot[(l,m)]))
        return dJzdt
    
    def _compute_Joft(self):
        """
        Radiated angular momentum vector as a function of time;
        Obtained by integrating Eq(15),(16),(17) of https://arxiv.org/pdf/1802.04276.pdf
        """
        Jxoft = integrate.cumtrapz(self.J_dot[0], self.time, initial=0.0)
        Jyoft = integrate.cumtrapz(self.J_dot[1], self.time, initial=0.0)
        Jzoft = integrate.cumtrapz(self.J_dot[2], self.time, initial=0.0)
        return np.array([Jxoft, Jyoft, Jzoft])
    
    def _compute_initial_angular_momentum(self):
        """
        Computes initial angular momentum L_orb using Eq(2.36) https://arxiv.org/pdf/1111.5378.pdf;
        Post-Newtonian calculation at 3PN order
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2,2)])
        orb_frequency = abs(np.gradient(orb_phase,edge_order=2)/np.gradient(self.time,edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        term_1 = (3/2 + nu/6) * x
        term_2 = (27/8 - 19*nu/8 + nu*nu/24) * x * x
        term_3 = (135/16 + (-6889/144 + 41*np.pi*np.pi/24)*nu 
                  + 31*nu*nu/24 + 7*nu*nu*nu/1296) * x * x * x
        
        return (nu/x**0.5) * (1 + term_1 + term_2 + term_3)
    
    def _compute_spin_evolution(self):
        """
        Spin evolution of the binary;
        Eq(20) of https://arxiv.org/pdf/2101.11015.pdf
        """
        spin_f = np.zeros(len(self.time))
        for i in range(len(spin_f)):
            spin_f[i] = (self.L_initial - self.Joft[2][i])/self.remnant_mass**2
        return spin_f
    
    def _compute_remnant_spin(self):
        """
        compute the final spin of the binary;
        Eq(20) of https://arxiv.org/pdf/2101.11015.pdf
        """
        remnant_spin = (self.L_initial - self.Joft[2][-1])/self.remnant_mass**2
        return remnant_spin
    
    
    
    
