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

from .initial_energy_momenta import InitialEnergyMomenta
from .remnant_mass_calculator import RemnantMassCalculator
from .kick_velocity_calculator import LinearMomentumCalculator


class AngularMomentumCalculator(LinearMomentumCalculator, RemnantMassCalculator, InitialEnergyMomenta):
    """
    Class to compute the final spin of the binary black hole;
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
                             
        Output:
        
        J_dot: angular momentum change
        Joft: angular momentum evolution
        spinoft: spin evolution of the binary
        remnant_spin: spin of the remnant black hole
        """
        super().__init__(time, hdict, qinput, spin1_input, spin2_input, 
                 ecc_input, E_initial, L_initial, M_initial, use_filter)
        
        self.J_dot = np.array([self._compute_dJxdt(), self._compute_dJydt(), 
                               self._compute_dJzdt()])
        self.Joft = self._compute_Joft()
        self.spinoft = self._compute_spin_evolution()
        self.remnant_spin = self._compute_remnant_spin()
        
    def _coeff_f(self,l,m):
        """
        Eq. (3.25) of arXiv:0707.4654
        """
        return  (l*(l+1)-m*(m+1))**0.5
    
    def _compute_dJxdt(self):
        """
        Provides the derivative of the emitted angular momentum in the x-direction;
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
        Provides the derivative of the emitted angular momentum in the y-direction;
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
        Provides the derivative of the emitted angular momentum in the z-direction;
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
        try:
            Jxoft = integrate.cumtrapz(self.J_dot[0], self.time, initial=0.0)
            Jyoft = integrate.cumtrapz(self.J_dot[1], self.time, initial=0.0)
            Jzoft = integrate.cumtrapz(self.J_dot[2], self.time, initial=0.0)
        except:
            Jxoft = integrate.cumulative_trapezoid(self.J_dot[0], self.time, initial=0.0)
            Jyoft = integrate.cumulative_trapezoid(self.J_dot[1], self.time, initial=0.0)
            Jzoft = integrate.cumulative_trapezoid(self.J_dot[2], self.time, initial=0.0)
        return np.array([Jxoft, Jyoft, Jzoft])
    
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
    
    
    
    
