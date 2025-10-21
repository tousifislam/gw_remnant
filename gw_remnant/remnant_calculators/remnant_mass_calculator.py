#############################################################################
##
##      Filename: remnant_mass_calculator.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: Estimates mass of the final black hole in a BBH merger
##
##      Modified:
##
#############################################################################

import numpy as np
import scipy.integrate as integrate
from scipy import signal
from scipy.interpolate import splev, splrep
import gwtools
from .initial_energy_momenta import InitialEnergyMomenta

class RemnantMassCalculator(InitialEnergyMomenta):
    """
    Class to compute the remnant mass of a binary given a inspiral-merger-ringdown 
    waveform.
    
    This is the base class for all remnant property calculations performed
    in this package.
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
        """
        super().__init__(time, hdict, qinput, spin1_input, spin2_input, 
                 ecc_input, E_initial, L_initial)
    
        self.use_filter = use_filter
        self.M_initial = M_initial
        self.h_dot = self._compute_dhdt()
        self.E_dot = self._compute_energy_flux_dEdt()
        self.Eoft = self._compute_radiated_energy()
        self.E_rad = self.Eoft[-1]
        self.Moft = self._compute_bondi_mass()
        self.remnant_mass = self._compute_remnant_mass()

    def _compute_dhdt(self):
        """
        Computes time derivative of each mode in the waveform dictionary;
        pass hdict having both positive and negative m modes;
        """
        hdot_dict = {mode : np.gradient(self.hdict[mode],edge_order=2)
                     /np.gradient(self.time,edge_order=2) 
                     for mode in self.hdict.keys()}
        
        if self.use_filter:
            
            sdict = {(2,2): 0.01, (2,1): 0.001, (3,1): 0.00008, 
                     (3,2): 0.00008, (3,3): 0.001, (4,2): 0.00001, 
                     (4,3): 0.00001, (4,4): 0.00001}
                        
            hdot_dict = {}
            for mode in self.hdict.keys():
                tmp = gwtools.interpolant_h(self.time, self.hdict[mode], 
                                            s=sdict[(mode[0],abs(mode[1]))])
                hdot_dict[mode] = splev(self.time, tmp[0], der=1) + \
                                    1j*splev(self.time, tmp[1], der=1)
    
        return hdot_dict

    def _compute_energy_flux_dEdt(self):
        """
        Computes the total energy flux from all modes of a waveform given their 
        respective time derivatives;
        Eq(2) of https://arxiv.org/pdf/1802.04276.pdf
        """
        dEdt = 0.0
        for mode in self.h_dot.keys():
            # individual mode contribution
            dEdt_mode = (1/(16*np.pi))*(np.abs(self.h_dot[mode])**2) 
            dEdt += dEdt_mode
        # if use_filter=True, we will smooth out the energy derivate a bit
        if self.use_filter:
            tmp = gwtools.interpolant_h(self.time, dEdt, s=max(dEdt)*0.0003)
            dEdt = splev(self.time, tmp, der=0)
        return dEdt
    
    def _compute_radiated_energy(self):
        """
        Compute total radiated energy E(t)
        """
        try:
            E_rad = integrate.cumtrapz(self.E_dot, self.time, initial=0.0) + self.E_initial
        except:
            E_rad = integrate.cumulative_trapezoid(self.E_dot, self.time, initial=0.0) + self.E_initial
        return E_rad

    def _compute_bondi_mass(self):
        """ 
        Computes Bondi mass or the dynamic mass of the system;
        Eq(4) of https://arxiv.org/pdf/1802.04276.pdf
        """   
        return self.M_initial * (1.0 - self.Eoft + self.E_initial)

    def _compute_remnant_mass(self):
        """ 
        Computes remnant mass of the binary at a reference time of t=t_end;
        Eq(9) of https://arxiv.org/abs/2301.07215
        """
        M_remnant = self.M_initial * (1.0 - self.E_rad)
        return M_remnant    
    

