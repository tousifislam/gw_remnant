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


class RemnantMassCalculator():
    """
    Class to compute the remnant mass of a binary given a inspiral-merger-ringdown 
    waveform.
    
    This is the base class for all remnant property calculations performed
    in this package.
    """
    
    def __init__(self, time, hdict, qinput, M_initial=1, use_filter=False):
        
        self.time = time
        self.hdict = hdict
        self.qinput = qinput
        self.use_filter = use_filter
        self.M_initial = M_initial
        self.h_dot = self._compute_dhdt()
        self.E_dot = self._compute_energy_flux_dEdt()
        self.E0 = self._compute_integration_constant_E0_from_PN()
        self.Eoft = self._compute_radiated_energy()
        self.E_rad = self.Eoft[-1]
        self.Moft = self._compute_bondi_mass()
        self.remnant_mass = self._compute_remnant_mass()

    def _compute_dhdt(self):
        """
        computes time derivative of each mode in the waveform dictionary;
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
        computes the total energy flux from all modes of a waveform given their 
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
 
    def _compute_integration_constant_E0_from_PN(self):
        """
        integration constant for E(t) obtained using Newtonian calculation;
        Eq(2.35) of https://arxiv.org/pdf/1111.5378.pdf
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2,2)])
        orb_frequency = abs(np.gradient(orb_phase,edge_order=2)/np.gradient(self.time,edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        term_1 = (-3/4 - nu/12) * x
        term_2 = (-27/8 + 19*nu/8 - nu*nu/24) * x * x
        term_3 = (-675/64 + (34445/576 - 205*np.pi*np.pi/96)*nu 
                  - 144*nu*nu/96 - 35*nu*nu*nu/5184) * x * x * x
        return 0.5*nu*x * ( 1 +  term_1 + term_2 + term_3 )
    
    def _compute_radiated_energy(self):
        """
        compute total radiated energy E(t)
        """
        E_rad = integrate.cumtrapz(self.E_dot, self.time, initial=0.0) + self.E0
        return E_rad

    def _compute_bondi_mass(self):
        """ 
        computes Bondi mass or the dynamic mass of the system;
        Eq(4) of https://arxiv.org/pdf/1802.04276.pdf
        """   
        return self.M_initial * (1.0 - self.Eoft + self.E0)

    def _compute_remnant_mass(self):
        """ 
        computes remnant mass of the binary at a reference time of t=t_end;
        Eq(9) of https://arxiv.org/abs/2301.07215
        """
        M_remnant = self.M_initial * (1.0 - self.E_rad)
        return M_remnant    
    

