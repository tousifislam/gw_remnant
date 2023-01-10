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
        if self.use_filter:
            tmp = gwtools.interpolant_h(self.time, dEdt, s=max(dEdt)*0.0003)
            #tmp = gwtools.interpolant_h(self.time, dEdt, s=max(dEdt)*0.00003) # used for q=64
            #tmp = gwtools.interpolant_h(self.time, dEdt, s=max(dEdt)*0.00004) # used for q=32
            dEdt = splev(self.time, tmp, der=0)
        return dEdt

    def _compute_avg_energy_flux_at_start(self):
        """
        computes the average energy flux at the start of the waveform;
        following procedure described in page 2, after Eq(3) of
        https://arxiv.org/pdf/1802.04276.pdf
        """
        indx = np.where(self.time <= self.time[0]+100)
        E_over_100M = integrate.trapz(self.E_dot[indx],self.time[indx])
        Edot_over_100M = E_over_100M/100
        return Edot_over_100M

    def _compute_integration_constant_E0(self, E0_dot):
        """
        integration constant for E(t) obtained using Newtonian calculation;
        Eq(3) of https://arxiv.org/pdf/1802.04276.pdf
        """
        return ((5./1024.)*((self.qinput**3.)/(1.+self.qinput)**6.)*E0_dot)**(1./5.) 

    def _compute_radiated_energy(self):
        """
        compute total radiated energy E(t)
        """
        # compute averaged energy flux at the start of the waveform
        E_dot_avg_at_start = self._compute_avg_energy_flux_at_start()
        # Newtonian calculation : intgration constant
        E_dot_int_const = self._compute_integration_constant_E0(E_dot_avg_at_start)
        # radiated energy
        E_rad = integrate.cumtrapz(self.E_dot, self.time, initial=0.0) + E_dot_int_const 
        return E_rad

    def _compute_bondi_mass(self):
        """ 
        computes Bondi mass or the dynamic mass of the system;
        Eq(4) of https://arxiv.org/pdf/1802.04276.pdf
        """   
        return self.M_initial * (1.0 - self.Eoft)

    def _compute_remnant_mass(self):
        """ 
        computes remnant mass of the binary at a reference time of t=t_end;
        Eq(5) of https://arxiv.org/pdf/1802.04276.pdf
        """
        # compute radiated energy at the final point
        E_rad_final = self.Eoft[-1]
        # compute remnant mass
        M_remnant = self.M_initial * (1 - E_rad_final/(self.M_initial+E_rad_final)) 
        return M_remnant    
    

