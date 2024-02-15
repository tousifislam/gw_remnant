#############################################################################
##
##      Filename: energy_momenta.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: Computes initial energy and angular momentum using mostly
##                   PN expressions
##
##      Modified:
##
#############################################################################

import numpy as np
import gwtools
import matplotlib.pyplot as plt

class InitialEnergyMomenta():
    """
    Class to assign the initial energy and angular momentum for a binary
    """
    def __init__(self, time, hdict, qinput, spin1_input=None, spin2_input=None, 
                 ecc_input=None, E_initial=None, L_initial=None):
        """
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
        """
        self.time = time
        self.hdict = hdict
        # check modes
        for mode in self.hdict.keys():
            if mode[1]==0:
                pass
            else:
                if (mode[0],mode[1]) in self.hdict.keys():
                    pass
                else:
                    self.hdict[(mode[0],-mode[1])] = (-1)**mode[0] * self.hdict[mode]
        
        # mass ratio
        self.qinput = qinput        
        
        # spin1
        if spin1_input is None:
            self.spin1_input = np.array([0.0,0.0,0.0])
        else:
            self.spin1_input = np.array(spin1_input)
          
        # spin2
        if spin2_input is None:
            self.spin2_input = np.array([0.0,0.0,0.0])
        else:
            self.spin2_input = np.array(spin2_input)
           
        # eccentricity
        if ecc_input is None:
            self.ecc_input = 0.0 
        else:
            self.ecc_input = ecc_input
            
        # energy
        if E_initial is None:
            self.E_initial = self._E0_from_PN()
        else:
            self.E_initial = E_initial
            
        # angular momentum
        if L_initial is None:
            self.L_initial = self._L0_from_PN_nonspinning()
        else:
            self.L_initial = L_initial
        
    
    def _E0_from_PN(self):
        """
        Integration constant for E(t) obtained using post-Newtonian calculation 
        We combine 4PN non-spinning result and 3PN eccentric corrections from the 
        following papers
        Eq(2.35) of https://arxiv.org/pdf/1111.5378.pdf
        Eq.(3) from https://arxiv.org/pdf/2304.11185.pdf
        Eq.(6.1a) and Eq.(6.5a) of https://arxiv.org/pdf/0908.3854.pdf
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2,2)])
        orb_frequency = abs(np.gradient(orb_phase,edge_order=2)/np.gradient(self.time,edge_order=2))
        x = orb_frequency[0]**(2/3)
        
        nu = gwtools.q_to_nu(self.qinput)
        delta = gwtools.q_to_delta(self.qinput) 
        
        chi_s = (self.spin1_input + self.spin2_input)/2
        chi_a = (self.spin1_input - self.spin2_input)/2 
        L_N = np.array([0,0,1]) 
        
        # Euler's constant
        gamma_E = 0.57721566490153286060
        
        # spinning correction
        term_15 = ((8/3 - 4/3 * nu) * np.dot(chi_s, L_N) + (8/3) * delta * np.dot(chi_a, L_N)) * x**(1.5)
        term_25 = ((8 - 121/9 * nu + 2/9 * nu**2) * np.dot(chi_s, L_N) 
                   + (8 - 31/9 * nu) * delta * np.dot(chi_a, L_N)) * x * x
        
        # non-spinning + eccentric corrections + spinning corrections
        term_1 =  x / (1 - self.ecc_input**2) * (-3/4 - nu/12 + self.ecc_input**2 * (-5/4 + nu/12))
        term_2 = x**2 / (1 - self.ecc_input**2)**2 * (-67/8 + 35/8 * nu - 1/24 * nu**2
                                              + self.ecc_input**2 * (-19/4 + 21/4 * nu + 1/12 * nu**2)
                                              + self.ecc_input**4 * (5/8 - 5/8 * nu - 1/24 * nu**2)
                                              + (1 - self.ecc_input**2)**(3/2) * (5 - 2 * nu)) + x * x * nu * (
                                              (np.dot(chi_s, chi_s) - np.dot(chi_a, chi_a)) - 3 * (
                                                    (np.dot(chi_s, L_N))**2 - (np.dot(chi_a, L_N))**2
                                               )
                                               ) + (1/2 - nu) * (
                                               np.dot(chi_s, chi_s) + np.dot(chi_a, chi_a) - 3 * (
                                               (np.dot(chi_s, L_N))**2 + (np.dot(chi_a, L_N))**2
                                                )
                                                ) + delta * (
                                                    np.dot(chi_s, chi_a) - 3 * np.dot(chi_s, L_N) * np.dot(chi_a, L_N)
                                                )
        term_3 = x**3 / (1 - self.ecc_input**2)**3 * (-835/64 + (18319/192 - 41/16 * np.pi**2) * nu
                                               - 169/32 * nu**2 - 35/5184 * nu**3
                                               + self.ecc_input**2 * (-3703/64 + (21235/192 - 41/64 * np.pi**2) * nu
                                                               - 7733/288 * nu**2 + 35/1728 * nu**3)
                                               + self.ecc_input**4 * (103/64 - 547/192 * nu - 1355/288 * nu**2 - 35/1728 * nu**3)
                                               + self.ecc_input**6 * (185/192 + 75/64 * nu + 25/288 * nu**2 + 35/5184 * nu**3)
                                               + np.sqrt(1 - self.ecc_input**2) * (5/2 + (-641/18 + 41/96 * np.pi**2) * nu
                                                                           + 11/3 * nu**2 + self.ecc_input**2 * (-35 + (394/9 - 41/96 * np.pi**2) * nu
                                                                                                           - 1/3 * nu**2)
                                                                           + self.ecc_input**4 * (5/2 + 23/6 * nu - 10/3 * nu**2)))

        # non-spinning circular
        term_4 = (-3969/128 + (-123671/5760 + 9037/1536 * np.pi**2
                             + 896/15 * gamma_E + 448/15 * np.log(16 * x)) * nu
                + (-498449/3456 + 3157/576 * np.pi**2) * nu**2
                + 301/1728 * nu**3 + 77/31104 * nu**4) * x**4
        
        return 0.5 * nu * x * ( 1 +  term_1 + term_15 + term_2 + term_25 + term_3 + term_4)
    
    def _E0_from_PN_nonspinning(self):
        """
        Integration constant for E(t) obtained using post-Newtonian calculation;
        Eq(2.35) of https://arxiv.org/pdf/1111.5378.pdf
        term_4 is taken from Eq.(3) from https://arxiv.org/pdf/2304.11185.pdf
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2,2)])
        orb_frequency = abs(np.gradient(orb_phase,edge_order=2)/np.gradient(self.time,edge_order=2))
        
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        
        term_1 = (-3/4 - nu/12) * x
        term_2 = (-27/8 + 19*nu/8 - nu*nu/24) * x * x
        term_3 = (-675/64 + (34445/576 - 205*np.pi*np.pi/96)*nu 
                  - 144*nu*nu/96 - 35*nu*nu*nu/5184) * x * x * x
        
        term_4 = (-3969/128 + (-123671/5760 + 9037/1536 * np.pi**2
                             + 896/15 * gamma_E + 448/15 * np.log(16 * x)) * nu
                + (-498449/3456 + 3157/576 * np.pi**2) * nu**2
                + 301/1728 * nu**3 + 77/31104 * nu**4) * x**4
        
        return 0.5 * nu * x * ( 1 +  term_1 + term_2 + term_3 + term_4)
    
    
    def _E0_from_PN_nonspinning_eccentric(self):
        """
        Integration constant for E(t) obtained using post-Newtonian calculation 
        for eccentric non-spinning binaries;
        Eq.(6.1a) and Eq.(6.5a) of https://arxiv.org/pdf/0908.3854.pdf
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2,2)])
        orb_frequency = abs(np.gradient(orb_phase,edge_order=2)/np.gradient(self.time,edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)

        # Euler's constant
        gamma_E_value = 0.57721566490153286060
        
        term1 =  x / (1 - self.ecc_input**2) * (-3/4 - nu/12 + self.ecc_input**2 * (-5/4 + nu/12))
        term2 = x**2 / (1 - self.ecc_input**2)**2 * (-67/8 + 35/8 * nu - 1/24 * nu**2
                                              + self.ecc_input**2 * (-19/4 + 21/4 * nu + 1/12 * nu**2)
                                              + self.ecc_input**4 * (5/8 - 5/8 * nu - 1/24 * nu**2)
                                              + (1 - self.ecc_input**2)**(3/2) * (5 - 2 * nu))
        term3 = x**3 / (1 - self.ecc_input**2)**3 * (-835/64 + (18319/192 - 41/16 * np.pi**2) * nu
                                               - 169/32 * nu**2 - 35/5184 * nu**3
                                               + self.ecc_input**2 * (-3703/64 + (21235/192 - 41/64 * np.pi**2) * nu
                                                               - 7733/288 * nu**2 + 35/1728 * nu**3)
                                               + self.ecc_input**4 * (103/64 - 547/192 * nu - 1355/288 * nu**2 - 35/1728 * nu**3)
                                               + self.ecc_input**6 * (185/192 + 75/64 * nu + 25/288 * nu**2 + 35/5184 * nu**3)
                                               + np.sqrt(1 - self.ecc_input**2) * (5/2 + (-641/18 + 41/96 * np.pi**2) * nu
                                                                           + 11/3 * nu**2 + self.ecc_input**2 * (-35 + (394/9 - 41/96 * np.pi**2) * nu
                                                                                                           - 1/3 * nu**2)
                                                                           + self.ecc_input**4 * (5/2 + 23/6 * nu - 10/3 * nu**2)))

        return 0.5 * nu * x * (1 + term1 + term2 + term3)
    
    
    def _E0_from_PN_spinning_(self):
        """
        Integration constant for E(t) obtained using post-Newtonian calculation 
        for eccentric non-spinning binaries;
        Eq.(6.1a) and Eq.(6.5a) of https://arxiv.org/pdf/0908.3854.pdf
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2,2)])
        orb_frequency = abs(np.gradient(orb_phase,edge_order=2)/np.gradient(self.time,edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        delta = gwtools.q_to_delta(self.qinput) 
        
        chi_s = (self.chi1 + self.chi2)/2
        chi_a = (self.chi1 - self.chi2)/2 
        L_N = np.array([0,0,1]) 
                
        term_1 = (-3/4 - nu/12) * x
        term_15 = ((8/3 - 4/3 * nu) * np.dot(chi_s, L_N) + (8/3) * delta * np.dot(chi_a, L_N)) * x**(1.5)
        term_2 = (-27/8 + 19*nu/8 - nu*nu/24) * x * x + x * x * nu * (
                (np.dot(chi_s, chi_s) - np.dot(chi_a, chi_a)) - 3 * (
                    (np.dot(chi_s, L_N))**2 - (np.dot(chi_a, L_N))**2
                )
                ) + (1/2 - nu) * (
                    np.dot(chi_s, chi_s) + np.dot(chi_a, chi_a) - 3 * (
                        (np.dot(chi_s, L_N))**2 + (np.dot(chi_a, L_N))**2
                    )
                ) + delta * (
                    np.dot(chi_s, chi_a) - 3 * np.dot(chi_s, L_N) * np.dot(chi_a, L_N)
                )
        term_25 = ((8 - 121/9 * nu + 2/9 * nu**2) * np.dot(chi_s, L_N) 
                   + (8 - 31/9 * nu) * delta * np.dot(chi_a, L_N)) * x * x
        term_3 = (-675/64 + (34445/576 - 205*np.pi*np.pi/96)*nu 
                  - 144*nu*nu/96 - 35*nu*nu*nu/5184) * x * x * x
        
        return 0.5 * nu * x * ( 1 +  term_1 + term_15 + term_2 + term_25 + term_3)
    

    def _L0_from_PN_nonspinning(self):
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