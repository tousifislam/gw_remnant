#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : initial_energy_momenta.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Computes initial energy and angular momentum for binary black hole systems
#   using post-Newtonian expressions. Supports non-spinning, spinning, and
#   eccentric binaries up to 4PN order.
#
#====================================================================================

from __future__ import annotations

import numpy as np
import gwtools


class InitialEnergyMomenta:
    """
    Calculator for initial energy and angular momentum of binary black holes.
    
    This class computes the initial orbital energy and angular momentum of a
    binary black hole system using post-Newtonian (PN) approximations. It supports:
    - Non-spinning circular binaries (up to 4PN)
    - Spinning binaries (up to 3PN with spin corrections)
    - Eccentric binaries (up to 3PN with eccentricity corrections)
    
    Initial conditions can be either computed from PN expressions or provided
    directly by the user (e.g., from numerical relativity simulations).
    
    Args:
        time (np.ndarray): Array of time values in geometric units (M)
        hdict (dict): Dictionary of complex waveform modes with (l,m) tuple keys,
            e.g., {(2,2): h_22(t), (3,3): h_33(t), ...}
        qinput (float): Mass ratio q = m1/m2, where m1 >= m2
        spin1_input (list or np.ndarray): Spin vector [sx, sy, sz] for primary black
            hole at reference time, in dimensionless units. Default is None (no spin)
        spin2_input (list or np.ndarray): Spin vector [sx, sy, sz] for secondary black
            hole at reference time, in dimensionless units. Default is None (no spin)
        ecc_input (float): Eccentricity at reference time. User must provide accurate
            value; code does not validate. Default is None (circular orbit)
        E_initial (float): Initial energy in units of total mass M. If None, computed
            using PN expressions. Set to 0 to track energy changes relative to reference.
            Default is None
        L_initial (float): Initial angular momentum in units of M^2. If None, computed
            using PN expressions. Set to 0 to track angular momentum changes relative
            to reference. Default is None
    
    Attributes:
        time (np.ndarray): Time array
        hdict (dict): Waveform mode dictionary
        qinput (float): Mass ratio
        spin1_input (np.ndarray): Primary spin vector
        spin2_input (np.ndarray): Secondary spin vector
        ecc_input (float): Eccentricity
        E_initial (float): Initial orbital energy
        L_initial (float): Initial orbital angular momentum
    """
    
    def __init__(self, time, hdict, qinput, spin1_input=None, spin2_input=None, 
                 ecc_input=None, E_initial=None, L_initial=None):
        
        self.time = time
        self.hdict = hdict
        
        # Ensure negative m modes are present using symmetry relation
        for mode in self.hdict.keys():
            if mode[1] == 0:
                pass
            else:
                if (mode[0], mode[1]) in self.hdict.keys():
                    pass
                else:
                    self.hdict[(mode[0], -mode[1])] = (-1)**mode[0] * np.conjugate(self.hdict[mode])
        
        # Mass ratio
        self.qinput = qinput        
        
        # Spin vectors
        if spin1_input is None:
            self.spin1_input = np.array([0.0, 0.0, 0.0])
        else:
            self.spin1_input = np.array(spin1_input)
          
        if spin2_input is None:
            self.spin2_input = np.array([0.0, 0.0, 0.0])
        else:
            self.spin2_input = np.array(spin2_input)
           
        # Eccentricity
        if ecc_input is None:
            self.ecc_input = 0.0 
        else:
            self.ecc_input = ecc_input
            
        # Initial energy
        if E_initial is None:
            self.E_initial = self._E0_from_PN()
        else:
            self.E_initial = E_initial
            
        # Initial angular momentum
        if L_initial is None:
            self.L_initial = self._L0_from_PN_nonspinning()
        else:
            self.L_initial = L_initial
        
    def _E0_from_PN(self):
        """
        Compute initial energy using post-Newtonian expressions.
        
        Combines 4PN non-spinning circular terms with 3PN eccentric and spinning
        corrections to compute the initial orbital energy. This is the most complete
        expression available in the class.
        
        References are from:
        - Eq. (2.35) of https://arxiv.org/pdf/1111.5378.pdf (circular non-spinning 4PN)
        - Eq. (3) of https://arxiv.org/pdf/2304.11185.pdf (4PN coefficient)
        - Eq. (6.1a) and (6.5a) of https://arxiv.org/pdf/0908.3854.pdf (eccentric 3PN)
        
        Returns:
            [float]: Initial orbital energy in units of total mass M.
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        x = orb_frequency[0]**(2/3)
        
        nu = gwtools.q_to_nu(self.qinput)
        delta = gwtools.q_to_delta(self.qinput) 
        
        chi_s = (self.spin1_input + self.spin2_input) / 2
        chi_a = (self.spin1_input - self.spin2_input) / 2 
        L_N = np.array([0, 0, 1]) 
        
        # Euler's constant
        gamma_E = 0.57721566490153286060
        
        # 1.5PN and 2.5PN spinning corrections
        term_15 = ((8/3 - 4/3 * nu) * np.dot(chi_s, L_N) + 
                   (8/3) * delta * np.dot(chi_a, L_N)) * x**(1.5)
        term_25 = ((8 - 121/9 * nu + 2/9 * nu**2) * np.dot(chi_s, L_N) + 
                   (8 - 31/9 * nu) * delta * np.dot(chi_a, L_N)) * x**2
        
        # 1PN with eccentricity
        term_1 = (x / (1 - self.ecc_input**2) * 
                 (-3/4 - nu/12 + self.ecc_input**2 * (-5/4 + nu/12)))
        
        # 2PN with eccentricity and spin-orbit/spin-spin
        term_2 = (x**2 / (1 - self.ecc_input**2)**2 * 
                 (-67/8 + 35/8 * nu - 1/24 * nu**2 + 
                  self.ecc_input**2 * (-19/4 + 21/4 * nu + 1/12 * nu**2) +
                  self.ecc_input**4 * (5/8 - 5/8 * nu - 1/24 * nu**2) +
                  (1 - self.ecc_input**2)**(3/2) * (5 - 2 * nu)) + 
                 x**2 * nu * ((np.dot(chi_s, chi_s) - np.dot(chi_a, chi_a)) - 
                             3 * ((np.dot(chi_s, L_N))**2 - (np.dot(chi_a, L_N))**2)) + 
                 (1/2 - nu) * (np.dot(chi_s, chi_s) + np.dot(chi_a, chi_a) - 
                              3 * ((np.dot(chi_s, L_N))**2 + (np.dot(chi_a, L_N))**2)) + 
                 delta * (np.dot(chi_s, chi_a) - 3 * np.dot(chi_s, L_N) * np.dot(chi_a, L_N)))
        
        # 3PN with eccentricity
        term_3 = (x**3 / (1 - self.ecc_input**2)**3 * 
                 (-835/64 + (18319/192 - 41/16 * np.pi**2) * nu - 
                  169/32 * nu**2 - 35/5184 * nu**3 +
                  self.ecc_input**2 * (-3703/64 + (21235/192 - 41/64 * np.pi**2) * nu - 
                                      7733/288 * nu**2 + 35/1728 * nu**3) +
                  self.ecc_input**4 * (103/64 - 547/192 * nu - 1355/288 * nu**2 - 35/1728 * nu**3) +
                  self.ecc_input**6 * (185/192 + 75/64 * nu + 25/288 * nu**2 + 35/5184 * nu**3) +
                  np.sqrt(1 - self.ecc_input**2) * 
                  (5/2 + (-641/18 + 41/96 * np.pi**2) * nu + 11/3 * nu**2 + 
                   self.ecc_input**2 * (-35 + (394/9 - 41/96 * np.pi**2) * nu - 1/3 * nu**2) +
                   self.ecc_input**4 * (5/2 + 23/6 * nu - 10/3 * nu**2))))

        # 4PN circular non-spinning
        term_4 = ((-3969/128 + (-123671/5760 + 9037/1536 * np.pi**2 + 
                   896/15 * gamma_E + 448/15 * np.log(16 * x)) * nu +
                  (-498449/3456 + 3157/576 * np.pi**2) * nu**2 +
                  301/1728 * nu**3 + 77/31104 * nu**4) * x**4)
        
        return 0.5 * nu * x * (1 + term_1 + term_15 + term_2 + term_25 + term_3 + term_4)
    
    def _E0_from_PN_nonspinning(self):
        """
        Compute initial energy for non-spinning circular binaries.
        
        Uses 4PN post-Newtonian expressions for non-spinning circular binaries.
        This is a simpler version of _E0_from_PN() without spin or eccentricity.
        
        References:
        - Eq. (2.35) of https://arxiv.org/pdf/1111.5378.pdf
        - Eq. (3) of https://arxiv.org/pdf/2304.11185.pdf
        
        Returns:
            [float]: Initial orbital energy in units of total mass M.
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        
        # Euler's constant
        gamma_E = 0.57721566490153286060
        
        term_1 = (-3/4 - nu/12) * x
        term_2 = (-27/8 + 19*nu/8 - nu**2/24) * x**2
        term_3 = ((-675/64 + (34445/576 - 205*np.pi**2/96)*nu - 
                  144*nu**2/96 - 35*nu**3/5184) * x**3)
        term_4 = ((-3969/128 + (-123671/5760 + 9037/1536 * np.pi**2 + 
                   896/15 * gamma_E + 448/15 * np.log(16 * x)) * nu +
                  (-498449/3456 + 3157/576 * np.pi**2) * nu**2 +
                  301/1728 * nu**3 + 77/31104 * nu**4) * x**4)
        
        return 0.5 * nu * x * (1 + term_1 + term_2 + term_3 + term_4)
    
    def _E0_from_PN_nonspinning_eccentric(self):
        """
        Compute initial energy for non-spinning eccentric binaries.
        
        Uses 3PN post-Newtonian expressions with eccentricity corrections for
        non-spinning binaries.
        
        References:
        - Eq. (6.1a) and (6.5a) of https://arxiv.org/pdf/0908.3854.pdf
        
        Returns:
            [float]: Initial orbital energy in units of total mass M.
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        
        term1 = (x / (1 - self.ecc_input**2) * 
                (-3/4 - nu/12 + self.ecc_input**2 * (-5/4 + nu/12)))
        
        term2 = (x**2 / (1 - self.ecc_input**2)**2 * 
                (-67/8 + 35/8 * nu - 1/24 * nu**2 +
                 self.ecc_input**2 * (-19/4 + 21/4 * nu + 1/12 * nu**2) +
                 self.ecc_input**4 * (5/8 - 5/8 * nu - 1/24 * nu**2) +
                 (1 - self.ecc_input**2)**(3/2) * (5 - 2 * nu)))
        
        term3 = (x**3 / (1 - self.ecc_input**2)**3 * 
                (-835/64 + (18319/192 - 41/16 * np.pi**2) * nu - 
                 169/32 * nu**2 - 35/5184 * nu**3 +
                 self.ecc_input**2 * (-3703/64 + (21235/192 - 41/64 * np.pi**2) * nu - 
                                     7733/288 * nu**2 + 35/1728 * nu**3) +
                 self.ecc_input**4 * (103/64 - 547/192 * nu - 1355/288 * nu**2 - 35/1728 * nu**3) +
                 self.ecc_input**6 * (185/192 + 75/64 * nu + 25/288 * nu**2 + 35/5184 * nu**3) +
                 np.sqrt(1 - self.ecc_input**2) * 
                 (5/2 + (-641/18 + 41/96 * np.pi**2) * nu + 11/3 * nu**2 + 
                  self.ecc_input**2 * (-35 + (394/9 - 41/96 * np.pi**2) * nu - 1/3 * nu**2) +
                  self.ecc_input**4 * (5/2 + 23/6 * nu - 10/3 * nu**2))))

        return 0.5 * nu * x * (1 + term1 + term2 + term3)
    
    def _E0_from_PN_spinning_(self):
        """
        Compute initial energy for spinning circular binaries.
        
        Uses 3PN post-Newtonian expressions with spin corrections for circular
        binaries. Includes spin-orbit and spin-spin coupling terms.
        
        Note: Method name has trailing underscore, suggesting it may be deprecated
        or under development. Consider using _E0_from_PN() instead.
        
        Returns:
            [float]: Initial orbital energy in units of total mass M.
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        delta = gwtools.q_to_delta(self.qinput) 
        
        chi_s = (self.chi1 + self.chi2) / 2
        chi_a = (self.chi1 - self.chi2) / 2 
        L_N = np.array([0, 0, 1]) 
                
        term_1 = (-3/4 - nu/12) * x
        term_15 = ((8/3 - 4/3 * nu) * np.dot(chi_s, L_N) + 
                   (8/3) * delta * np.dot(chi_a, L_N)) * x**(1.5)
        term_2 = ((-27/8 + 19*nu/8 - nu**2/24) * x**2 + 
                 x**2 * nu * ((np.dot(chi_s, chi_s) - np.dot(chi_a, chi_a)) - 
                             3 * ((np.dot(chi_s, L_N))**2 - (np.dot(chi_a, L_N))**2)) + 
                 (1/2 - nu) * (np.dot(chi_s, chi_s) + np.dot(chi_a, chi_a) - 
                              3 * ((np.dot(chi_s, L_N))**2 + (np.dot(chi_a, L_N))**2)) + 
                 delta * (np.dot(chi_s, chi_a) - 3 * np.dot(chi_s, L_N) * np.dot(chi_a, L_N)))
        term_25 = ((8 - 121/9 * nu + 2/9 * nu**2) * np.dot(chi_s, L_N) + 
                   (8 - 31/9 * nu) * delta * np.dot(chi_a, L_N)) * x**2
        term_3 = ((-675/64 + (34445/576 - 205*np.pi**2/96)*nu - 
                  144*nu**2/96 - 35*nu**3/5184) * x**3)
        
        return 0.5 * nu * x * (1 + term_1 + term_15 + term_2 + term_25 + term_3)
    
    def _L0_from_PN_nonspinning(self):
        """
        Compute initial orbital angular momentum for non-spinning binaries.
        
        Uses 3PN post-Newtonian expressions for the orbital angular momentum
        of non-spinning circular binaries.
        
        Reference:
        - Eq. (2.36) of https://arxiv.org/pdf/1111.5378.pdf
        
        Returns:
            [float]: Initial orbital angular momentum magnitude in units of M^2.
        """
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        
        term_1 = (3/2 + nu/6) * x
        term_2 = (27/8 - 19*nu/8 + nu**2/24) * x**2
        term_3 = ((135/16 + (-6889/144 + 41*np.pi**2/24)*nu + 
                  31*nu**2/24 + 7*nu**3/1296) * x**3)
        
        return (nu / x**0.5) * (1 + term_1 + term_2 + term_3)