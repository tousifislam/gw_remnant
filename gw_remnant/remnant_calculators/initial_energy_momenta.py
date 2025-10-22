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
        
        # read waveforms
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
        
        # Spin vectors : primary black hole
        if spin1_input is None:
            self.spin1_input = np.array([0.0, 0.0, 0.0])
        else:
            self.spin1_input = np.array(spin1_input)
        
        # Spin vectors : secondary black hole
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
        - Eq. (2.35) of https://arxiv.org/pdf/1111.5378.pdf (circular non-spinning 4PN, 5PN)
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
                   (8 - 31/9 * nu) * delta * np.dot(chi_a, L_N)) * x**2.5
        
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
        term_4 = (-3969/128 + (-123671/5760 + 9037/1536 * np.pi**2 + 
                   896/15 * gamma_E + 448/15 * np.log(16 * x)) * nu +
                  (-498449/3456 + 3157/576 * np.pi**2) * nu**2 +
                  301/1728 * nu**3 + 77/31104 * nu**4) * x**4

        # Pseudo-PN term (5PN)
        # Table II of https://arxiv.org/pdf/1111.5378.pdf
        gamma5 = -37.72
        # Eq. (4.24b) of https://arxiv.org/pdf/1111.5378.pdf
        e5 = 3*gamma5 + 9359293/161280
        # Eq. (2.35) of https://arxiv.org/pdf/1111.5378.pdf (5PN structure)
        term_5 = ((45927/512 + nu*e5 + (4988/35 - 656*nu/5)*nu*np.log(x)) * x**5)
        
        return 0.5 * nu * x * (1 + term_1 + term_15 + term_2 + term_25 + term_3 + term_4 + term_5)
    
    def _E0_from_PN_nonspinning(self):
        """
        Compute initial energy for non-spinning circular binaries.
        
        Uses post-Newtonian expressions up to 5PN order for non-spinning circular
        binaries. Includes 4PN standard PN terms and 5PN pseudo-PN terms calibrated
        to numerical relativity simulations.
        
        The calculation combines:
        1. Standard PN expansion (1PN-4PN) from analytical calculations
        2. Pseudo-PN term (5PN) calibrated to NR simulations
        
        References:
        - Eq. (2.35) of https://arxiv.org/pdf/1111.5378.pdf (PN framework)
        - Eq. (3) of https://arxiv.org/pdf/2304.11185.pdf (4PN terms)
        - Table II of https://arxiv.org/pdf/1111.5378.pdf (pseudo-PN parameters)
        - Eq. (4.24b) of https://arxiv.org/pdf/1111.5378.pdf (energy coefficient)
        
        Returns:
            [float]: Initial orbital energy in units of total mass M.
        """

        # Different choices for initial 
        # Obtain PN expansion parameter x = (M*omega)^(2/3)
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        
        # Euler's constant
        gamma_E = 0.57721566490153286060
        
        # Standard PN terms (1PN-4PN)
        # Eq. (3) of https://arxiv.org/pdf/2304.11185.pdf
        term_1 = (-3/4 - nu/12) * x
        term_2 = (-27/8 + 19*nu/8 - nu**2/24) * x**2
        term_3 = ((-675/64 + (34445/576 - 205*np.pi**2/96)*nu - 
                  144*nu**2/96 - 35*nu**3/5184) * x**3)
        term_4 = ((-3969/128 + (-123671/5760 + 9037/1536 * np.pi**2 + 
                   896/15 * gamma_E + 448/15 * np.log(16 * x)) * nu +
                  (-498449/3456 + 3157/576 * np.pi**2) * nu**2 +
                  301/1728 * nu**3 + 77/31104 * nu**4) * x**4)
        
        # Pseudo-PN term (5PN)
        # Table II of https://arxiv.org/pdf/1111.5378.pdf
        gamma5 = -37.72
        # Eq. (4.24b) of https://arxiv.org/pdf/1111.5378.pdf
        e5 = 3*gamma5 + 9359293/161280
        # Eq. (2.35) of https://arxiv.org/pdf/1111.5378.pdf (5PN structure)
        term_5 = ((45927/512 + nu*e5 + (4988/35 - 656*nu/5)*nu*np.log(x)) * x**5)
        
        # FIXED: Include term_5 in the return!
        return 0.5 * nu * x * (1 + term_1 + term_2 + term_3 + term_4 + term_5)

    
    def _E0_from_PN_nonspinning_eccentric(self):
        """
        Compute initial energy for non-spinning eccentric binaries.
        
        Uses 3PN post-Newtonian expressions with eccentricity corrections for
        non-spinning binaries. The calculation includes terms up to e^6 in the
        eccentricity expansion at each PN order.
        
        The energy is computed in terms of the dimensionless binding energy
        ε = -E/(μc²), where μ is the reduced mass. Each PN order includes
        corrections for both the PN parameter x and eccentricity e.
        
        References:
        - Eq. (6.1a) of https://arxiv.org/pdf/0908.3854.pdf (energy definition)
        - Eq. (6.5a) of https://arxiv.org/pdf/0908.3854.pdf (3PN eccentric expansion)
        
        Returns:
            [float]: Initial orbital energy in units of total mass M.
        """
        # Obtain PN expansion parameter x = (M*omega)^(2/3)
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        
        # 1PN term with eccentricity corrections up to e^2
        # Eq. (6.5a) of https://arxiv.org/pdf/0908.3854.pdf
        term1 = (x / (1 - self.ecc_input**2) * 
                (-3/4 - nu/12 + self.ecc_input**2 * (-5/4 + nu/12)))
        
        # 2PN term with eccentricity corrections up to e^4
        # Includes tidal deformation term (1 - e²)^(3/2)
        term2 = (x**2 / (1 - self.ecc_input**2)**2 * 
                (-67/8 + 35/8 * nu - 1/24 * nu**2 +
                 self.ecc_input**2 * (-19/4 + 21/4 * nu + 1/12 * nu**2) +
                 self.ecc_input**4 * (5/8 - 5/8 * nu - 1/24 * nu**2) +
                 (1 - self.ecc_input**2)**(3/2) * (5 - 2 * nu)))
        
        # 3PN term with eccentricity corrections up to e^6
        # Most complex term with multiple eccentricity-dependent contributions
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
        
        # Dimensionless binding energy parameter ε
        # Eq. (6.1a) of https://arxiv.org/pdf/0908.3854.pdf
        epsilon = x * (1 + term1 + term2 + term3)
        
        # Convert to energy E = -μc²ε = -(1/2)νMc²ε
        # Eq. (6.5a) of https://arxiv.org/pdf/0908.3854.pdf
        E = 0.5 * nu * epsilon
        
        return E

    
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

        # symmetric spin
        chi_s = (self.chi1 + self.chi2) / 2
        # anti symmetric spin
        chi_a = (self.chi1 - self.chi2) / 2 
        # assume total angular momentum direction to be along the z axis
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
        
        Uses post-Newtonian expressions up to 5PN order for the orbital angular
        momentum of non-spinning circular binaries. Includes 4PN and 5PN terms
        calibrated to numerical relativity through pseudo-PN parameters.
        
        The calculation proceeds in two steps:
        1. Compute standard PN expansion coefficients (1PN-3PN)
        2. Add pseudo-PN terms (4PN-5PN) calibrated to NR simulations
        
        References:
        Alexandre Le Tiec, Luc Blanchet, and Bernard F. Whiting
        - Eq. (2.36) of https://arxiv.org/pdf/1111.5378.pdf (standard PN terms)
        - Table II of https://arxiv.org/pdf/1111.5378.pdf (pseudo-PN parameters)
        - Eqs. (4.24a, 4.24b) of https://arxiv.org/pdf/1111.5378.pdf (energy coefficients)
        - Eqs. (2.49a, 2.49b) of https://arxiv.org/pdf/1111.5378.pdf (angular momentum coefficients)
        
        Returns:
            [float]: Initial orbital angular momentum magnitude in units of M^2.
        """
        # Obtain PN expansion parameter x = (M*omega)^(2/3)
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        
        # Standard PN terms (1PN-3PN)
        # Eq. (2.36) of https://arxiv.org/pdf/1111.5378.pdf
        term_1 = (3/2 + nu/6) * x
        term_2 = (27/8 - 19*nu/8 + nu**2/24) * x**2
        term_3 = ((135/16 + (-6889/144 + 41*np.pi**2/24)*nu + 
                  31*nu**2/24 + 7*nu**3/1296) * x**3)
        
        # Pseudo-PN terms (4PN-5PN) calibrated to NR
        # Table II of https://arxiv.org/pdf/1111.5378.pdf
        gamma4 = 53.432
        gamma5 = -37.72
        
        # Convert energy coefficients to angular momentum coefficients
        # Eqs. (4.24a, 4.24b) of https://arxiv.org/pdf/1111.5378.pdf
        e4 = 7*gamma4/3 + 28037/960
        e5 = 3*gamma5 + 9359293/161280
        
        # Angular momentum coefficients at 4PN and 5PN
        # Eqs. (2.49a, 2.49b) of https://arxiv.org/pdf/1111.5378.pdf
        j4 = -(5/7)*e4 + 64/35
        j5 = -(2/3)*e5 - 4988/945 - 656*nu/135
        
        # 4PN and 5PN terms
        term_4 = (2835/128 + nu*j4) * x**4
        term_5 = ((15309/256 + nu*j5 + (9976/105 + 1312*nu/15)*nu*np.log(x)) * x**5)
        
        return (nu / x**0.5) * (1 + term_1 + term_2 + term_3 + term_4 + term_5)

    
    def _L0_from_PN_nonspinning_eccentric(self):
        """
        Compute initial orbital angular momentum for non-spinning eccentric binaries.
        
        Uses 3PN post-Newtonian expressions with eccentricity corrections for
        non-spinning eccentric binaries. The calculation includes terms up to e^6
        in the eccentricity expansion at each PN order.
        
        The angular momentum is computed as j = L/(μ√(GMa)), where μ is the reduced
        mass, M is the total mass, and a is the semi-major axis. Each PN order includes
        corrections for both the PN parameter x and eccentricity e_t.
        
        References:
        - Eq. (6.1b) of https://arxiv.org/pdf/0908.3854.pdf (angular momentum definition)
        - Eq. (6.5b) of https://arxiv.org/pdf/0908.3854.pdf (3PN eccentric expansion)
        
        Returns:
            [float]: Initial orbital angular momentum magnitude in units of M^2.
        """
        # Obtain PN expansion parameter x = (M*omega)^(2/3)
        orb_phase = 0.5 * gwtools.phase(self.hdict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) / 
                           np.gradient(self.time, edge_order=2))
        x = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.qinput)
        
        # Use v = nu for symmetric mass ratio (as commonly done in PN expansions)
        v = nu
        e_t = self.ecc_input
        
        # Overall factor (1 - e_t²)
        prefactor = (1 - e_t**2)
        
        # Newtonian term
        term_0 = 1
        
        # 1PN term with eccentricity corrections up to e^2
        term_1 = (x / (1 - e_t**2) * 
                 (9/4 + 1/4 * v + 
                  e_t**2 * (-17/4 + 7/4 * v)))
        
        # 2PN term with eccentricity corrections up to e^4
        # Includes tidal deformation term √(1 - e_t²)
        term_2 = (x**2 / (1 - e_t**2)**2 * 
                 (27/8 - 19/8 * v + 1/24 * v**2 +
                  e_t**2 * (-1/4 + 53/12 * v - 5/4 * v**2) +
                  e_t**4 * (75/8 - 277/24 * v + 29/24 * v**2) +
                  np.sqrt(1 - e_t**2) * e_t**2 * (-15 + 6 * v)))
        
        # 3PN term with eccentricity corrections up to e^6
        # Most complex term with multiple eccentricity-dependent contributions
        term_3 = (x**3 / (1 - e_t**2)**3 * 
                 (-747/64 + (-5797/192 + 41/32 * np.pi**2) * v + 167/96 * v**2 - 1/192 * v**3 +
                  e_t**2 * (5281/64 + (-8955/64 + 39/32 * np.pi**2) * v + 1751/96 * v**2 + 67/192 * v**3) +
                  e_t**4 * (1023/64 - 3701/64 * v + 947/32 * v**2 - 131/192 * v**3) +
                  e_t**6 * (-757/64 + 7381/192 * v - 1207/96 * v**2 + 65/192 * v**3) +
                  np.sqrt(1 - e_t**2) * (45/4 - 13/4 * v - 1/2 * v**2 +
                                        e_t**2 * (-505/4 + (2227/12 - 41/32 * np.pi**2) * v - 47/2 * v**2) +
                                        e_t**4 * (55 - 40 * v + 3 * v**2))))
        
        # Dimensionless angular momentum parameter j
        j = prefactor * (term_0 + term_1 + term_2 + term_3)
        
        # Convert to angular momentum L = j * ν * M² / √x
        # Using the relation between j and L from PN theory
        E0 = self._E0_from_PN_nonspinning_eccentric()
        
        return np.sqrt(nu * j / (2*E0))