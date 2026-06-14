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
#   Computes the initial orbital binding energy and angular momentum of a binary
#   black hole system using post-Newtonian (PN) expressions.
#
#   The default expressions are taken from the SEOBNRv5EHM orbit-averaged,
#   PN-expanded energy and angular momentum (valid for eccentric orbits and
#   aligned spins to 3PN), supplemented by the analytic 4PN and NR-calibrated 5PN
#   non-spinning circular terms:
#
#     - A. Gamboa, M. Khalil, A. Buonanno, "Third post-Newtonian dynamics for
#       eccentric orbits and aligned spins in the effective-one-body waveform
#       model SEOBNRv5EHM", arXiv:2412.12823. (Orbit-averaged binding energy
#       ``Heob`` and angular momentum ``Leob``, supplementary file
#       EOB_Keplerian.dat.m.)
#     - M. Khalil et al., "Theoretical groundwork ... SEOBNRv5", arXiv:2303.18143.
#       (Analytic 4PN non-spinning circular energy/angular momentum, ``ES0``/``LS0``.)
#     - A. Le Tiec, L. Blanchet, B. F. Whiting, arXiv:1111.5378.
#       (NR-calibrated pseudo-5PN coefficients, Table II and Eqs. (4.24), (2.49).)
#
#====================================================================================

from __future__ import annotations

import numpy as np
import gwtools


class InitialEnergyMomenta:
    """
    Calculator for the initial orbital energy and angular momentum of a binary.

    Computes the initial orbital binding energy and orbital angular momentum of a
    binary black hole using post-Newtonian (PN) approximations. The defaults are
    valid for:
      - non-spinning circular binaries to 5PN (analytic 4PN + NR-calibrated 5PN);
      - aligned-spin binaries to 3PN (spin-orbit and spin-spin);
      - eccentric binaries to 3PN (including spin x eccentricity cross terms).

    Initial conditions can either be computed from the PN expressions above or be
    supplied directly by the user (e.g. read from numerical-relativity metadata).

    Sign conventions (geometric units, G = c = 1):
      - ``E_initial`` is the POSITIVE binding-energy magnitude, E_initial = -E_bind,
        so that the ADM mass is M_ADM = M_initial - E_initial. To use an NR value
        pass ``E_initial = M_initial - M_ADM`` (= 1 - initial_ADM_energy for unit
        total mass).
      - ``L_initial`` is the orbital angular momentum magnitude (in units of M^2).

    Args:
        time (np.ndarray): Array of time values in geometric units (M)
        h_dict (dict): Dictionary of complex waveform modes with (l,m) tuple keys,
            e.g., {(2,2): h_22(t), (3,3): h_33(t), ...}
        q (float): Mass ratio q = m1/m2, where m1 >= m2
        chi1 (list or np.ndarray): Spin vector [sx, sy, sz] for the primary
            black hole at the reference time, dimensionless. Only the aligned
            (z) component enters the PN energy/angular momentum. Default None.
        chi2 (list or np.ndarray): Spin vector [sx, sy, sz] for the secondary
            black hole at the reference time, dimensionless. Default None.
        e_ref (float): Eccentricity at the reference time. Default None (circular).
        E_initial (float): Initial binding-energy magnitude in units of total mass M.
            If None, computed from the PN expressions. Set to 0 to track energy
            changes relative to the reference. Default None.
        L_initial (float): Initial orbital angular momentum in units of M^2. If None,
            computed from the PN expressions. Set to 0 to track angular momentum
            changes relative to the reference. Default None.

    Attributes:
        time (np.ndarray): Time array
        h_dict (dict): Waveform mode dictionary
        q (float): Mass ratio
        chi1 (np.ndarray): Primary spin vector
        chi2 (np.ndarray): Secondary spin vector
        e_ref (float): Eccentricity
        E_initial (float): Initial orbital binding-energy magnitude
        L_initial (float): Initial orbital angular momentum
    """

    def __init__(self, time: np.ndarray, h_dict: dict[tuple[int, int], np.ndarray],
                 q: float, chi1: np.ndarray | list[float] | None = None,
                 chi2: np.ndarray | list[float] | None = None,
                 e_ref: float | None = None, E_initial: float | None = None,
                 L_initial: float | None = None) -> None:

        # Input validation
        self._validate_inputs(time, h_dict, q, chi1, chi2, e_ref)

        # read waveforms
        self.time = time
        self.h_dict = h_dict
        # Ensure negative m modes are present using symmetry relation
        for mode in list(self.h_dict.keys()):
            if mode[1] != 0 and (mode[0], -mode[1]) not in self.h_dict:
                self.h_dict[(mode[0], -mode[1])] = (-1)**mode[0] * np.conjugate(self.h_dict[mode])

        # Mass ratio
        self.q = q

        # Spin vectors : primary black hole
        if chi1 is None:
            self.chi1 = np.array([0.0, 0.0, 0.0])
        else:
            self.chi1 = np.array(chi1)

        # Spin vectors : secondary black hole
        if chi2 is None:
            self.chi2 = np.array([0.0, 0.0, 0.0])
        else:
            self.chi2 = np.array(chi2)

        # Eccentricity
        if e_ref is None:
            self.e_ref = 0.0
        else:
            self.e_ref = e_ref

        # Cache PN expansion parameter and symmetric mass ratio (used by all PN methods)
        self._x0, self._nu = self._compute_initial_pn_parameter()

        # Initial energy
        if E_initial is None:
            self.E_initial = self._E0_from_PN()
        else:
            self.E_initial = E_initial

        # Initial angular momentum
        if L_initial is None:
            self.L_initial = self._L0_from_PN()
        else:
            self.L_initial = L_initial

    def _compute_initial_pn_parameter(self):
        """Compute the PN expansion parameter x = (M*omega)^(2/3) and symmetric mass
        ratio nu from the initial orbital frequency of the (2,2) mode.

        Returns:
            tuple: (x0, nu) where x0 is the PN parameter at the start of the waveform
                and nu is the symmetric mass ratio.
        """
        orb_phase = 0.5 * gwtools.phase(self.h_dict[(2, 2)])
        orb_frequency = abs(np.gradient(orb_phase, edge_order=2) /
                           np.gradient(self.time, edge_order=2))
        x0 = orb_frequency[0]**(2/3)
        nu = gwtools.q_to_nu(self.q)
        return x0, nu

    @staticmethod
    def _validate_inputs(time, h_dict, q, chi1, chi2, e_ref):
        """Validate user inputs and raise ValueError on invalid data."""
        # Time array
        if len(time) < 2:
            raise ValueError("Time array must have at least 2 elements.")
        dt = np.diff(time)
        if np.any(dt <= 0):
            raise ValueError("Time array must be monotonically increasing.")

        # Required (2,2) mode
        if (2, 2) not in h_dict:
            raise ValueError(
                "Waveform dictionary must contain the (2,2) mode. "
                f"Available modes: {list(h_dict.keys())}"
            )

        # Mass ratio
        if q < 1:
            raise ValueError(
                f"Mass ratio q must be >= 1 (q = m1/m2 with m1 >= m2), got q={q}."
            )

        # Spin magnitudes
        for label, spin in [("chi1", chi1), ("chi2", chi2)]:
            if spin is not None:
                spin_arr = np.asarray(spin)
                if spin_arr.shape != (3,):
                    raise ValueError(
                        f"{label} must be a 3-component vector [sx, sy, sz], "
                        f"got shape {spin_arr.shape}."
                    )
                mag = np.linalg.norm(spin_arr)
                if mag > 1.0:
                    raise ValueError(
                        f"{label} magnitude must be <= 1 for a black hole, "
                        f"got |chi| = {mag:.4f}."
                    )

        # Eccentricity
        if e_ref is not None:
            if e_ref < 0 or e_ref >= 1:
                raise ValueError(
                    f"Eccentricity must satisfy 0 <= e < 1, got e={e_ref}."
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spin_combinations(self):
        """Symmetric/antisymmetric spin and mass-difference combinations.

        Only the aligned (z) components of the spins enter the orbit-averaged PN
        energy and angular momentum.

        Returns:
            tuple: (delta, chiS, chiA) where delta = (m1-m2)/M,
                chiS = (chi1z + chi2z)/2 and chiA = (chi1z - chi2z)/2.
        """
        chi1z = self.chi1[2]
        chi2z = self.chi2[2]
        delta = gwtools.q_to_delta(self.q)
        chiS = 0.5 * (chi1z + chi2z)
        chiA = 0.5 * (chi1z - chi2z)
        return delta, chiS, chiA

    # ------------------------------------------------------------------
    # Binding energy
    # ------------------------------------------------------------------
    def _E0_from_PN(self):
        """
        Compute the initial binding-energy magnitude E_initial (positive).

        Combines the orbit-averaged 3PN energy (eccentric, aligned spin) with the
        analytic 4PN and NR-calibrated 5PN non-spinning circular terms. The return
        value is E_initial = -E_bind = -nu * Ehat, where Ehat = E_bind/mu is the
        binding energy per reduced mass (negative for a bound system), so that the
        ADM mass is M_ADM = M_initial - E_initial.

        References:
            - SEOBNRv5EHM ``Heob`` (arXiv:2412.12823) for the 3PN base.
            - arXiv:2303.18143 ``ES0`` and arXiv:1111.5378 for the 4PN/5PN terms.

        Returns:
            [float]: Initial binding-energy magnitude in units of total mass M.
        """
        Ehat = (self._binding_energy_hat_3PN()
                + self._binding_energy_hat_nonspinning_4PN5PN())
        return -self._nu * Ehat

    def _binding_energy_hat_3PN(self):
        """
        Orbit-averaged binding energy per reduced mass, Ehat = E_bind/mu, to 3PN.

        Translation of the conservative (Cos/Sin[zeta]-averaged) part of the
        SEOBNRv5EHM ``Heob`` expression for eccentric orbits and aligned spins,
        with black-hole spin-induced quadrupole constants (kappa = 1). Valid for
        non-spinning, aligned-spin, and eccentric binaries.

        See ``Heob`` (Eq. block "Energy / Hamiltonian") of arXiv:2412.12823.

        Returns:
            [float]: Binding energy per reduced mass (negative for bound systems).
        """
        x = self._x0
        nu = self._nu
        e = self.e_ref
        delta, chiS, chiA = self._spin_combinations()
        om = 1.0 - e**2

        # non-spinning: 0PN, 1PN, 2PN, 3PN
        E = -x / 2
        E += (x**2 * (9 - e**2 * (-15 + nu) + nu)) / (24 * om)
        E += (x**3 * (201 - 105 * nu + nu**2 + 24 * om**1.5 * (-5 + 2 * nu)
                      + e**4 * (-15 + 15 * nu + nu**2)
                      - 2 * e**2 * (87 + 15 * nu + nu**2))) / (48 * om**2)
        E += -(x**4 * (-67635 - 27 * (-18319 + 492 * np.pi**2) * nu - 27378 * nu**2
                       - 35 * nu**3 + 5 * e**6 * (999 + 1215 * nu + 90 * nu**2 + 7 * nu**3)
                       + 3 * e**2 * (55539 - 9 * (-1651 + 123 * np.pi**2) * nu
                                     - 6078 * nu**2 + 35 * nu**3)
                       - 3 * e**4 * (33507 - 261 * nu + 66 * nu**2 + 35 * nu**3)
                       + 18 * om**1.5 * (720 + (-10256 + 123 * np.pi**2) * nu + 1056 * nu**2
                                         + 48 * e**2 * (-105 + 43 * nu + 8 * nu**2)))) / (10368 * om**3)

        # spin-orbit: 1.5PN
        E += (2 * x**2.5 * (-2 * delta * chiA + (-2 + nu) * chiS)) / (3 * om**1.5)
        # spin-spin: 2PN
        E += (x**3 * ((1 - 4 * nu) * chiA**2 + 2 * delta * chiA * chiS + chiS**2)) / (2 * om**2)
        # spin-orbit: 2.5PN
        E += (x**3.5 * (delta * (-144 + 55 * nu + 8 * e**2 * (21 + nu)) * chiA
                        + (-144 + 217 * nu - 14 * nu**2 - 4 * e**2 * (-42 + 10 * nu + nu**2)) * chiS
                        + om**1.5 * (-24 * delta * (-3 + nu) * chiA
                                     + 12 * (6 - 8 * nu + nu**2) * chiS))) / (18 * om**2.5)
        # spin-spin: 3PN
        E += (x**4 * ((197 - 857 * nu + 132 * nu**2 + 3 * e**2 * (-117 + 461 * nu + 16 * nu**2)) * chiA**2
                      - 2 * delta * (-197 + 253 * nu + 3 * e**2 * (117 + nu)) * chiA * chiS
                      + (197 - 437 * nu + 128 * nu**2 + 3 * e**2 * (-117 + 5 * nu)) * chiS**2
                      + om**1.5 * (-12 * (11 - 46 * nu + 6 * nu**2) * chiA**2
                                   + 24 * delta * (-11 + 9 * nu) * chiA * chiS
                                   - 12 * (11 - 16 * nu + 4 * nu**2) * chiS**2))) / (36 * om**3)
        return E

    def _binding_energy_hat_nonspinning_4PN5PN(self):
        """
        Analytic 4PN and NR-calibrated 5PN non-spinning circular terms of the
        binding energy per reduced mass, beyond the 3PN base.

        These are the x^4 and x^5 corrections to Ehat = -x/2 (1 + e1 x + ...); the
        4PN coefficient is the exact analytic term (arXiv:2303.18143 ``ES0``), the
        5PN term is the NR-calibrated pseudo-PN coefficient of arXiv:1111.5378.
        Returns 0 for eccentric orbits (the 4PN/5PN eccentricity corrections are
        not implemented).

        Returns:
            [float]: 4PN+5PN non-spinning circular contribution to E_bind/mu.
        """
        if self.e_ref != 0.0:
            return 0.0
        x = self._x0
        nu = self._nu
        gamma = np.euler_gamma
        log2 = np.log(2.0)

        # 4PN (analytic), Eq. (3) of arXiv:2304.11185 / ES0 of arXiv:2303.18143
        e4 = (-3969/128
              + (-123671/5760 + 9037/1536 * np.pi**2 + 896/15 * gamma
                 + 448/15 * np.log(16 * x)) * nu
              + (-498449/3456 + 3157/576 * np.pi**2) * nu**2
              + 301/1728 * nu**3 + 77/31104 * nu**4)

        # 5PN (NR-calibrated pseudo-PN), Table II and Eq. (4.24b) of arXiv:1111.5378
        gamma5 = -37.72
        e5coef = 3 * gamma5 + 9359293/161280
        e5 = (45927/512 + nu * e5coef + (4988/35 - 656 * nu / 5) * nu * np.log(x))

        return -0.5 * x * (e4 * x**4 + e5 * x**5)

    # ------------------------------------------------------------------
    # Angular momentum
    # ------------------------------------------------------------------
    def _L0_from_PN(self):
        """
        Compute the initial orbital angular momentum L_initial.

        Combines the orbit-averaged 3PN angular momentum (eccentric, aligned spin)
        with the analytic 4PN and NR-calibrated 5PN non-spinning circular terms.
        Returns L_initial = nu * Lhat, where Lhat = L/mu is the angular momentum per
        reduced mass.

        References:
            - SEOBNRv5EHM ``Leob`` (arXiv:2412.12823) for the 3PN base.
            - arXiv:2303.18143 ``LS0`` and arXiv:1111.5378 for the 4PN/5PN terms.

        Returns:
            [float]: Initial orbital angular momentum magnitude in units of M^2.
        """
        Lhat = (self._angular_momentum_hat_3PN()
                + self._angular_momentum_hat_nonspinning_4PN5PN())
        return self._nu * Lhat

    def _angular_momentum_hat_3PN(self):
        """
        Orbit-averaged angular momentum per reduced mass, Lhat = L/mu, to 3PN.

        Translation of the conservative part of the SEOBNRv5EHM ``Leob`` expression
        for eccentric orbits and aligned spins, with black-hole spin-induced
        quadrupole constants (kappa = 1).

        See ``Leob`` (Eq. block "Angular momentum") of arXiv:2412.12823.

        Returns:
            [float]: Angular momentum per reduced mass.
        """
        x = self._x0
        nu = self._nu
        e = self.e_ref
        delta, chiS, chiA = self._spin_combinations()
        om = 1.0 - e**2
        sq = np.sqrt(om)

        # non-spinning: 0PN, 1PN, 2PN, 3PN
        L = sq / np.sqrt(x)
        L += (np.sqrt(x) * (9 - e**2 * (-9 + nu) + nu)) / (6 * sq)
        L += (x**1.5 * (141 - 81 * nu + nu**2 - 2 * e**2 * nu * (19 + nu)
                        + om**1.5 * (-60 + 24 * nu)
                        + e**4 * (-3 + 11 * nu + nu**2))) / (24 * om**1.5)
        L += (x**2.5 * (100440 + 54 * (-12604 + 369 * np.pi**2) * nu + 32400 * nu**2 + 56 * nu**3
                        + 3 * e**2 * (-7992 + 9 * (-23512 + 861 * np.pi**2) * nu + 6144 * nu**2 - 56 * nu**3)
                        + 24 * e**4 * (-1431 + 1521 * nu - 204 * nu**2 + 7 * nu**3)
                        - 8 * e**6 * (837 + 621 * nu + 72 * nu**2 + 7 * nu**3)
                        - 18 * om**1.5 * (720 + (-10256 + 123 * np.pi**2) * nu + 1056 * nu**2
                                          + 48 * e**2 * (-75 + 31 * nu + 8 * nu**2)))) / (10368 * om**2.5)

        # spin-orbit: 1.5PN
        L += -((5 + 3 * e**2) * x * (2 * delta * chiA + (2 - nu) * chiS)) / (3 * om)
        # spin-spin: 2PN
        L += (x**1.5 * (-(2 + 3 * e**2) * (-1 + 4 * nu) * chiA**2
                        + 2 * (2 + 3 * e**2) * delta * chiA * chiS
                        + (2 + 3 * e**2) * chiS**2)) / (2 * om**1.5)
        # spin-orbit: 2.5PN
        L += (x**2 * (delta * (-1584 + 626 * nu + 1063 * e**2 * nu + 3 * e**4 * (144 + 49 * nu)) * chiA
                      + (e**2 * (3133 - 410 * nu) * nu + e**4 * (432 + 201 * nu - 114 * nu**2)
                         - 2 * (792 - 1231 * nu + 62 * nu**2)) * chiS
                      + om**1.5 * (-192 * delta * (-3 + nu) * chiA
                                   + 96 * (6 - 8 * nu + nu**2) * chiS))) / (144 * om**2)
        # spin-spin: 3PN
        L += -(x**2.5 * ((27 * e**4 * (9 - 37 * nu + 4 * nu**2) - 3 * e**2 * (-67 + 237 * nu + 40 * nu**2)
                          - 4 * (59 - 260 * nu + 42 * nu**2)) * chiA**2
                         + 2 * delta * (e**4 * (243 - 99 * nu) + 3 * e**2 * (67 + 77 * nu)
                                        + 4 * (-59 + 85 * nu)) * chiA * chiS
                         + (e**2 * (201 + 369 * nu - 132 * nu**2) + 9 * e**4 * (27 - 19 * nu + 4 * nu**2)
                            - 4 * (59 - 146 * nu + 44 * nu**2)) * chiS**2
                         + om**1.5 * (12 * (11 - 46 * nu + 6 * nu**2) * chiA**2
                                      + 24 * delta * (11 - 9 * nu) * chiA * chiS
                                      + 12 * (11 - 16 * nu + 4 * nu**2) * chiS**2))) / (36 * om**2.5)
        return L

    def _angular_momentum_hat_nonspinning_4PN5PN(self):
        """
        Analytic 4PN and NR-calibrated 5PN non-spinning circular terms of the
        angular momentum per reduced mass, beyond the 3PN base.

        The 4PN coefficient is the exact analytic term (arXiv:2303.18143 ``LS0``),
        the 5PN term is the NR-calibrated pseudo-PN coefficient of arXiv:1111.5378
        (Table II and Eqs. (2.49)). Returns 0 for eccentric orbits.

        Returns:
            [float]: 4PN+5PN non-spinning circular contribution to L/mu.
        """
        if self.e_ref != 0.0:
            return 0.0
        x = self._x0
        nu = self._nu
        gamma = np.euler_gamma
        log2 = np.log(2.0)

        # 4PN (analytic), LS0 of arXiv:2303.18143
        l4 = (2835/128
              + (356035/3456 - 2255/576 * np.pi**2) * nu**2
              - 215/1728 * nu**3 - 55/31104 * nu**4
              + nu * (98869/5760 - 128/3 * gamma - 6455/1536 * np.pi**2
                      - 256/3 * log2 - 128/3 * 0.5 * np.log(x)))

        # 5PN (NR-calibrated pseudo-PN), Table II and Eqs. (4.24b), (2.49b) of arXiv:1111.5378
        gamma5 = -37.72
        e5coef = 3 * gamma5 + 9359293/161280
        j5 = -(2/3) * e5coef - 4988/945 - 656 * nu / 135
        l5 = (15309/256 + nu * j5 + (9976/105 + 1312 * nu / 15) * nu * np.log(x))

        return (1 / np.sqrt(x)) * (l4 * x**4 + l5 * x**5)
