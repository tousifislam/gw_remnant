#############################################################################
##
##      Filename: gw_remnant_calculator.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: Estimates energy and momenta profiles of a BBH merger and 
##                   calculates the properties of the final black hole in a 
##                   BBH merger
##
##      Modified:
##
#############################################################################

from .remnant_calculators.peak_luminosity_calculator import PeakLuminosityCalculator
from .remnant_calculators.kick_velocity_calculator import LinearMomentumCalculator
from .remnant_calculators.remnant_mass_calculator import RemnantMassCalculator
from .remnant_calculators.remnant_spin_calculator import AngularMomentumCalculator
from .gw_utils.gw_plotter import GWPlotter

class GWRemnantCalculator(GWPlotter, PeakLuminosityCalculator, AngularMomentumCalculator,
                          LinearMomentumCalculator, RemnantMassCalculator):
    """
    Class to compute the following remnant quantities from a given waveform;
        (i) h_dot : derivative of the input waveform in dictionary format
        (ii) E_dot : enrgy flux as a function of time
        (iii) E_rad : total energy radiated
        (iv) Eoft : radiated energy as a function of time
        (v) remnant_mass : mass of the remnant black hole
        (vi) P_dot :derivate of the linear momentum vector as a function of time
        (vii) Poft : linear momentum vector as a function of time
        (viii) voft : velocity vector of the center of mass a function of time
        (ix) kickoft : imparted kick valocity magnitude as a function of time
        (x) remnant_kick : final kick velocity of the remnant black hole 
        (xi) J_dot : derivative of the angluar momentum vector as a function of time
        (xii) Joft : angular momentum vector as a function of time
        (xiii) spinoft : spin as a function of time
        (xiv) remnant_spin : spin of the remnant black hole
        (xv) L_peak : peak luminosity of the binary black hole merger
        (xvi) peak_kick : peak kick velocity
        
    It also provides methods to print quantities of interests and to plot time evolution
    of mass, energy, momentum and kick;
    """
    def __init__(self, time, hdict, qinput, M_initial=1, use_filter=False):
        super().__init__(time, hdict, qinput, M_initial, use_filter)
        
    def print_remnants(self):
        """
        prints interesting remnant quantities
        """
        print("Mass ratio : %.3f"%self.qinput)
        print("Initial mass : %.8f M"%self.M_initial)
        print("Total enery radiated : %.8f M"%self.E_rad)
        print("Peak luminosity : %.8f "%self.L_peak)
        print("Remnant mass : %.8f M"%self.remnant_mass)
        print("Remnant spin (dimensionless) : %.8f M"%self.remnant_spin)
        print("Remnant kick velocity : %.8f c"%self.remnant_kick)