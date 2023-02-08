#############################################################################
##
##      Filename: gw_waveform_generator.py
##
##      Author: Tousif Islam
##
##      Created: 01-05-2023
##
##      Description: generates waveforms from different models and NRSurRemnant model
##
##      Modified:
##
#############################################################################

from .gw_utils.waveform_generator import WaveformGenerator

class GWWaveformGenerator(WaveformGenerator):
    """
    Class to generate gravitational waveforms for different models;
    At this moment, the class supports three models:
        (i) BHPTNRSur1dq1e4;
        (ii) NRHybSur3dq8;
        (iii) SEOBNRv4HM;
    It further computes the remnant quantities using the following remnant
    surrogates:
        (i) NRHybSur3dq8Remnant;
    """
    def __init__(self, mass_ratio, modes=None, common_times=None, f_low=None, 
                get_NRSur=True, get_BHPT=True):
        super().__init__(mass_ratio, modes, common_times, f_low, 
                         get_NRSur, get_BHPT)