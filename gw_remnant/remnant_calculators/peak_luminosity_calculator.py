import numpy as np
from .remnant_mass_calculator import RemnantMassCalculator


class PeakLuminosityCalculator(RemnantMassCalculator):
    """
    Class to compute the peak luminosity of the GW radiation using a spline
    fit to the data
    """
    def __init__(self, time, hdict, qinput, M_initial=1, use_filter=False):
        
        super().__init__(time, hdict, qinput, M_initial, use_filter)
        self.L_peak = self._compute_peak_luminosity()
        
    def _get_peak_via_quadratic_fit(self, t, func):
        """
        Finds the peak time of a function quadratically
        Fits the function to a quadratic over the 5 points closest to the argmax func.
        t : an array of times
        func : array of function values
        Returns: tpeak, fpeak
        """
        # Find the time closest to the peak, making sure we have room on either side
        index = np.argmax(func)
        index = max(2, min(len(t) - 3, index))
        # Do a quadratic fit to 5 points,
        # subtracting t[index] to make the matrix inversion nice
        testTimes = t[index-2:index+3] - t[index]
        testFuncs = func[index-2:index+3]
        xVecs = np.array([np.ones(5),testTimes,testTimes**2.])
        invMat = np.linalg.inv(np.array([[v1.dot(v2) for v1 in xVecs] \
            for v2 in xVecs]))
        yVec = np.array([testFuncs.dot(v1) for v1 in xVecs])
        coefs = np.array([yVec.dot(v1) for v1 in invMat])
        return t[index] - coefs[1]/(2.*coefs[2]), coefs[0] - coefs[1]**2./4/coefs[2]
    
    def _compute_peak_luminosity(self):
        """
        computes the peak luminosity;
        Eq(1) of https://arxiv.org/pdf/2010.00120.pdf
        """
        L_peak = self._get_peak_via_quadratic_fit(self.time, self.E_dot)[1]
        return L_peak
        

