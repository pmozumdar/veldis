"""
gausssian_fit.py

This file contains the 'Gaussfit' class to fit sky lines of the 
spectra to Gaussian profiles in order to calculate sigma of the 
instrument's LSF(line of sight).
"""
#-----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from specim.specfuncs import spec1d


class Gaussfit(object):
    """
    A class to fit sky lines of the spectra to Gaussian profiles in 
    order to calculate sigma of the instrument's LSF(line of sight).
    """
    
    def __init__(self, inspec=None, informat='text', trimsec=None):
        
        """
        Initialize an object by reading the provided 1d spectra
        
        Describe the input arguments here....
        """
        self.spec = spec1d.Spec1d(inspec, informat=informat, 
                                                     trimsec=trimsec)
         
        self.wav = self.spec['wav']
        self.flux = self.spec['flux']
        
        if 'sky' in self.spec.columns:
            self.sky = self.spec['sky']
            print("\nsky model is avaiable in the spectra.")
            
        elif 'var' in self.spec.columns:
            self.sky = np.sqrt(self.spec['var'])
            print("\nsky data is not available, so variance is being " \
                  "used as sky model.")
        else:
            print("\nError: Calculating sigma of instrument's LSF is " \
                  "not possible as there is no 'sky' or 'var' data " \
                  "included in the provided spectra.")
        
#-----------------------------------------------------------------------      
     
    def plot(self, color='g', linestyle='-', title='Sky Spectrum', 
             xlabel='Wavelength (Angstroms)', drawstyle='steps'):
        """
        This function will plot the sky model for the spectra
        """
        
        plt.plot(self.wav, self.sky, color=color, ls=linestyle, 
                                                     ds=drawstyle)
        plt.xlabel(xlabel)
        plt.ylabel('Relative flux')
        plt.title(title)
        plt.xlim(self.wav[0], self.wav[-1])
                   
#----------------------------------------------------------------------- 



























