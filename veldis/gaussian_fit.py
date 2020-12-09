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

    def exact_wavrange(self, skylines=None):
        """
        This function extracts the exact wavelength range values 
        from the wavelength vector given a crude waverange range.
        """
        
        exact_wav_range = []
        wav_index = []

        if skylines is None:
            print("\nneed to provide a list of wavelength ranges.")

        else:
            for i, p in enumerate(skylines):

                wmin = abs(self.wav - p[0])
                wmax = abs(self.wav - p[1])

                """Converting the above arrays into list so that
                   element value would be collectible given index"""

                wmin_list = wmin.tolist()
                wmax_list = wmax.tolist()

                start_index = wmin_list.index(min(wmin))
                stop_index = wmax_list.index(min(wmax))

                start_val = self.wav[start_index]
                stop_val = self.wav[stop_index]

                exact_wav_range.append((start_val, stop_val))
                wav_index.append((start_index, stop_index))
            
        return  exact_wav_range, wav_index
       
#-----------------------------------------------------------------------
  
    def trimspec(self, wavrange=None):
        """
        This function trims the spectra if asked
        """
        if wavrange is None:
            print("\nneed to provide a valid wavelength range.")
            
        else:
            trim_range = [(wavrange[0], wavrange[1])]
            
            """find the indices of the closest wavelength values to the 
               given 'wavrange' from the wavelength vector."""
            
            exact_wav_range, wav_index = self.exact_wavrange(trim_range)
            
            """trim the spectra data for the given range"""
            
            xmin = wav_index[0][0]
            xmax = wav_index[0][1]
            
            self.wav = self.wav[xmin:xmax+1]
            self.flux = self.flux[xmin:xmax+1]
            self.sky = self.sky[xmin:xmax+1]
         
            print("\nspectra has been trimed, now...")
            print("\nwav_min : %f" %self.wav[xmin])
            print("\nwav_max : %f" %self.wav[xmax])
        
#-----------------------------------------------------------------------

























