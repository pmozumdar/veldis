"""
 veldis.py
 
This file contains the 'Veldis' class to facilitate velocity 
dispersion calculations.
"""
#---------------------------------------------------------------------

import ppxf.ppxf_util as util
import numpy as np
import matplotlib.pyplot as plt
import glob
#import pandas as pd
#import seaborn as sn

from scipy.constants import c
from ppxf.ppxf import ppxf
from specim.specfuncs import spec1d
#from random import sample
#from collections import Counter
#from keckcode.deimos import deimosmask1d


class Veldis(spec1d.Spec1d):
    """
    A class to facilitate velocity dispersion calculations. This
    class inherits from superclass 'Spec1d' for reading various
    file types and important plotting properties. 
    """
    
    def __init__(self, inspec=None, informat='text', trimsec=None):
        
        """
        Initialize an object by reading the provided 1d spectra
        
        Describe the input arguments here....
        """
        
        super().__init__(inspec=inspec, informat=informat, 
                                                  trimsec=trimsec)
        self.wav = self['wav']
        self.flux = self['flux']
        
        if 'var' in self.columns:
            self.var = self['var']    
        else:
            print("\n Error: velocity dispersion cann't be "\
                  "calculated without 'var' data.")
            
#-----------------------------------------------------------------------
  
    def trimspec(self, wavrange=None, doplot=True):
        """
        This function trims the spectra if asked given a wavelength
        range
        """
        if wavrange is None:
            print("\nneed to provide a valid wavelength range.")
            
        else:            
            """find the indices of the closest wavelength values to
               the given 'wavrange' from the wavelength vector."""
            
            wmin = abs(self.wav - wavrange[0])
            wmax = abs(self.wav - wavrange[1])
            
            wmin_list = wmin.tolist()
            wmax_list = wmax.tolist()

            start = wmin_list.index(min(wmin))
            stop = wmax_list.index(min(wmax))
            
            """trim the spectra data for the given range"""
            
            self.wav = self.wav[start:stop+1]
            self.flux = self.flux[start:stop+1]
            
            if self.var is not None:
                self.var = self.var[start:stop+1]
                
            print("\nspectra has been trimed, now...")
            print("\nwav_min : %f" %self.wav[0])
            print("\nwav_max : %f" %self.wav[-1])
            
            if doplot:
                """Create a new spec1d object with the trimmed data"""
                
                self.trim_spec = spec1d.Spec1d(wav=self.wav, flux=self.flux,
                                          var=self.var)
                print("\n")
                self.trim_spec.plot()
                
        
#-----------------------------------------------------------------------

    def velocity_scale(self, wav_gal=None, verbose=True):
    '''
    This function calculates and returns the associated velocity scale 
    of the galaxy of interest.
    
    Parameters
    ---------------
    wav_gal: array (optional)
        An array containing the wavelengths of the galaxy spectra. 
    
    Returns
    -------------
    vel_scale: float
        Velocity scale of the galaxy.
    '''
    
    """speed of light in km/s"""
    c = c / 10**3       
    
    """ Constant wav fraction per pixel """
    if wav_gal is None:
        frac_wav = self.wav[1] / self.wav[0]   
    else:
        frac_wav = wav_gal[1] / wav_gal[0] 
        
    """velocity scale in km/s per pixel """
    vel_scale =  np.log(frac_wav) * c    
    
    if verbose:
        print('Velocity scale = %f km/s' %vel_scale)
    
    return vel_scale

#-----------------------------------------------------------------------

    