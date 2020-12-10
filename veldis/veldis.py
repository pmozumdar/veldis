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
        """
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
        """
        
        """ Constant wav fraction per pixel """
        if wav_gal is None:
            frac_wav = self.wav[1] / self.wav[0]   
        else:
            frac_wav = wav_gal[1] / wav_gal[0] 

        """velocity scale in km/s per pixel """
        vel_scale =  np.log(frac_wav) * (c / 10**3)   

        if verbose:
            print('Velocity scale = %f km/s' %vel_scale)

        return vel_scale

#-----------------------------------------------------------------------
    #we may not need this function
    def gen_dv(self, wav_gal=None, wav_temp=None, verbose=True):
        """
        This function calculates the parameter 'dv' to account for the 
        difference of the initial wavelength in the galaxy and template 
        spectra.

        Parameters
        ---------------
        wav_gal: float (optional)
            Starting wavelength of the galaxy spectra.

        wav_temp: float (optional)
            Starting wavelength of a template spectra in the library.

        Returns
        -------------
        dv: float
            The parameter to account for the initial wavelegth difference.
        """

        c = 299792.458               # speed of light in km/s
        wav_temp = 3465.00         # starting wavelength of the templates
                                     # in the Indo-US library.
        dv = c*np.log(wav_temp / wav_gal) 
        print('dv = %f ' %dv)

        return dv
#-----------------------------------------------------------------------

    def gen_sigma_diff(self, wav_temp=None, sig_ins=None, fwhm_temp=None,
                       doplot=True, verbose=True):
        """
        This function calculates and returns the differences in sigma 
        per wavelength between the two instrumental LSF's, used to 
        collect galaxy spectrum and template spectra.

        Parameters
        ---------------
        sig_ins: single float or array of floats
            sigma value of the instrumental LSF used to collect galaxy
            spectra. One can provide the average value of sigma over the
            wavelength range or provide sigma per wavelength.

        fwhm_temp: float
            FWHM value of the template spectra.

        wav_temp: array
           An array containing the wavelengths of the template spectra.

        Returns
        -------------
        sigma_diff: array
            An array containing the differences in sigma per wavelength.

        """
        
        """Create an array of FWHM per wavelength for the galaxy of 
           interest."""
        
        if sig_ins is None:
            print("\nError : need to provide sigma of the instrument's"\
                  " LSF through the input argument 'sig_ins'.")
        else:
            if isinstance(sig_ins, array):
                fwhm_galaxy = 2.355 * sig_ins
            else:
                fwhm_galaxy = 2.355 * sig_ins
                fwhm_galaxy = np.full(len(self.wav), fwhm_galaxy)

        if fwhm_temp is None:
            print("\nAs no \'fwhm_temp\' value is provided, FWHM for "\
                  "the Indo-US template library will be used as "\
                  "default value")
            
            """ FWHMfor indo-us template library """
            fwhm_temp = 1.35                            
        else:
            fwhm_temp = fwhm_temp

        """Create an array interpolating FWHM of galaxy at the place of
           template wavelengths."""
        
        fwhm_interp = np.interp(wav_temp, self.wav, fwhm_galaxy)
        
        """Calculate difference in sigma"""
        
        fwhm_diff = np.sqrt(fwhm_interp**2 - fwhm_temp**2)
        sigma_diff = fwhm_diff / 2.355
        
        """Plot the sigma difference value per wavelength if requested"""
        if doplot:
            plt.figure()
            plt.plot(wav_temp, sigma_diff,'.', label='sigma_diff')
            plt.legend()
            plt.show()

        return sigma_diff
    
#-----------------------------------------------------------------------

   