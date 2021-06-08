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

from scipy.constants import c, pi
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
    
    def __init__(self, inspec=None, informat='text', trimsec=None,
                 wav=None, flux=None, var=None, sky=None, logwav=False):
        
        """
        Initialize an object by reading the provided 1d spectra
        
        Describe the input arguments here....
        """
        
        super().__init__(inspec=inspec, informat=informat, 
                         trimsec=trimsec,  wav=wav, flux=flux,
                         var=var, sky=sky, logwav=logwav)
        
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
            self.var = self.var[start:stop+1]
                
            print("\nspectra has been trimmed, now...")
            print("\nwav_min : %f" %self.wav[0])
            print("\nwav_max : %f" %self.wav[-1])
            
            if doplot:
                """Create a new spec1d object with the trimmed data"""
                
                self.trim_spec = spec1d.Spec1d(wav=self.wav, flux=self.flux,
                                          var=self.var)
                print("\n")
                self.trim_spec.plot()
                
#-----------------------------------------------------------------------

    def cal_parm(self, z=None, doplot=True, logscale=True,
                 high_z=False, noise_scale=0.05, veldis_start=200.0):
        """
        This function will calculate some required parameters for
        velocity dispersion calculation like logarithimically
        rebinned galaxy spectra (both flux and wavelength) and noise,
        and velocity scale. Besides initial guess for velocity
        dispersion fit parameters to be estimated. If the
        spectra are already in log scale (like echelle spectra) than
        there is no need to log_rebin, only velocity scale is calculated
        explicitly. However, if the spectra are in linear scale then
        'log_rebin' function will be used to logarithmically rebin
        the galaxy spectra which will also calculate velocity scale.
        This velocity scale (either calculate explicity or using the
        function) will be used to logarithmically rebinn template
        spectra (also noise spectra if it is in linear scale).
        
        """
        
        """
        If the galaxy is at significant redshift, one should bring
        the galaxy spectrum roughly to the rest-frame wavelength,
        before calling pPXF (See Sec.2.4 of Cappellari 2017). In
        practice there is no need to modify the spectrum in any way,
        given that a red shift corresponds to a linear shift of the
        log-rebinned spectrum. One just needs to compute the wavelength
        range in the rest-frame and adjust the instrumental resolution
        of the galaxy observations.
        
        """
        if z is None:
            print("\nError : redshift is required")
            
        if high_z :
            self.wav = self.wav/(1.0 + z)
            self.z = z
            self.high_z = high_z
            
        if logscale:
            self.v = self.velocity_scale()
            self.flux_rebinned = self.flux / np.median(self.flux)
            self.wav_rebinned = np.log(self.wav)
            noise = np.sqrt(self.var)
            self.noise_rebinned = noise_scale*(noise/np.median(noise))
        else:
            wav_range = np.array([self.wav[0], self.wav[-1]])
            flux = self.flux #/ np.median(self.flux)
            self.flux_rebinned, self.wav_rebinned, self.v = util.log_rebin(
                                                      wav_range, flux)
            self.flux_rebinned = self.flux_rebinned / np.median(self.flux_rebinned)
            noise = np.sqrt(self.var)
            self.noise_rebinned = util.log_rebin(wav_range, noise,
                                                      velscale=self.v)[0]
            self.noise_rebinned = noise_scale*(self.noise_rebinned /
                                                   np.median(self.noise_rebinned))
        
        """Initial guess for velocity and velocity dispersion. Using
           eq.(8) of Cappellari(2017). 'vel' is in km/s"""
        
        if high_z:
            """z=0 as everything already redshifted to rest frame """
            vel = c / 10**3 #* np.log(1.0 + z)
        else:
            vel = (c / 10**3) * np.log(1.0 + z)
            
        self.start = [vel, veldis_start]
            
        """Plot logarithmically rebinned galaxy spectra and noise
           if requested"""    
        if doplot:
            plt.figure()
            plt.plot(self.wav_rebinned, self.flux_rebinned)
            plt.title('logarithmically rebinned galaxy spectra')
            plt.show()
            
            plt.figure()
            plt.plot(self.wav_rebinned, self.noise_rebinned)
            plt.title('logarithmically rebinned noise')
            plt.show()
        
        """a return statement isn't required anymore as 'flux_rebinned',
           'noise_rebinned' and ' start' are accessible as instance
           attribute."""
        #return flux_rebinned, noise_rebinned, start
    
#-----------------------------------------------------------------------

    def gen_sigma_diff(self, wav_temp=None, sig_ins=None,
                       doplot=False, verbose=True, fwhm_temp=None,
                       wav_disp=0.4):
        """
        This function calculates and returns the differences in sigma
        per wavelength between the two instrumental LSF's, used to
        collect galaxy spectrum and template spectra. This function
        also calculates a required parameter 'dv' for velocity
        dispersion measuremnt. The parameter 'dv' accounts for the
        difference of the initial wavelength in the galaxy and template
        spectra.
        
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
               
        wav_disp: float
           average dispersion in wavelength.
           
        Returns
        -------------
        sigma_diff: array 
            An array containing the differences in sigma per wavelength.

        """ 
        try:
            if self.high_z:
                sig_ins = sig_ins / (1+self.z)
            fwhm_galaxy = 2.355 * sig_ins
        except:
            print("\nError : need to provide sigma of the instrument's"\
                  " LSF through the input argument 'sig_ins'.")
            
        if fwhm_temp is None:
            print("\nAs no \'fwhm_temp\' value is provided, FWHM for "\
                  "the Indo-US template library will be used as "\
                  "default value")
            fwhm_temp = 1.35    # FWHM for indo-us template library                        
        else:
            fwhm_temp = fwhm_temp
        
        """Create an array interpolating FWHM of galaxy at the place of
           template wavelengths if FWHM galaxy varies with wavelength."""
        
        if isinstance(fwhm_galaxy, np.ndarray):
            fwhm_interp = np.interp(wav_temp, self.wav, fwhm_galaxy)
            fwhm_diff = np.sqrt(fwhm_interp**2 - fwhm_temp**2)
        else:
            fwhm_diff = np.sqrt(fwhm_galaxy**2 - fwhm_temp**2)
        
        """Calculate difference in sigma"""
        
        sigma_diff = (fwhm_diff / 2.355 )/ wav_disp
        
        """Calculating 'vsyst' in km/s which requires wavelength 
           info of a template spectra."""
        
        self.vsyst = (c / 10**3) * np.log(wav_temp[0] / self.wav[0])       
        if verbose:
            print('vsyst = %f ' %self.vsyst)
            print("\nsigma_diff : %f" %sigma_diff)
        
        """Plot the sigma difference value per wavelength if requested"""
        if doplot:
            plt.figure()
            plt.plot(wav_temp, sigma_diff,'.', label='sigma_diff')
            plt.legend()
            plt.show()

        return sigma_diff
    
#-----------------------------------------------------------------------

    def gen_rebinned_temp(self, lib_path=None, temp_array=None,  
                          informat='text', temp_num=None, sig_ins=None,
                          rand_temp=False, fwhm_temp=None, doplot=True, 
                          verbose=True, high_z=False, wav_disp=0.4,
                         velscale_ratio=1.0): 
        """
        This function generates and returns an array containing 
        logarithmically rebinned template spectra.

        Parameters
        ---------------
        lib_path: string
            path to the directory containing template library.

        temp_num: int
            Number of templates that would be logarithmically rebinned.
            If given that amount of template spectra would be fetched 
            from library, rebinned and stored in the array.

        temp_array: array
            An array containing template file names which would be 
            logarithmically rebinned. If given only those template 
            spectra would be fetched from library, rebinned and stored 
            in the array.

        sig_ins: single float or array of floats
            sigma value of the instrumental LSF used to collect galaxy
            spectra. One can provide the average value of sigma over the
            wavelength range or provide sigma per wavelength.

        fwhm_temp: float
            FWHM value of the template spectra.

        Returns
        -------------
        temp_spec: array
            An array containging all the logarithmically rebinned and 
            normalized template spectra.

        """
        
        """Container to store the the convolved, logarithmically 
           rebinned and normalized template spectra."""

        temp_spec = []
        
        """Colleting the template files to be used """
        
        if lib_path is None:
            if temp_array is None:
                print("\nError : need to provide a string containing the "\
                      "path to the template library or an array containing "\
                      "filenames and paths of the templates, none given.")
            else:
                if temp_num is None:
                    templates =  temp_array
                else:
                    templates = temp_array[:temp_num]
                    
        else:
            if temp_num is None:
                templates =  glob.glob(lib_path)
            else:
                templates = glob.glob(lib_path)[:temp_num]

        """Collect the sigma difference data to convolve the template 
           spectra """

        wav_temp = spec1d.Spec1d(templates[0], informat=informat,
                                                    verbose=False)['wav']
        
        sigma_diff = self.gen_sigma_diff(wav_temp=wav_temp, sig_ins=sig_ins,
                                         fwhm_temp=fwhm_temp, high_z=high_z,
                                        wav_disp=wav_disp)
        
        wav_range = [wav_temp[0], wav_temp[-1]]
        
        """Logarithmically rebin the template spectra.The array containing
           this data should have a shape of [nPixels, nTemplates]"""
        
        for i, file in enumerate(templates):

            temp_data = spec1d.Spec1d(file, informat=informat, verbose=False)
            temp_flux = temp_data['flux']
  
            #convolved_temp = util.gaussian_filter1d(temp_flux, sigma_diff)  

            #temp_rebinned = util.log_rebin(wav_range, convolved_temp, 
            #                                              velscale=self.v)[0]
            temp_rebinned = util.log_rebin(wav_range, temp_flux, 
                                               velscale=self.v/velscale_ratio)[0]
            
            nor_temp = temp_rebinned / np.median(temp_rebinned)  

            temp_spec.append(nor_temp)
            
        #temp_spec = np.array(temp_spec).T
        temp_spec = np.swapaxes(np.array(temp_spec), 0, 1)

        return temp_spec

#-----------------------------------------------------------------------
    
    def masking(self, pixel_range=None, wav_rebinned=None):
        """
        This function generate and returns a boolean array with value 'False'
        in the pixel locations which should be excluded from the fit. The
        size of the array is equal to the size of the logarithmically
        rebinned galaxy flux or wavelength.

        Parameters
        ---------------   
        pixel_range: list
            A list of tuples where each tuple contains start and end values 
            of the pixel range needs to be excluded. The values should be
            from logarithmically rebinned galaxy wavelength.

        wav_rebinned: array
            This array contains the values of the logarithmically 
            rebinned wavelengths.

        Returns
        -------------
        mask : boolean array
            Boolean array with value 'False' in the pixel locations 
            which should be excluded from the fit.

        """
        if wav_rebinned is None:
            wav_rebinned = self.wav_rebinned
        else:
            wav_rebinned = wav_rebinned
        
        """Create a boolean array of the length of rebinned wavelength"""
        
        mask = np.zeros(len(wav_rebinned), dtype=bool)
        for i, p in enumerate(pixel_range):
            mask |= (wav_rebinned>=p[0]) & (wav_rebinned <= p[1])
            
        return (~mask)

#-----------------------------------------------------------------------

    def check_temp_coverage(self, intemp=None, informat='text',
                            z=None):
        """
        This function will check the range of wavelength the template
        spectra will cover given a particular redshift.
        """
        
        if intemp is None:
            print("\n Error : need to provide a string containing "\
                  "path to a template file")
        else:
            wav_temp = spec1d.Spec1d(inspec=intemp, informat=informat,
                                                  verbose=False)['wav']
        
        if z is None:
            print("\n Error : need to provide redshfit")
                  
        else:
            wav_min = wav_temp[0] * (1+z)
            wav_max = wav_temp[-1] * (1+z)

            print("\nCovered range for redshift %f : "\
                  "%f - %f" %(z, wav_min, wav_max))
                  
#-----------------------------------------------------------------------

    def cal_redshift(self, **kwargs):
        """
        This function will calculate redshift comparing given observed
        wavelength with actual emitted wavelength. Currently compares
        CaII K+H, G-band, MgI b and NaI D.
        """
              
        wav_dict = {'CaK' : 3933.67, 'CaH' : 3968.47, 'G' : 4305.0,
                    'Mgb' : 5176, 'NaD' : 5895.92}
        
        for key, val in kwargs.items():
            if key in wav_dict.keys():   
                z = (val / wav_dict[key]) - 1.0
                print("\n Observed wavelength for %s : %f" %(key, val))
                print("\nredshift z : %f" %z)

#-----------------------------------------------------------------------

    def cal_veldis(self, temp_spec=None, lib_path=None, temp_array=None,
                   informat='text', temp_num=None, sig_ins=None,
                   rand_temp=False, fwhm_temp=None, doplot=True,
                   verbose=True, moments=4, plot=True, degree=None, 
                   mask_reg=None, quiet=False, show_weight=False,
                   clean=False):
        """
        This function calculates velocity dispersion using 'ppxf'
        method.
        """
        """First Setup some parameters """
        if temp_spec is None:
            self.temp_spec = self.gen_rebinned_temp(lib_path=lib_path, 
                        temp_array=temp_array, informat=informat, 
                        temp_num=temp_num, sig_ins=sig_ins,
                        rand_temp=rand_temp, fwhm_temp=fwhm_temp, 
                        doplot=doplot, verbose=verbose)
        else:
            self.temp_spec = temp_spec
            
        if mask_reg is not None:
            self.mask_region = self.masking(pixel_range=mask_reg)

        if degree is None:
            deg = np.arange(4, 6)
        else:
            deg = np.arange(degree[0], degree[1])
        
        """Setup the containers to store velocity dispersion and error 
           values """
        vel_dis = np.zeros(len(deg)) 
        error = np.zeros(len(deg))
        best_fit = []
        """good pixels are the pixels which have been used in the fit"""
        good_pixels = []    
        
        """Do the velocity dispersion calculation """
        for i, d in enumerate(deg):
            print('\ndegree : %d' %d)
            if mask_reg is None:
                pp = ppxf(self.temp_spec, self.flux_rebinned, 
                          self.noise_rebinned, self.v, self.start, 
                          moments=moments, plot=plot, vsyst=self.vsyst, 
                          degree=d, quiet=quiet, clean=clean,
                          lam=np.exp(self.wav_rebinned))
            else:
                pp = ppxf(self.temp_spec, self.flux_rebinned, 
                          self.noise_rebinned, self.v, self.start, 
                          moments=moments, plot=plot, vsyst=self.vsyst, 
                          degree=d, mask=self.mask_region, quiet=quiet, 
                          lam=np.exp(self.wav_rebinned), clean=clean)

            vel_dis[i] = pp.sol[1]
            error[i] = pp.error[1]
            best_fit.append(pp.bestfit)
            good_pixels.append(pp.goodpixels)
            if plot:
                plt.figure()

            if show_weight:
                [print('%d, %f'%(i,w)) for i,w in enumerate(pp.weights)\
                                                               if w>10]
        self.vel_dis = vel_dis
        self.error = error
        self.deg = deg
        self.best_fit = best_fit
        self.goodpixels = good_pixels

#----------------------------------------------------------------------------

    def plot_veldis(self, xlim=None, ylim=None, xlabel='degree', 
                    ylabel='velocity dispersion'):
        """
        This function plots velocity dispersion over degree
        """
        plt.figure()
        plt.plot(self.deg, self.vel_dis, 'b.', ms=10)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if xlim is None:
            pass
        else:
            plt.xlim(xlim[0], xlim[1])

        if ylim is None:
            pass
        else:
            plt.ylim(ylim[0], ylim[1])

#----------------------------------------------------------------------------

    def plot_error(self, xlim=None, ylim=None, xlabel='degree',
                   ylabel='error'):
        """
        This function plots error over degree
        """
        plt.figure()
        plt.plot(self.deg, self.error, 'r.', ms=10)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if xlim is None:
            pass
        else:
            plt.xlim(xlim[0], xlim[1])

        if ylim is None:
            pass
        else:
            plt.ylim(ylim[0], ylim[1])

#---------------------------------------------------------------------------

#----------------------------------------------------------------------------

#    def velocity_scale(self, wav_gal=None, verbose=True):
#        """
#        This function calculates and returns the associated velocity scale 
#        of the galaxy of interest.
#
#        Parameters
#        ---------------
#        wav_gal: array (optional)
#            An array containing the wavelengths of the galaxy spectra. 
#
#        Returns
#        -------------
#        vel_scale: float
#            Velocity scale of the galaxy.
#        """
#        
#        """ Constant wav fraction per pixel """
#        if wav_gal is None:
#            frac_wav = self.wav[1] / self.wav[0]   
#        else:
#            frac_wav = wav_gal[1] / wav_gal[0] 
#
#        """velocity scale in km/s per pixel """
#        vel_scale =  np.log(frac_wav) * (c / 10**3)
#
#        if verbose:
#            print('Velocity scale = %f km/s' %vel_scale)
#
#        return vel_scale

