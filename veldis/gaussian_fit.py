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

    def closest_wavrange(self, wavrange=None, verbose=True):
        """
        This function extracts the closest wavelength range values 
        from the wavelength vector to a given crude waverange range.
        """
        
        clst_wav_range = []
        wav_index = []

        if wavrange is None:
            print("\nneed to provide a list of wavelength ranges.")

        else:
            for i, p in enumerate(wavrange):

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

                clst_wav_range.append((start_val, stop_val))
                wav_index.append((start_index, stop_index))
                
            """Print the given and closest waverange if requested """
            
            if verbose:
                print("\nGiven waverange(assumed) : \n")
                [print(*wvrange) for wvrange in wavrange]
                print("\nClosest waverange to the given ones : \n")
                [print(*wvrange) for wvrange in clst_wav_range]

        return  clst_wav_range, wav_index
       
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
            
            clst_wav_range, wav_index = self.closest_wavrange(trim_range, 
                                                          verbose=False)
            
            """trim the spectra data for the given range"""
            
            xmin = wav_index[0][0]
            xmax = wav_index[0][1]
            
            self.wav = self.wav[xmin:xmax+1]
            self.flux = self.flux[xmin:xmax+1]
            self.sky = self.sky[xmin:xmax+1]
         
            print("\nspectra has been trimed, now...")
            print("\nwav_min : %f" %self.wav[0])
            print("\nwav_max : %f" %self.wav[-1])
        
#-----------------------------------------------------------------------

    def collect_skydata(self, wavrange=None, wavindex=None, 
                        verbose=True, doplot=True):
        """
        This function collects the sky data for given range of 
        wavelengths to fit with Gaussian distribution.
        """
        
        """empty list to contain data"""
        
        flux_sky_line = []           
        wav_sky_line = [] 
        
        """First collect the indices of the given wavelength range"""
         
        clst_wav_range, wav_index = self.closest_wavrange(wavrange, 
                                                        verbose=True)
        """Collect data """
        
        for i, p in enumerate(wav_index):
            
            sk = self.sky[p[0]:(p[1]+1)] 
            wv = self.wav[p[0]:(p[1]+1)]
            
            flux_sky_line.append(sk)
            wav_sky_line.append(wv)
            
        """Plot a bar over the skylines going to fit if requested"""
        
        if doplot:
            
            self.plot_skyline(wav_sky_line)
            
        return flux_sky_line, wav_sky_line

#-----------------------------------------------------------------------

    def plot_skyline(self, wav_sky_line=None, color='y', title=None, 
                     label='sky lines', width=10.0):
        
        """
        A function to plot bars over the skylines going to fit with a
        Gaussian.
        """
        if wav_sky_line is None:
            print("\n Error: Need to provide wavelengths of the skylines")
            
        else:
            if title is None:
                title = 'Sky lines to fit'
                
            cen = np.zeros(len(wav_sky_line))
            wd = np.zeros(len(wav_sky_line))
            
            for i, p in enumerate(wav_sky_line):
                
                cen[i]= np.median(p)
                if len(p) >= width:
                    wd[i] = len(p)
                else:
                    wd[i] = width 
                    
            plt.figure()
            plt.plot(self.wav, self.sky)
            plt.bar(cen, height=max(self.sky), width=wd, color=color, 
                    label=label)
            plt.xlabel('Wavelength')
            plt.ylabel('Relative Flux')
            plt.legend()
            plt.title(title)
            
#-----------------------------------------------------------------------            
    
    def fit_gauss(self, wavrange=None, flux_sky_line=None, 
                  wav_sky_line=None):
        """
        This function fits each sky line to a Gaussian profile. Before 
        performing the fit each flux array has been normalized with the 
        median value of the array. Both the 'flux' and 'wav' data 
        have also been shifted. The 'flux'data has been shifted by 
        subtracting the minimum value of the data array so that the tails
        touch the x axis. And the 'wav' data has been shifted by the
        median value of the 'wav' array.
        """
        
        """Collect data if a wavrange is provided"""
        if wavrange is not None:
            self.flux_sky_line, \
                      self.wav_sky_line =  self.collect_skydata(wavrange)
            
        model_gauss = models.Gaussian1D()        
        fitter_gauss = fitting.LevMarLSQFitter()
            
        """Fit each skyline to Gaussian profile and plot"""
        best_fit = []
        
        for f, w in zip(self.flux_sky_line, self.wav_sky_line):
            
            f = f / np.median(f)
            x = w - np.median(w)
            y = f - np.min(f)
            
            best_fit_gauss = fitter_gauss(model_gauss, x, y)
            best_fit.append(best_fit_gauss)
            
            print(best_fit_gauss)
            
            plt.figure()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(w, f)
            ax1.set_title('Original sky data')
            ax1.set_xlabel('Wavelength')
            ax1.set_ylabel('Relative flux')


            ax2.plot(x, y, label='shifted sky data') 
            ax2.plot(x, best_fit_gauss(x), 'r', label='Gaussian fit')
            ax2.set_title('Fit to Gaussian')
            ax2.set_xlabel('shifted wavelength')
            ax2.set_ylabel('shifted relative flux')
            plt.legend()
            plt.show()
            
        return  best_fit
    
#-----------------------------------------------------------------------            

    def sigma_inst(self, best_fit=None, doplot=True, ylim=None):
        """
        Collect the 'stddev' of the fitted skylines
        """
         
        if best_fit is None:
            print("\nError: need to provide a list of best fit "\
                  "model, none given.")
        else:
            sig = []
            for i, p in enumerate(best_fit):
                sig.append(p.stddev.value)

            avg_sig = np.sum(sig) / len(sig)
            print('average : %f' %avg_sig)
            self.sig = sig

        if doplot:
            
            wav = np.zeros(len(self.wav_sky_line))
            fwhm = np.zeros(len(self.wav_sky_line))
            
            for i, p in enumerate(self.wav_sky_line):
                wav[i] = np.median(p)
                fwhm[i] = sig[i] * 2.355
            self.fwhm = fwhm  
            
            plt.figure()
            ax1 = plt.subplot(211)
            plt.plot(wav, sig, '.', ms=10)
            #ax1.set_xticklabels([])
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.ylabel('Sigma')
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            plt.subplots_adjust(hspace=0.001)
            
            ax2 = plt.subplot(212, sharex=ax1)
            plt.plot(wav, fwhm, '.', ms=10)
            plt.xlabel('median wavelength of selected sky lines')
            plt.ylabel('FWHM')
            plt.setp(ax2.get_xticklabels(), visible=True)
            if ylim is not None:
                plt.ylim(ylim[0]*2.355, ylim[1]*2.355)
                
#-----------------------------------------------------------------------            

                
                















