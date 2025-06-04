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
    
    def __init__(self, inspec=None, informat='text', trimsec=None,
                 wav=None, flux=None, var=None, sky=None, 
                 logwav=False):
        
        """
        Initialize an object by reading the provided 1d spectra
        
        Describe the input arguments here....
        """
        self.spec = spec1d.Spec1d(inspec=inspec, informat=informat, 
                                  trimsec=trimsec, wav=wav, flux=flux,
                                  var=var, sky=sky, logwav=logwav)
         
        self.wav = self.spec['wav']
        self.flux = self.spec['flux']
        
        if 'sky' in self.spec.columns:
            sky = self.spec['sky']
            """check whether 'sky' column is not just zeros"""
            if np.max(sky)==0 and np.min(sky)==0:
                print('\nsky data only contains zeros')
                if 'var' in self.spec.columns:
                    self.sky = self.spec['var'] ## variance not noise as Tommaso suggested
                    print("\nvariance is being used as sky model.")
                else:
                    print("\nError: Calculating sigma of instrument's" \
                          "LSF is not possible as there is no 'sky' or" \
                          "'var' data included in the provided spectra.")
            else:
                self.sky = self.spec['sky']
                print("\nsky model is avaiable in the spectrum.")
            
        elif 'var' in self.spec.columns:
            self.sky = self.spec['var'] ## variance not noise as Tommaso suggested
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
        from the wavelength vector to a given crude wavelength range.
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
        This function trims the spectra if asked.
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
         
            print("\nspectra has been trimmed, now...")
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
                try:
                    if len(p) >= width:
                        wd[i] = len(p)
                    else:
                        wd[i] = width 
                except:
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
    
    def fit_gauss(self, wavrange=None, flux_sky_line=None, wav_sky_line=None, 
                  up_sigma_threshold=10.0, low_sigma_threshold=0.3, mean_pos_threshold=2.0, 
                  gof_threshold=5.0, chi2_threshold=1.0, only_good_fit=False,
                 norm_flux=False):
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
            
        model_gauss = models.Gaussian1D() + models.Polynomial1D(degree=0)     
        fitter_gauss = fitting.LevMarLSQFitter()
            
        """Fit each skyline to Gaussian profile and plot"""
        best_fit, bad_fit, good_fit_skyline = [], [], []

        for f, w in zip(self.flux_sky_line, self.wav_sky_line):
            if norm_flux:
                y = f / np.median(f)
            else:
                y = f
            x = w - np.median(w)
            # y = f #- np.min(f)
            
            best_fit_gauss = fitter_gauss(model_gauss, x, y)
            residuals = y - best_fit_gauss(x)
            #residual_sig_est = np.sqrt(np.sum(residuals**2) /(len(x)-4))
            reduced_chi2 = np.sum((residuals/np.std(y))**2) /(len(x)-4)
            rms_residual =  np.std(residuals)
            #print(f'rms residual {rms_residual}')
            #print(residuals)
            bad_fit_mask = np.abs(residuals)>gof_threshold*rms_residual
            
            print(f'reduced chi2 : {reduced_chi2}')
            if np.sum(bad_fit_mask) > 1:
                print(f'residual higher than {gof_threshold} times rms_residual')
            
            ### Reject bad fits
            if best_fit_gauss.stddev_0.value > up_sigma_threshold or best_fit_gauss.stddev_0.value < low_sigma_threshold: # Reject very broad Gaussians
                bad_fit.append(best_fit_gauss)
            elif best_fit_gauss.amplitude_0.value < 0 :  # Reject negative amplitudes
                bad_fit.append(best_fit_gauss)
            elif np.abs(best_fit_gauss.mean_0.value - 0) > mean_pos_threshold : # Mean too far from expected
                bad_fit.append(best_fit_gauss)
            elif np.sum(bad_fit_mask) > 1: # bad fit atleast at one pixel
                bad_fit.append(best_fit_gauss)
            elif reduced_chi2 > chi2_threshold: # High chi-square
                bad_fit.append(best_fit_gauss)
            else:
                best_fit.append(best_fit_gauss)
                good_fit_skyline.append(np.round(np.median(w), 2))
                if only_good_fit:
                    print(best_fit_gauss)
                    median_w = np.round(np.median(w), 2)
                    print(f'median wav : {median_w}')
                    
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
                
            if not only_good_fit:
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
            
        return  best_fit, good_fit_skyline
    
#-----------------------------------------------------------------------            

    def sigma_inst(self, best_fit, good_fit_skyline, wavrange_to_fit=None,
                   doplot=True, exclude_wav=[], ylim_km=None, ylim_ang=None,
                  plot_axline=None):
        """
        Collect the 'stddev' of the fitted skylines.
        """

        c = 299792.458 # km/s
        
        if best_fit is None:
            print("\nError: need to provide a list of best fit "\
                  "model, none given.")
        else:
            sig_ang = []
            included_wav = []
            
            for i, (p, w) in enumerate(zip(best_fit, good_fit_skyline)):
                if w not in exclude_wav:
                    sig_ang.append(p.stddev_0.value)
                    included_wav.append(w)
                    
            sig_ang, included_wav = np.array(sig_ang), np.array(included_wav)
            fwhm_ang = sig_ang * 2.355
            sig_km = (sig_ang/included_wav)*c
            fwhm_km = sig_km *2.355

            avg_sig_ang = np.sum(sig_ang) / len(sig_ang)
            avg_sig_km = np.sum(sig_km) / len(sig_km)
            print('average sigma (Ang): %f' %avg_sig_ang)
            print('average FWHM (Ang): %f' %(avg_sig_ang*2.355))
            print('average sigma (km): %f' %avg_sig_km)
            print('average FWHM (km): %f' %(avg_sig_km*2.355))
            self.sig_ang, self.sig_km = sig_ang, sig_km
            self.fwhm_ang, self.fwhm_km = fwhm_ang, fwhm_km
            self.final_fitted_wav = included_wav
            
        if doplot:
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            plt.subplots_adjust(hspace=0.02)
            
            ax[0].plot(included_wav, sig_km, '.', ms=10)
            ax[0].set_ylabel('Sigma (km/s)')
            ax[0].set_xticks([])
            ax[0].axhline(y=avg_sig_km, color='black', linestyle='-')
            ax[0].axhline(y=avg_sig_km*1.1, color='black', linestyle='--')
            ax[0].axhline(y=avg_sig_km*0.9, color='black', linestyle='--')
            if plot_axline is not None:
                ax[0].axhline(y=plot_axline, color='red', linestyle='-')
            if ylim_km is not None:
                ax[0].set_ylim(ylim_km[0], ylim_km[1])
            
            ax[1].plot(included_wav, sig_ang, '.', ms=10)
            ax[1].set_ylabel('Sigma (Ang)')
            ax[1].axhline(y=avg_sig_ang, color='black', linestyle='-')
            ax[1].axhline(y=avg_sig_ang*1.1, color='black', linestyle='--')
            ax[1].axhline(y=avg_sig_ang*0.9, color='black', linestyle='--')
            if ylim_ang is not None:
                ax[1].set_ylim(ylim_ang[0], ylim_ang[1])
            ax[1].set_xlabel('median wavelength of selected sky lines')
            
                
#----------------------------------------------------------------------- 

## code not in use right now....

# to enable functions to test gaussianity need to import

# from scipy.stats import normaltest, shapiro

    def normtest(x):
        """
        Performs 'shapiro' and 'normality' test.
        """
    
        stat, ps = shapiro(x)
        if len(x) > 7 :
            stat, p = normaltest(x)
        else:
            p = 0

        return ps, p

    def chi2(exp, obs):
        """
        calculate chi2
        """
    
        obs = obs / max(obs)
        exp = exp / max(exp)
        ch = np.sum((obs-exp)**2 / exp)

        return ch


# here we tried to get the sky lines automatically to exclude
# manual collection

#sky = []
#sky_in =[]

#i = 0
#while i < len(variance_spectra_galaxy[600:1000]):
    
#    if variance_spectra_galaxy[i] < 0.07:
#        i +=1
#    else:
#        j = i
#        start_in = j
#        base = variance_spectra_galaxy[j+2]
        
#        while j<len(variance_spectra_galaxy[600:1000]):
            
#            if j <=(start_in +2) and variance_spectra_galaxy[j] > 0.07:
#                j +=1
            
#            elif j > (start_in +2) and variance_spectra_galaxy[j] > base:
#                j +=1
                
#            else:
#                stop_in = j
#                break
        
#        i = j
        
#        if ( stop_in - start_in ) >= 3 :
#            brd = int(np.ceil(( stop_in - start_in )*0.5) )

#            if (stop_in + brd) < len(variance_spectra_galaxy):
#                data = variance_spectra_galaxy[(start_in-brd):(stop_in + brd)]
#                wav = lamda_galaxy[(start_in-brd):(stop_in + brd)]
#            else:
#                data = variance_spectra_galaxy[start_in:stop_in]
#                wav  = lamda_galaxy [start_in:stop_in]

#            sky.append((wav, data))
#            sky_in.append((start_in, stop_in))
        
    
