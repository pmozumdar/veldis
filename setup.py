import sys
import setuptools

setuptools.setup(
    name = 'veldis',
    version = '0.1',
    description = 'Code for calculating velocity dispersion' 
                   'for galaxies using galaxy spectra.',
    author = 'Pritom Mozumdar',
    author_email = 'pmozumdar@ucdavis.edu',
    url = 'https://github.com/pmozumdar/veldis.git',
    classifiers = [
        'License :: OSI Approved :: MIT License',
        #'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        ],
    #'zip_safe': False,
    requires = ['numpy','scipy','astropy','matplotlib', 
                  'cdfutils','specim', 'ppxf'],
    packages = setuptools.find_packages()
)
