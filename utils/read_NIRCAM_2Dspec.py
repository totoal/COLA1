from astropy.io import fits
import numpy as np

hdu = fits.open('stacked_2D_COLA1_9269.fits')
hd = hdu['EMLINE'].header
SCI_DATA = hdu['SCI'].data  # reduced grism spectrum
ERR_DATA = hdu['ERR'].data  # error spectrum

# reduced and continuum-filtered grism spectrum (=average of module A and B)
EMLINE_DATA = hdu['EMLINE'].data
# reduced and continuum-filtered grism spectrum in module A
EMLINE_DATA_A = hdu['EMLINEA'].data
# reduced and continuum-filtered grism spectrum in module B
EMLINE_DATA_B = hdu['EMLINEB'].data

ly, lx = np.shape(SCI_DATA)  # just to store the dimensiomns

x_array = np.arange(0, lx, 1)
wav_array = x_array*hd['CDELT1'] + hd['CRVAL1']  # create wavelength array
