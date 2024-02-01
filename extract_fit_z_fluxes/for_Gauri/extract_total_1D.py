import os
from lmfit import Model
from astropy.io import fits
import numpy as np
from matplotlib import pyplot
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def gaussian(x, totflux, c, x0, sigma):
    return totflux*((sigma)**-1 * (2*np.pi)**-0.5 * np.exp(-(x-x0)**2/(2*sigma**2)))+c


# This is the folder with the grism spectra
FOLDER = 'stacks_2D_spectra/'

# We will save the line profiles here
SAVE_FOLDER = 'spectra/OPTIMAL_PROFILES/'

# We will save the 1D spectra here
ONED_FOLDER = 'spectra/ONED/'

# Create the folders if they don't exist
os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(ONED_FOLDER, exist_ok=True)

# This is your catalog.
CATALOG = './Stacks_catalog.fits'

noise_rescale = 0.8008

# J1030: 0.715
cat = fits.open(CATALOG)

data = cat[1].data

# IDs of the objects to extract the spectra
IDlist = data.field('NUMBER')

for q in range(len(IDlist)):
    thisID = int(IDlist[q])

    thisz = data['z_O3doublet'] # This is the column with the predicted O3 doublet

    print('now doing id', thisID)

    hdu = fits.open(FOLDER + f'stacked_2D_{thisID}.fits')
    hd = hdu['EMLINE'].header
    data = hdu['EMLINE'].data

    ly, lx = np.shape(data)

    x_array = np.arange(0, lx, 1)
    wav_array = x_array*hd['CDELT1'] + hd['CRVAL1']

    COLS = []
    COLS.append(fits.Column(name='wavelength',
                unit='angstrom', format='E', array=wav_array))
    opt_weight = np.zeros(np.shape(data))

    data = hdu['EMLINE'].data
    errdata = hdu['ERR'].data * noise_rescale

    # Select region in 2D that contains the 5008 line ##Could also generalise to any other line
    sel_wav = (wav_array > 5008.24*(1+thisz) - 5008.24*(1+thisz)*400/3E5) * \
        (wav_array < 5008.24*(1+thisz) + 5008.24*(1+thisz)*250/3E5) 

    subset_data = data[:, sel_wav]

    sel_real = np.isfinite(subset_data)
    print(len(subset_data[sel_real]))
    if len(subset_data[sel_real]) == 0:
        opt_weight = np.zeros(np.shape(data))
        continue

    collapse_y = np.nansum(subset_data, axis=1)

    x_fit = np.arange(0, len(collapse_y), 1)
    x_fit_oversample = np.arange(0, len(collapse_y), 0.01)

    model = Model(gaussian, independent_vars=('x'), prefix='m1_')
    model.set_param_hint('m1_totflux', min=0., max=20)
    model.set_param_hint('m1_sigma', min=0.8, max=4.)
    model.set_param_hint('m1_c', min=-0.5, max=0.02)
    model.set_param_hint('m1_x0', min=21., max=29.)

    params = model.make_params(totflux=0.5, c=0, x0=26, sigma=1)
    result = model.fit(collapse_y, x=x_fit, params=params)

    print(result.fit_report())

    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    ax1.imshow(np.arcsinh(subset_data), origin='lower')
    ax2.plot(x_fit, collapse_y)
    ax2.plot(x_fit_oversample, model.eval(
        result.params, x=x_fit_oversample))
    ax1.axhline(25, color='w', ls=':')
    ax2.axvline(25, color='dimgray', ls=':')
    ax2.set_xlim(0, 50)
    pyplot.savefig(SAVE_FOLDER+f'opt_profile_fit_{thisID}.png')
    pyplot.clf()

    result.params['m1_c'].value = 0.
    model_y = model.eval(result.params, x=x_fit)
    model_y = model_y/np.nansum(model_y)

    for j in range(len(opt_weight[0, :])):
        opt_weight[:, j] = model_y
    fits.writeto(SAVE_FOLDER + f'opt_weight_{thisID}.fits',
                   opt_weight, overwrite=True)

    hdu.append(fits.ImageHDU(data=opt_weight,
                header=hd, name='OPT_WEIGHT'))

    # NOW EXTRACT THE 1D
    # Hornes optimal extraction of model
    t1 = np.nansum(data*opt_weight, axis=0)
    t2 = np.nansum(opt_weight**2, axis=0)
    emline_extracted = t1/t2

    ivar = errdata**-2

    t1 = np.nansum(ivar*opt_weight**2, axis=0)
    t2 = np.nansum(opt_weight, axis=0)
    emline_extracted_err = (t1/t2)**-0.5

    COLS.append(fits.Column(name='flux_tot',
                unit='1E-18 erg/s/cm2/A', format='E', array=emline_extracted))
    COLS.append(fits.Column(name='flux_tot_err',
                unit='1E-18 erg/s/cm2/A', format='E', array=emline_extracted_err))

    cols = fits.ColDefs(COLS)  # ,col13,col14])
    hdu_1D = fits.BinTableHDU.from_columns(cols)
    hdu = fits.PrimaryHDU(numpy.arange(100.))
    hdu_1D.writeto(ONED_FOLDER + f'spectrum_1D_{thisID}.fits',
                   overwrite=True)
