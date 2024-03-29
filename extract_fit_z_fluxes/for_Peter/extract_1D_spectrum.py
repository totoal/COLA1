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
FOLDER = '../spectra/SPECTRA_O3_FINAL/'

# We will save the line profiles here
SAVE_FOLDER = '../spectra/SPECTRA_O3_FINAL/OPTIMAL_PROFILES/'

# We will save the 1D spectra here
ONED_FOLDER = '../spectra/SPECTRA_O3_FINAL/ONED/'

# Create the folders if they don't exist
os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(ONED_FOLDER, exist_ok=True)

# This is your catalog.
CATALOG = 'doublet_catalog.fits'
cat = fits.open(CATALOG)
data = cat[1].data

FIELD = 'COLA1'

noise_rescale = 0.8008

IDlist = data.field('NUMBER') # IDs of the candidates

z_guesslist = data.field('z_O3doublet') # This is the predicted redshift

# This is a manual column that you might want to add manually in your catalog beforehand.
# set them all to 1, check the fits visually and change the values manually if necessary.
Nclumps_Y = data.field('Nclumps_Y')


for q in range(len(IDlist)):
    thisID = int(IDlist[q])
    thisz = z_guesslist[q]

    print('now doing id', thisID)

    thisNclumps_Y = Nclumps_Y[q]

    hdu = fits.open(FOLDER + f'stacked_2D_{FIELD}_{thisID}.fits')
    hd = hdu['EMLINE'].header
    data = hdu['EMLINE'].data

    ly, lx = np.shape(data)

    x_array = np.arange(0, lx, 1)
    wav_array = x_array*hd['CDELT1'] + hd['CRVAL1']

    COLS = []
    COLS.append(fits.Column(name='wavelength',
                unit='angstrom', format='E', array=wav_array))
    opt_weight = np.zeros(np.shape(data))

    for module in ['A', 'B']:
        data = hdu['EMLINE%s' % module].data
        errdata = hdu['ERR'].data * noise_rescale

        # Select region in 2D that contains the 5008 line ##Could also generalise to any other line
        if thisNclumps_Y == 1.:
            sel_wav = (wav_array > 5008.24*(1+thisz) - 5008.24*(1+thisz)*400/3E5) * \
                (wav_array < 5008.24*(1+thisz) + 5008.24*(1+thisz)*250/3E5)  # this
        if thisNclumps_Y > 1.:
            sel_wav = (wav_array > 5008.24*(1+thisz) - 5008.24*(1+thisz)*400/3E5) * \
                (wav_array < 5008.24*(1+thisz) + 5008.24*(1+thisz)*250/3E5)  # this

        subset_data = data[:, sel_wav]


        sel_real = np.isfinite(subset_data)
        print(len(subset_data[sel_real]))
        if len(subset_data[sel_real]) == 0:
            opt_weight = np.zeros(np.shape(data))
            continue

        collapse_y = np.nansum(subset_data, axis=1)

        x_fit = np.arange(0, len(collapse_y), 1)
        x_fit_oversample = np.arange(0, len(collapse_y), 0.01)

        print(f'{thisNclumps_Y=}')

        if thisNclumps_Y == 1.:
            model = Model(gaussian, independent_vars=('x'), prefix='m1_')
            model.set_param_hint('m1_totflux', min=0., max=20)
            model.set_param_hint('m1_sigma', min=0.8, max=4.)
            model.set_param_hint('m1_c', min=-0.5, max=0.02)
            model.set_param_hint('m1_x0', min=21., max=29.)

            params = model.make_params(totflux=0.5, c=0, x0=26, sigma=1)

        if thisNclumps_Y == 2.:
            model = Model(gaussian, prefix='m1_') + Model(gaussian, prefix='m2_')
            model.set_param_hint('m1_sigma', min=0.8, max=3.5)
            model.set_param_hint('m2_sigma', min=0.8, max=3.5)

            model.set_param_hint('m1_totflux', min=0., max=20)
            model.set_param_hint('m2_totflux', min=0., max=20)

            model.set_param_hint('m1_c', min=-0.5, max=0.02)

            model.set_param_hint('m1_x0', min=21., max=29.)
            model.set_param_hint('m2_x0', min=20., max=30)

            params = model.make_params(
                m1_totflux=1, m1_c=0, m1_x0=22, m1_sigma=1, m2_totflux=2., m2_c=0, m2_x0=26, m2_sigma=1)
            params['m2_c'].vary = False

        if thisNclumps_Y == 3.:
            model = Model(gaussian, prefix='m1_') +\
                    Model(gaussian, prefix='m2_') +\
                    Model(gaussian, prefix='m3_')
            model.set_param_hint('m1_sigma', min=0.8, max=3.5)
            model.set_param_hint('m2_sigma', min=0.8, max=3.5)
            model.set_param_hint('m3_sigma', min=0.8, max=3.5)

            model.set_param_hint('m1_totflux', min=0., max=20)
            model.set_param_hint('m2_totflux', min=0., max=20)
            model.set_param_hint('m3_totflux', min=0., max=20)

            model.set_param_hint('m1_c', min=-0.5, max=0.5)

            model.set_param_hint('m1_x0', min=21., max=29.)
            model.set_param_hint('m2_x0', min=20., max=30)
            model.set_param_hint('m3_x0', min=20., max=30)

            params = model.make_params(m1_totflux=2, m1_c=0, m1_x0=26, m1_sigma=2, m2_totflux=0.2,
                                       m2_c=0, m2_x0=24, m2_sigma=2, m3_totflux=0.2, m3_c=0, m3_x0=28, m3_sigma=2)

            params['m2_c'].vary = False
            params['m3_c'].vary = False

        if thisNclumps_Y == 4.:  # You probably don't need that many clumps
            model = Model(gaussian, prefix='m1_') +\
                    Model(gaussian, prefix='m2_') +\
                    Model(gaussian, prefix='m3_') +\
                    Model(gaussian, prefix='m4_')
            model.set_param_hint('m1_sigma', min=0.8, max=3.5)
            model.set_param_hint('m2_sigma', min=0.8, max=3.5)
            model.set_param_hint('m3_sigma', min=0.8, max=3.5)
            model.set_param_hint('m4_sigma', min=0.8, max=3.5)

            model.set_param_hint('m1_totflux', min=0., max=20)
            model.set_param_hint('m2_totflux', min=0., max=20)
            model.set_param_hint('m3_totflux', min=0., max=20)
            model.set_param_hint('m4_totflux', min=0., max=20)

            model.set_param_hint('m1_c', min=-0.5, max=0.5)

            model.set_param_hint('m1_x0', min=22., max=28)
            model.set_param_hint('m2_x0', min=5., max=26)
            model.set_param_hint('m3_x0', min=26., max=38)
            model.set_param_hint('m4_x0', min=5., max=26)

            params = model.make_params(m1_totflux=2, m1_c=0, m1_x0=26, m1_sigma=2,
                                       m2_totflux=0.2, m2_c=0, m2_x0=10,
                                       m2_sigma=2, m3_totflux=0.2, m3_c=0, m3_x0=30,
                                       m3_sigma=2, m4_totflux=0.2, m4_c=0, m4_x0=18, m4_sigma=2)

            params['m2_c'].vary = False
            params['m3_c'].vary = False
            params['m4_c'].vary = False

        result = model.fit(collapse_y, x=x_fit, params=params)

        print(result.fit_report())

        fig, (ax1, ax2) = pyplot.subplots(1, 2)
        ax1.imshow(np.arcsinh(subset_data), origin='lower')
        ax2.plot(x_fit, collapse_y)
        ax2.plot(x_fit_oversample, model.eval(result.params, x=x_fit_oversample))
        ax1.axhline(25, color='w', ls=':')
        ax2.axvline(25, color='dimgray', ls=':')
        ax2.set_xlim(0, 50)
        pyplot.savefig(SAVE_FOLDER+'opt_profile_fit_%s_%s_mod%s.png' %
                       (FIELD, thisID, module))
        pyplot.clf()

        result.params['m1_c'].value = 0.
        model_y = model.eval(result.params, x=x_fit)
        model_y = model_y/np.nansum(model_y)

        for j in range(len(opt_weight[0, :])):
            opt_weight[:, j] = model_y
        fits.writeto(SAVE_FOLDER+'opt_weight_%s_%s_mod%s.fits' %
                       (FIELD, thisID, module), opt_weight, overwrite=True)

        hdu.append(fits.ImageHDU(data=opt_weight,
                   header=hd, name='OPT_WEIGHT%s' % module))

        # NOW EXTRACT THE 1D
        t1 = np.nansum(data*opt_weight, axis=0)
        t2 = np.nansum(opt_weight**2, axis=0)
        emline_extracted = t1/t2

        ivar = errdata**-2

        t1 = np.nansum(ivar*opt_weight**2, axis=0)
        t2 = np.nansum(opt_weight, axis=0)
        emline_extracted_err = (t1/t2)**-0.5

        COLS.append(fits.Column(name='flux_tot_%s' % module,
                    unit='1E-18 erg/s/cm2/A', format='E', array=emline_extracted))
        COLS.append(fits.Column(name='flux_tot_%s_err' % module,
                    unit='1E-18 erg/s/cm2/A', format='E', array=emline_extracted_err))

    cols = fits.ColDefs(COLS)
    hdu_1D = fits.BinTableHDU.from_columns(cols)
    hdu = fits.PrimaryHDU(numpy.arange(100.))
    hdu_1D.writeto(ONED_FOLDER+'spectrum_1D_%s_%s.fits' %
                   (FIELD, thisID), overwrite=True)
