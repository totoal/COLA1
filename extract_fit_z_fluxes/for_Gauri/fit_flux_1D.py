from lmfit import Model
from astropy.io import fits
import numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import matplotlib
from matplotlib import pyplot
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

line_name_dict = {
    5008.24: r'[OIII]$\lambda$5008',
    4960.295: r'[OIII]$\lambda$4960',
    4862.69: r'H$\beta$',
    4341.69: r'H$\gamma$',
    4364.44: r'[OIII]$\lambda$4364'
}

def gaussian(x, totflux, c, x0, sigma):
    return totflux*((sigma)**-1 * (2*np.pi)**-0.5 * np.exp(-(x-x0)**2/(2*sigma**2)))+c


# if true, rescales the mean(err_1d) to be equal to the std(data_1d) with some outlier removal
SAVE_FOLDER = 'flux_fits/'
SPECTRA_FOLDER = 'spectra/ONED/'

# Save the final catalog of fluxes to
SAVE_CATALOG = './Stacks_catalog_fitted_flux.fits'

CATALOG = 'Stacks_catalog.fits'

cat = fits.open(CATALOG)

data = cat[1].data


with fits.open(CATALOG) as hdul:
    orig_table = hdul[1].data
    orig_cols = orig_table.columns


cat = fits.open(CATALOG)

data = cat[1].data

IDlist = cat[1].data['NUMBER']


flux_5008 = np.zeros(len(IDlist))
flux_5008_err = np.zeros(len(IDlist))


flux_4960 = np.zeros(len(IDlist))
flux_4960_err = np.zeros(len(IDlist))


flux_4862 = np.zeros(len(IDlist))
flux_4862_err = np.zeros(len(IDlist))

flux_4341 = np.zeros(len(IDlist))
flux_4341_err = np.zeros(len(IDlist))

flux_4363 = np.zeros(len(IDlist))
flux_4363_err = np.zeros(len(IDlist))



for q in range(len(IDlist)):
    thisID = int(IDlist[q])

    thisz = data['z_O3doublet'] # This is the column with the predicted O3 doublet

    print('now doing id', thisID)

    dat = fits.open(SPECTRA_FOLDER + f'spectrum_1D_{thisID}.fits')
    data_1d = dat[1].data

    obs_wav = data_1d.field('wavelength')  # in Angstroms

    flux_tot = data_1d.field('flux_tot')
    flux_tot_err = data_1d.field('flux_tot_err')

    if np.nanmax(flux_tot_err) == 0:
        flux_tot_err = np.zeros(
            len(flux_tot_err)) + np.nanstd(flux_tot[(obs_wav > 3.15E4)*(obs_wav < 3.95E4)])

    thisline = 5008.24
    sel_include = (obs_wav > 4975*(1+thisz)) * \
        (obs_wav < 5050*(1+thisz))

    model = Model(gaussian)
    model.set_param_hint('totflux', min=0., max=200)
    model.set_param_hint('sigma', min=0.1, max=50)
    model.set_param_hint('c', min=-0.005, max=0.005)
    model.set_param_hint('x0', min=(
        1+thisz)*thisline - 3, max=(1+thisz)*thisline + 3)
    params = model.make_params(
        totflux=10, sigma=15., c=0, x0=(1+thisz)*thisline)


    for i in range(5): # Try 5 times
        result = model.fit(flux_tot[sel_include], x=obs_wav[sel_include], params=params, weights=1. /
                        flux_tot_err[sel_include], nan_policy='propagate', method='differential_evolution', max_nfev=100000)
        good_fit = result.errorbars
        if not good_fit:
            continue

    print(result.fit_report())

    
    ###### FIGURE #####

    fig, ax = pyplot.subplots()

    ax.plot(obs_wav[sel_include], flux_tot[sel_include],
                color='tab:blue', drawstyle='steps-mid')
    ax.fill_between(obs_wav[sel_include], -flux_tot_err[sel_include],
                        flux_tot_err[sel_include], lw=0, alpha=0.4, color='tab:blue')

    xx = numpy.arange(min(obs_wav[sel_include]), max(
        obs_wav[sel_include]), 0.01)
    ax.plot(xx, model.eval(result.params, x=xx),
                lw=2, color='k', alpha=0.5)

    this_line_name = line_name_dict[thisline]
    ax.set_title(this_line_name)
    ax.set_xlabel(r'$\lambda_\mathrm{obs}$ [\AA]')
    ax.set_ylabel(r'$f_\lambda\cdot 10^{-18}$ [erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$]')

    fig.savefig(SAVE_FOLDER + f'Stack_ID_{thisID}_line_{int(thisline)}.png',
                bbox_inches='tight', pad_inches=0.1, facecolor='w')
    fig.clf()


    ###################

    O3_5008_sigma = result.params['sigma'].value
    O3_5008_flux = result.params['totflux'].value
    if good_fit:
        O3_5008_flux_err = result.params['totflux'].stderr
    else:
        O3_5008_flux_err = 99.

    # NOW 4960 and Hbeta
    LINES = [4862.69, 4960.295, 4341.69, 4364.44]
    SELECTIONS = [(obs_wav > 4800*(1+thisz))*(obs_wav < 4900*(1+thisz)),
                (obs_wav > 4935*(1+thisz))*(obs_wav < 4985*(1+thisz)),
                (obs_wav > 4300*(1+thisz))*(obs_wav < 4400*(1+thisz)),
                (obs_wav > 4300*(1+thisz))*(obs_wav < 4410*(1+thisz))]

    for qq in range(len(LINES)):
        thisline = LINES[qq]
        sel_include = SELECTIONS[qq]

        model = Model(gaussian)
        model.set_param_hint('totflux', min=-0., max=200)
        model.set_param_hint('sigma', min=0.1, max=50)
        model.set_param_hint('c', min=-0.005, max=0.005)
        model.set_param_hint('x0', min=(
            1+thisz)*thisline - 3, max=(1+thisz)*thisline + 3)
        params = model.make_params(
            totflux=10, sigma=O3_5008_sigma*thisline/5008.24, c=0, x0=(1+thisz)*thisline)
        params['sigma'].vary = False


        result = model.fit(flux_tot[sel_include], x=obs_wav[sel_include], params=params, weights=1. /
                        flux_tot_err[sel_include], nan_policy='propagate', method='differential_evolution', max_nfev=100000)
        good_fit = result.errorbars

        print(result.fit_report())

        ###### FIGURE #####

        fig, ax = pyplot.subplots()

        ax.plot(obs_wav[sel_include], flux_tot[sel_include],
                    color='tab:blue', drawstyle='steps-mid')
        ax.fill_between(obs_wav[sel_include], -flux_tot_err[sel_include],
                            flux_tot_err[sel_include], lw=0, alpha=0.4, color='tab:blue')

        xx = numpy.arange(min(obs_wav[sel_include]), max(
            obs_wav[sel_include]), 0.01)
        ax.plot(xx, model.eval(result.params, x=xx),
                    lw=2, color='k', alpha=0.5)

        this_line_name = line_name_dict[thisline]
        ax.set_title(this_line_name)
        ax.set_xlabel(r'$\lambda_\mathrm{obs}$ [\AA]')
        ax.set_ylabel(r'$f_\lambda\cdot 10^{-18}$ [erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$]')

        fig.savefig(SAVE_FOLDER + f'Stack_ID_{thisID}_line_{int(thisline)}.png',
                    bbox_inches='tight', pad_inches=0.1, facecolor='w')

        fig.clf()


        ###################

        if thisline == 4862.69:
            Hb_flux = result.params['totflux'].value
            if good_fit:
                Hb_flux_err = result.params['totflux'].stderr
            else:
                Hb_flux_err = 99.


        if thisline == 4960.295:
            O3_4960_flux = result.params['totflux'].value
            if good_fit:
                O3_4960_flux_err = result.params['totflux'].stderr
            else:
                O3_4960_flux_err = 99.

        if thisline == 4364.44:
            O3_4364_flux = result.params['totflux'].value
            if good_fit:
                O3_4364_flux_err = result.params['totflux'].stderr
            else:
                O3_4364_flux_err = 99.


        if thisline == 4341.69:
            Hg_flux = result.params['totflux'].value
            if good_fit:
                Hg_flux_err = result.params['totflux'].stderr
            else:
                Hg_flux_err = 99.


    flux_5008[q] = O3_5008_flux
    flux_5008_err[q] = O3_5008_flux_err

    flux_4960[q] = O3_4960_flux
    flux_4960_err[q] = O3_4960_flux_err

    flux_4862[q] = Hb_flux
    flux_4862_err[q] = Hb_flux_err

    flux_4363[q] = O3_4364_flux
    flux_4363_err[q] = O3_4364_flux_err

    flux_4341[q] = Hg_flux
    flux_4341_err[q] = Hg_flux_err


col7 = fits.Column(name='f_Hb', format='D', array=flux_4862)
col8 = fits.Column(name='f_O3_4960', format='D', array=flux_4960)
col9 = fits.Column(name='f_O3_5008', format='D', array=flux_5008)
col10 = fits.Column(name='f_Hb_err', format='D', array=flux_4862_err)
col11 = fits.Column(name='f_O3_4960_err', format='D', array=flux_4960_err)
col12 = fits.Column(name='f_O3_5008_err', format='D', array=flux_5008_err)
col95 = fits.Column(name='f_Hg', format='D', array=flux_4341)
col96 = fits.Column(name='f_Hg_err', format='D', array=flux_4341_err)
col97 = fits.Column(name='f_O3_4363', format='D', array=flux_4363)
col98 = fits.Column(name='f_O3_4363_err', format='D', array=flux_4363_err)

new_cols = fits.ColDefs([col7, col8, col9, col10, col11, col12,
                        col95, col96, col97, col98])
hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols)

hdu.writeto(SAVE_CATALOG, overwrite=True)
