from lmfit import Model
from astropy.io import fits
import numpy as np
import numpy
import numpy
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
rescale_noise = False
SAVE_FOLDER = './'
SPECTRA_FOLDER = '../../spectra/SPECTRA_O3_FINAL/ONED/'

SAVE_CATALOG = './COLA1_O3_fitted_flux.fits'
CATALOG = '../../catalogs/COLA1_O3_fitted_redshift.fits'
FIELD = 'COLA1'
noise_rescale = 0.8008

cat = fits.open(CATALOG)

data = cat[1].data

z_guesslist = data.field('z_O3doublet')
Nclumps_Y = data.field('Nclumps_Y')


with fits.open(CATALOG) as hdul:
    orig_table = hdul[1].data
    orig_cols = orig_table.columns


cat = fits.open(CATALOG)

data = cat[1].data

# IDlist = [9269]
IDlist = cat[1].data['NUMBER_1']


z_guesslist = data.field('z_O3doublet_combined_n')
Nclumps_Y = data.field('Nclumps_Y')
Nclumps_spec = data.field('Nclumps_spec')
# this is a manual column where you can keep track of which Modules should be ignored (for example if one module is contaminated)
Modignore = data.field('Module_ignore')

flux_5008_A = np.zeros(len(IDlist))
flux_5008_A_err = np.zeros(len(IDlist))
flux_5008_B = np.zeros(len(IDlist))
flux_5008_B_err = np.zeros(len(IDlist))


flux_4960_A = np.zeros(len(IDlist))
flux_4960_A_err = np.zeros(len(IDlist))
flux_4960_B = np.zeros(len(IDlist))
flux_4960_B_err = np.zeros(len(IDlist))


flux_4862_A = np.zeros(len(IDlist))
flux_4862_A_err = np.zeros(len(IDlist))
flux_4862_B = np.zeros(len(IDlist))
flux_4862_B_err = np.zeros(len(IDlist))

flux_4341_A = np.zeros(len(IDlist))
flux_4341_A_err = np.zeros(len(IDlist))
flux_4341_B = np.zeros(len(IDlist))
flux_4341_B_err = np.zeros(len(IDlist))

flux_4363_A = np.zeros(len(IDlist))
flux_4363_A_err = np.zeros(len(IDlist))
flux_4363_B = np.zeros(len(IDlist))
flux_4363_B_err = np.zeros(len(IDlist))



for q in range(len(IDlist)):
    thisID = int(IDlist[q])
    if thisID != 9269:
        continue

    thisz = z_guesslist[q]
    print('now doing id', thisID)
    thisNclumps_Y = Nclumps_Y[q]
    thismod_ign = Modignore[q]
    thisNclumps_spec = Nclumps_spec[q]

    dat = fits.open(SPECTRA_FOLDER+'spectrum_1D_%s_%s.fits' % (FIELD, thisID))
    data_1d = dat[1].data

    obs_wav = data_1d.field('wavelength')  # in Angstroms
    for module in ['A', 'B']:
        if not 'flux_tot_%s' % module in data_1d.names:
            continue

        if module == thismod_ign:
            continue
        flux_tot = data_1d.field('flux_tot_%s' % module)
        flux_tot_err = data_1d.field('flux_tot_%s_err' % module)

        if np.nanmax(flux_tot_err) == 0:
            flux_tot_err = np.zeros(
                len(flux_tot_err)) + np.nanstd(flux_tot[(obs_wav > 3.15E4)*(obs_wav < 3.95E4)])

        thisline = 5008.24
        sel_include = (obs_wav > 4975*(1+thisz)) * \
            (obs_wav < 5050*(1+thisz))

        if thisNclumps_spec == 1.:
            model = Model(gaussian)
            model.set_param_hint('totflux', min=0., max=200)
            model.set_param_hint('sigma', min=0.1, max=50)
            model.set_param_hint('c', min=-0.005, max=0.005)
            model.set_param_hint('x0', min=(
                1+thisz)*thisline - 3, max=(1+thisz)*thisline + 3)
            params = model.make_params(
                totflux=10, sigma=15., c=0, x0=(1+thisz)*thisline)

        if thisNclumps_spec == 2.:
            model = Model(gaussian, prefix='m1_') + \
                Model(gaussian, prefix='m2_')
            model.set_param_hint('m1_totflux', min=0.1, max=200)
            model.set_param_hint('m1_sigma', min=0.1, max=27)
            model.set_param_hint('m1_c', min=-0.05, max=0.05)
            model.set_param_hint('m1_x0', min=(
                1+thisz)*thisline - 24, max=(1+thisz)*thisline + 24)

            model.set_param_hint('m2_totflux', min=0.1, max=200)
            model.set_param_hint('m2_sigma', min=0.1, max=27)
            model.set_param_hint('m2_c', min=-0.05, max=0.05)
            model.set_param_hint('m2_x0', min=(
                1+thisz)*thisline - 70, max=(1+thisz)*thisline + 70)
            params = model.make_params(m1_totflux=10, m1_sigma=15., m1_c=0, m1_x0=(
                1+thisz)*thisline, m2_totflux=10, m2_sigma=15., m2_c=0, m2_x0=(1+thisz)*thisline-5.)

            params['m2_c'].vary = False

        if thisNclumps_spec == 3.:
            model = Model(gaussian, prefix='m1_') + Model(gaussian,
                                                        prefix='m2_') + Model(gaussian, prefix='m3_')
            model.set_param_hint('m1_totflux', min=0.01, max=200)
            model.set_param_hint('m1_sigma', min=0.1, max=50)
            model.set_param_hint('m1_c', min=-0.05, max=0.05)
            model.set_param_hint('m1_x0', min=(
                1+thisz)*thisline - 10, max=(1+thisz)*thisline + 10)

            model.set_param_hint('m2_totflux', min=0.01, max=200)
            model.set_param_hint('m2_sigma', min=0.1, max=50)
            model.set_param_hint('m2_c', min=-0.05, max=0.05)
            model.set_param_hint('m2_x0', min=(
                1+thisz)*thisline - 70, max=(1+thisz)*thisline + 70)

            model.set_param_hint('m3_totflux', min=0.01, max=200)
            model.set_param_hint('m3_sigma', min=0.1, max=50)
            model.set_param_hint('m3_c', min=-0.05, max=0.05)
            model.set_param_hint('m3_x0', min=(
                1+thisz)*thisline - 90, max=(1+thisz)*thisline + 90)
            params = model.make_params(m1_totflux=10, m1_sigma=15., m1_c=0, m1_x0=(1+thisz)*thisline, m2_totflux=10, m2_sigma=15.,
                                    m2_c=0, m2_x0=(1+thisz)*thisline-5., m3_totflux=10, m3_sigma=15., m3_c=0, m3_x0=(1+thisz)*thisline+5.)

            params['m2_c'].vary = False
            params['m3_c'].vary = False

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

        fig.savefig(SAVE_FOLDER+'%s_ID_%s_line_%s_mod%s.png' %
                    (FIELD, thisID, int(thisline), module),
                    bbox_inches='tight', pad_inches=0.1, facecolor='w')
        fig.clf()
        # Save stuff
        np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_obswav.npy', obs_wav)
        np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_fluxtot.npy', flux_tot)
        np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_fluxtoterr.npy', flux_tot_err)
        np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_xx.npy', xx)
        np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_yy.npy', model.eval(result.params, x=xx))


        ###################

        if thisNclumps_spec == 1:
            O3_5008_sigma = result.params['sigma'].value
            O3_5008_flux = result.params['totflux'].value
            if good_fit:
                O3_5008_flux_err = result.params['totflux'].stderr
            else:
                O3_5008_flux_err = 99.

        if thisNclumps_spec == 2:
            O3_5008_sigma_1 = result.params['m1_sigma'].value
            O3_5008_sigma_2 = result.params['m2_sigma'].value
            O3_5008_flux = result.params['m1_totflux'].value + \
                result.params['m2_totflux'].value
            if good_fit:
                O3_5008_flux_err = (
                    result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2)**0.5
            else:
                O3_5008_flux_err = 99.

        if thisNclumps_spec == 3:
            O3_5008_sigma_1 = result.params['m1_sigma'].value
            O3_5008_sigma_2 = result.params['m2_sigma'].value
            O3_5008_sigma_3 = result.params['m3_sigma'].value
            O3_5008_flux = result.params['m1_totflux'].value + \
                result.params['m2_totflux'].value + + \
                result.params['m3_totflux'].value
            if good_fit:
                O3_5008_flux_err = (
                    result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2 + result.params['m3_totflux'].stderr**2)**0.5
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

            if thisNclumps_spec == 1.:
                model = Model(gaussian)
                model.set_param_hint('totflux', min=-0., max=200)
                model.set_param_hint('sigma', min=0.1, max=50)
                model.set_param_hint('c', min=-0.005, max=0.005)
                model.set_param_hint('x0', min=(
                    1+thisz)*thisline - 3, max=(1+thisz)*thisline + 3)
                params = model.make_params(
                    totflux=10, sigma=O3_5008_sigma*thisline/5008.24, c=0, x0=(1+thisz)*thisline)
                params['sigma'].vary = False

            if thisNclumps_spec == 2.:
                model = Model(gaussian, prefix='m1_') + \
                    Model(gaussian, prefix='m2_')
                model.set_param_hint('m1_totflux', min=-0.1, max=200)
                model.set_param_hint('m1_sigma', min=0.1, max=50)
                model.set_param_hint('m1_c', min=-0.025, max=0.025)
                model.set_param_hint('m1_x0', min=(
                    1+thisz)*thisline - 12, max=(1+thisz)*thisline + 12)

                model.set_param_hint('m2_totflux', min=-0.1, max=200)
                model.set_param_hint('m2_sigma', min=0.1, max=50)
                model.set_param_hint('m2_c', min=-0.05, max=0.05)
                model.set_param_hint('m2_x0', min=(
                    1+thisz)*thisline - 70, max=(1+thisz)*thisline + 70)
                params = model.make_params(m1_totflux=10, m1_sigma=O3_5008_sigma_1*thisline/5008.24, m1_c=0, m1_x0=(
                    1+thisz)*thisline, m2_totflux=10, m2_sigma=O3_5008_sigma_2*thisline/5008.24, m2_c=0, m2_x0=(1+thisz)*thisline-5.)

                params['m2_c'].vary = False
                params['m1_sigma'].vary = False
                params['m2_sigma'].vary = False

            if thisNclumps_spec == 3.:
                model = Model(gaussian, prefix='m1_') + Model(gaussian,
                                                            prefix='m2_') + Model(gaussian, prefix='m3_')
                model.set_param_hint('m1_totflux', min=-0.001, max=200)
                model.set_param_hint('m1_sigma', min=0.1, max=50)
                model.set_param_hint('m1_c', min=-0.05, max=0.05)
                model.set_param_hint('m1_x0', min=(
                    1+thisz)*thisline - 30, max=(1+thisz)*thisline + 30)

                model.set_param_hint(
                    'm2_totflux', min=-10.001, max=200)
                model.set_param_hint('m2_sigma', min=0.1, max=50)
                model.set_param_hint('m2_c', min=-0.05, max=0.05)
                model.set_param_hint('m2_x0', min=(
                    1+thisz)*thisline - 70, max=(1+thisz)*thisline + 70)

                model.set_param_hint(
                    'm3_totflux', min=-10.001, max=200)
                model.set_param_hint('m3_sigma', min=0.1, max=50)
                model.set_param_hint('m3_c', min=-0.05, max=0.05)
                model.set_param_hint('m3_x0', min=(
                    1+thisz)*thisline - 90, max=(1+thisz)*thisline + 90)
                params = model.make_params(m1_totflux=10, m1_sigma=O3_5008_sigma_1*thisline/5008.24, m1_c=0, m1_x0=(1+thisz)*thisline, m2_totflux=10, m2_sigma=O3_5008_sigma_2 *
                                        thisline/5008.24, m2_c=0, m2_x0=(1+thisz)*thisline-5., m3_totflux=10, m3_sigma=O3_5008_sigma_3*thisline/5008.24, m3_c=0, m3_x0=(1+thisz)*thisline+5.)

                params['m2_c'].vary = False
                params['m3_c'].vary = False
                params['m1_sigma'].vary = False
                params['m2_sigma'].vary = False
                params['m3_sigma'].vary = False

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

            fig.savefig(SAVE_FOLDER+'%s_ID_%s_line_%s_mod%s.png' %
                        (FIELD, thisID, int(thisline), module),
                        bbox_inches='tight', pad_inches=0.1, facecolor='w')
            fig.clf()
            # Save stuff
            np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_obswav.npy', obs_wav)
            np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_fluxtot.npy', flux_tot)
            np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_fluxtoterr.npy', flux_tot_err)
            np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_xx.npy', xx)
            np.save(f'{FIELD}_{thisID}_{int(thisline)}_mod{module}_yy.npy', model.eval(result.params, x=xx))


            ###################

            if thisline == 4862.69:
                if thisNclumps_spec == 1:
                    Hb_flux = result.params['totflux'].value
                    if good_fit:
                        Hb_flux_err = result.params['totflux'].stderr
                    else:
                        Hb_flux_err = 99.

                if thisNclumps_spec == 2:
                    Hb_flux = result.params['m1_totflux'].value + \
                        result.params['m2_totflux'].value
                    if good_fit:
                        Hb_flux_err = (
                            result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2)**0.5
                    else:
                        Hb_flux_err = 99.

                if thisNclumps_spec == 3:
                    Hb_flux = result.params['m1_totflux'].value + \
                        result.params['m2_totflux'].value + \
                        result.params['m3_totflux'].value
                    if good_fit:
                        Hb_flux_err = (
                            result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2 + result.params['m3_totflux'].stderr**2)**0.5
                    else:
                        Hb_flux_err = 99.

            if thisline == 4960.295:
                if thisNclumps_spec == 1:
                    O3_4960_flux = result.params['totflux'].value
                    if good_fit:
                        O3_4960_flux_err = result.params['totflux'].stderr
                    else:
                        O3_4960_flux_err = 99.

                if thisNclumps_spec == 2:
                    O3_4960_flux = result.params['m1_totflux'].value + \
                        result.params['m2_totflux'].value
                    if good_fit:
                        O3_4960_flux_err = (
                            result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2)**0.5
                    else:
                        O3_4960_flux_err = 99.

                if thisNclumps_spec == 3:
                    O3_4960_flux = result.params['m1_totflux'].value + \
                        result.params['m2_totflux'].value + + \
                        result.params['m3_totflux'].value
                    if good_fit:
                        O3_4960_flux_err = (
                            result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2 + result.params['m3_totflux'].stderr**2)**0.5
                    else:
                        O3_4960_flux_err = 99.

            if thisline == 4364.44:
                if thisNclumps_spec == 1:
                    O3_4364_flux = result.params['totflux'].value
                    if good_fit:
                        O3_4364_flux_err = result.params['totflux'].stderr
                    else:
                        O3_4364_flux_err = 99.

                if thisNclumps_spec == 2:
                    O3_4364_flux = result.params['m1_totflux'].value + \
                        result.params['m2_totflux'].value
                    if good_fit:
                        O3_4364_flux_err = (
                            result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2)**0.5
                    else:
                        O3_4364_flux_err = 99.

                if thisNclumps_spec == 3:
                    O3_4364_flux = result.params['m1_totflux'].value + \
                        result.params['m2_totflux'].value + + \
                        result.params['m3_totflux'].value
                    if good_fit:
                        O3_4364_flux_err = (
                            result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2 + result.params['m3_totflux'].stderr**2)**0.5
                    else:
                        O3_4364_flux_err = 99.

            if thisline == 4341.69:
                if thisNclumps_spec == 1:
                    Hg_flux = result.params['totflux'].value
                    if good_fit:
                        Hg_flux_err = result.params['totflux'].stderr
                    else:
                        Hg_flux_err = 99.

                if thisNclumps_spec == 2:
                    Hg_flux = result.params['m1_totflux'].value + \
                        result.params['m2_totflux'].value
                    if good_fit:
                        Hg_flux_err = (
                            result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2)**0.5
                    else:
                        Hg_flux_err = 99.

                if thisNclumps_spec == 3:
                    Hg_flux = result.params['m1_totflux'].value + \
                        result.params['m2_totflux'].value + + \
                        result.params['m3_totflux'].value
                    if good_fit:
                        Hg_flux_err = (
                            result.params['m1_totflux'].stderr**2 + result.params['m2_totflux'].stderr**2 + result.params['m3_totflux'].stderr**2)**0.5
                    else:
                        Hg_flux_err = 99.


        if module == 'A':
            flux_5008_A[q] = O3_5008_flux
            flux_5008_A_err[q] = O3_5008_flux_err

            flux_4960_A[q] = O3_4960_flux
            flux_4960_A_err[q] = O3_4960_flux_err

            flux_4862_A[q] = Hb_flux
            flux_4862_A_err[q] = Hb_flux_err

            flux_4363_A[q] = O3_4364_flux
            flux_4363_A_err[q] = O3_4364_flux_err

            flux_4341_A[q] = Hg_flux
            flux_4341_A_err[q] = Hg_flux_err

        if module == 'B':
            flux_5008_B[q] = O3_5008_flux
            flux_5008_B_err[q] = O3_5008_flux_err

            flux_4960_B[q] = O3_4960_flux
            flux_4960_B_err[q] = O3_4960_flux_err

            flux_4862_B[q] = Hb_flux
            flux_4862_B_err[q] = Hb_flux_err

            flux_4363_B[q] = O3_4364_flux
            flux_4363_B_err[q] = O3_4364_flux_err

            flux_4341_B[q] = Hg_flux
            flux_4341_B_err[q] = Hg_flux_err

        # need to reset!
        print(thisID, 'modA: ')
        print(flux_4862_A[q], flux_4960_A[q], flux_5008_A[q],
              flux_4862_A_err[q], flux_4960_A_err[q], flux_5008_A_err[q])

        print(thisID, 'modB: ')
        print(flux_4862_B[q], flux_4960_B[q], flux_5008_B[q],
              flux_4862_B_err[q], flux_4960_B_err[q], flux_5008_B_err[q])
        O3_5008_flux, O3_5008_flux_err, O3_4960_flux, O3_4960_flux_err, Hb_flux, Hb_flux_err = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        print('COMB')
        print(0.5*(flux_4862_A[q]+flux_4862_B[q]), 0.5*(flux_4960_A[q] +
              flux_4960_B[q]), 0.5*(flux_5008_A[q]+flux_5008_B[q]))


f_4862_COMB = np.zeros(len(flux_5008_A))
f_4862_COMB_err = np.zeros(len(flux_5008_A))

f_4960_COMB = np.zeros(len(flux_5008_A))
f_4960_COMB_err = np.zeros(len(flux_5008_A))

f_5008_COMB = np.zeros(len(flux_5008_A))
f_5008_COMB_err = np.zeros(len(flux_5008_A))

f_4341_COMB = np.zeros(len(flux_5008_A))
f_4341_COMB_err = np.zeros(len(flux_5008_A))

f_4363_COMB = np.zeros(len(flux_5008_A))
f_4363_COMB_err = np.zeros(len(flux_5008_A))

sel_A = (flux_5008_A > -1E9)*~(flux_5008_A == 0.)
sel_B = (flux_5008_B > -1E9)*~(flux_5008_B == 0.)
sel_AB = (flux_5008_A > -1E9)*(flux_5008_B > -1E9) * \
    ~(flux_5008_B == 0.)*~(flux_5008_A == 0.)

f_4862_COMB[sel_A] = flux_4862_A[sel_A]
f_4862_COMB[sel_B] = flux_4862_B[sel_B]
f_4862_COMB[sel_AB] = 0.5*(flux_4862_A[sel_AB]+flux_4862_B[sel_AB])

f_4341_COMB[sel_A] = flux_4341_A[sel_A]
f_4341_COMB[sel_B] = flux_4341_B[sel_B]
f_4341_COMB[sel_AB] = 0.5*(flux_4341_A[sel_AB]+flux_4341_B[sel_AB])

f_4363_COMB[sel_A] = flux_4363_A[sel_A]
f_4363_COMB[sel_B] = flux_4363_B[sel_B]
f_4363_COMB[sel_AB] = 0.5*(flux_4363_A[sel_AB]+flux_4363_B[sel_AB])


f_4960_COMB[sel_A] = flux_4960_A[sel_A]
f_4960_COMB[sel_B] = flux_4960_B[sel_B]
f_4960_COMB[sel_AB] = 0.5*(flux_4960_A[sel_AB]+flux_4960_B[sel_AB])


f_5008_COMB[sel_A] = flux_5008_A[sel_A]
f_5008_COMB[sel_B] = flux_5008_B[sel_B]
f_5008_COMB[sel_AB] = 0.5*(flux_5008_A[sel_AB]+flux_5008_B[sel_AB])


f_4862_COMB_err[sel_A] = flux_4862_A_err[sel_A]
f_4862_COMB_err[sel_B] = flux_4862_B_err[sel_B]
f_4862_COMB_err[sel_AB] = 0.5 * \
    (flux_4862_A_err[sel_AB]**2+flux_4862_B_err[sel_AB]**2)**0.5

f_4341_COMB_err[sel_A] = flux_4341_A_err[sel_A]
f_4341_COMB_err[sel_B] = flux_4341_B_err[sel_B]
f_4341_COMB_err[sel_AB] = 0.5 * \
    (flux_4341_A_err[sel_AB]**2+flux_4341_B_err[sel_AB]**2)**0.5

f_4363_COMB_err[sel_A] = flux_4363_A_err[sel_A]
f_4363_COMB_err[sel_B] = flux_4363_B_err[sel_B]
f_4363_COMB_err[sel_AB] = 0.5 * \
    (flux_4363_A_err[sel_AB]**2+flux_4363_B_err[sel_AB]**2)**0.5


f_4960_COMB_err[sel_A] = flux_4960_A_err[sel_A]
f_4960_COMB_err[sel_B] = flux_4960_B_err[sel_B]
f_4960_COMB_err[sel_AB] = 0.5 * \
    (flux_4960_A_err[sel_AB]**2+flux_4960_B_err[sel_AB]**2)**0.5


f_5008_COMB_err[sel_A] = flux_5008_A_err[sel_A]
f_5008_COMB_err[sel_B] = flux_5008_B_err[sel_B]
f_5008_COMB_err[sel_AB] = 0.5 * \
    (flux_5008_A_err[sel_AB]**2+flux_5008_B_err[sel_AB]**2)**0.5


col1 = fits.Column(name='f_Hb', format='D', array=f_4862_COMB)
col2 = fits.Column(name='f_O3_4960', format='D', array=f_4960_COMB)
col3 = fits.Column(name='f_O3_5008', format='D', array=f_5008_COMB)
col4 = fits.Column(name='f_Hb_err', format='D', array=f_4862_COMB_err)
col5 = fits.Column(name='f_O3_4960_err', format='D', array=f_4960_COMB_err)
col6 = fits.Column(name='f_O3_5008_err', format='D', array=f_5008_COMB_err)

col91 = fits.Column(name='f_Hg', format='D', array=f_4341_COMB)
col92 = fits.Column(name='f_Hg_err', format='D', array=f_4341_COMB_err)
col93 = fits.Column(name='f_O3_4363', format='D', array=f_4363_COMB)
col94 = fits.Column(name='f_O3_4363_err', format='D', array=f_4363_COMB_err)



# add A and B
col7 = fits.Column(name='f_Hb_A', format='D', array=flux_4862_A)
col8 = fits.Column(name='f_O3_4960_A', format='D', array=flux_4960_A)
col9 = fits.Column(name='f_O3_5008_A', format='D', array=flux_5008_A)
col10 = fits.Column(name='f_Hb_A_err', format='D', array=flux_4862_A_err)
col11 = fits.Column(name='f_O3_4960_A_err', format='D', array=flux_4960_A_err)
col12 = fits.Column(name='f_O3_5008_A_err', format='D', array=flux_5008_A_err)

col95 = fits.Column(name='f_Hg_A', format='D', array=flux_4341_A)
col96 = fits.Column(name='f_Hg_A_err', format='D', array=flux_4341_A_err)
col97 = fits.Column(name='f_O3_4363_A', format='D', array=flux_4363_A)
col98 = fits.Column(name='f_O3_4363_A_err', format='D', array=flux_4363_A_err)


col13 = fits.Column(name='f_Hb_B', format='D', array=flux_4862_B)
col14 = fits.Column(name='f_O3_4960_B', format='D', array=flux_4960_B)
col15 = fits.Column(name='f_O3_5008_B', format='D', array=flux_5008_B)
col16 = fits.Column(name='f_Hb_B_err', format='D', array=flux_4862_B_err)
col17 = fits.Column(name='f_O3_4960_B_err', format='D', array=flux_4960_B_err)
col18 = fits.Column(name='f_O3_5008_B_err', format='D', array=flux_5008_B_err)

col99 = fits.Column(name='f_Hg_B', format='D', array=flux_4341_B)
col100 = fits.Column(name='f_Hg_B_err', format='D', array=flux_4341_B_err)
col101 = fits.Column(name='f_O3_4363_B', format='D', array=flux_4363_B)
col102 = fits.Column(name='f_O3_4363_B_err', format='D', array=flux_4363_B_err)


new_cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8,
                        col9, col10, col11, col12, col13, col14, col15, col16, col17, col18,
                        col91, col92, col93, col94, col95, col96, col97, col98, col99,
                        col100, col101, col102])
hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols)

hdu.writeto(SAVE_CATALOG, overwrite=True)
