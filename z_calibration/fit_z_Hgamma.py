import os
from lmfit import Model
from astropy.io import fits
import numpy as np
from matplotlib import pyplot
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def gaussian_Hbeta(x, a, c, redshift, sigma):
    x0_Hb = (1+redshift)*4341.69
    return c + a*np.exp(-(x-x0_Hb)**2/(2*sigma**2))


# if true, rescales the mean(err_1d) to be equal to the std(data_1d) with some outlier removal
rescale_noise = False
SAVE_FOLDER = '../../spectra/SPECTRA_O3_FINAL/REDSHIFT_FIT/'
SPECTRA_FOLDER = '../../spectra/SPECTRA_O3_FINAL/ONED/'

os.makedirs(SAVE_FOLDER, exist_ok=True)

CATALOG = '../../catalogs/DOUBLESEARCH.fits'
SAVE_CATALOG = './COLA1_O3_all_lines_redshift.fits'
FIELD = 'COLA1'
noise_rescale = 0.8008


with fits.open(CATALOG) as hdul:
    orig_table = hdul[1].data
    orig_cols = orig_table.columns


cat = fits.open(CATALOG)

data = cat[1].data
IDlist = data.field('NUMBER_1')  # NUMBER for other fields than J0100


z_guesslist = data.field('z_O3doublet')
Nclumps_Y = data.field('Nclumps_spec')

redshift_O3doublet_A = np.zeros(len(IDlist))
redshift_O3doublet_A_err = np.zeros(len(IDlist))
redshift_O3doublet_B = np.zeros(len(IDlist))
redshift_O3doublet_B_err = np.zeros(len(IDlist))

dz_A = np.zeros(len(IDlist))
dz_B = np.zeros(len(IDlist))
dz_Hb_A = np.zeros(len(IDlist))
dz_Hb_B = np.zeros(len(IDlist))

fudge_A = np.zeros(len(IDlist))
fudge_B = np.zeros(len(IDlist))

fudge_A_err = np.zeros(len(IDlist))
fudge_B_err = np.zeros(len(IDlist))

Modignore = data.field('Module_ignore')


for q in range(len(IDlist)):
    thisID = int(IDlist[q])
    # Do ONLY COLA1
    if thisID != 9269:
        continue

    thisz = z_guesslist[q]
    print('now doing id', thisID, q)

    thismod_ign = Modignore[q]

    thisNclumps_spec = Nclumps_Y[q]

    dat = fits.open(SPECTRA_FOLDER+'spectrum_1D_%s_%s.fits' % (FIELD, thisID))
    data_1d = dat[1].data

    obs_wav = data_1d.field('wavelength')  # in Angstroms
    for module in ['A', 'B']:
        if module == thismod_ign:
            continue
        print(module)
        try:
            flux_tot = data_1d.field('flux_tot_%s' % module)
            flux_tot_err = data_1d.field('flux_tot_%s_err' % module)
        except:
            continue

        sel_include = (obs_wav > 4300*(1+thisz))*(obs_wav < 4400*(1+thisz))

        # print(flux_tot[sel_include],flux_tot_err[sel_include])

        model = Model(gaussian_Hbeta)
        model.set_param_hint('a', min=0., max=200)
        model.set_param_hint('sigma', min=0.01, max=200)
        model.set_param_hint('c', min=-0.5, max=0.5)
        model.set_param_hint('redshift', min=thisz-0.03, max=thisz+0.03)
        params = model.make_params(
            a=0.1, c=0., redshift=thisz, sigma=10.)

        result = model.fit(flux_tot[sel_include], x=obs_wav[sel_include], params=params,
                           weights=1./flux_tot_err[sel_include], nan_policy='propagate', method='ampgo')

        print(result.fit_report())
        redshift, redshifterr = result.params['redshift'].value, result.params['redshift'].stderr

        ########### PLOT ############
        pyplot.plot(obs_wav[sel_include],
                    flux_tot[sel_include], color='tab:blue')
        pyplot.fill_between(obs_wav[sel_include], -flux_tot_err[sel_include],
                            flux_tot_err[sel_include], lw=0, alpha=0.4, color='tab:blue')

        xx = np.arange(4300*(1+thisz), 4400*(1+thisz), 1.)
        pyplot.plot(xx, model.eval(result.params, x=xx), lw=2, color='k')
        try:
            pyplot.title(f'z={redshift:0.5f}({redshifterr:0.5f})')
        except:
            pyplot.title(f'z={redshift}({redshifterr})')
        pyplot.show(block=True)
