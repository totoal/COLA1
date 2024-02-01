import os
from lmfit import Model
from astropy.io import fits
import numpy as np
from matplotlib import pyplot
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import pickle



def gaussian_O3_withfudge(x, a, c, redshift, sigma, fudge):
    x0_O31 = (1+redshift)*4960.295
    x0_O32 = (1+redshift)*5008.24
    return c + a*(np.exp(-(x-x0_O31)**2/(2*sigma**2))+c + 2.98*fudge*np.exp(-(x-x0_O32)**2/(2*sigma**2)))


# if true, rescales the mean(err_1d) to be equal to the std(data_1d) with some outlier removal
rescale_noise = False
SAVE_FOLDER = '../spectra/SPECTRA_O3_FINAL/REDSHIFT_FIT/'
SPECTRA_FOLDER = '../spectra/SPECTRA_O3_FINAL/ONED/'

os.makedirs(SAVE_FOLDER, exist_ok=True)

CATALOG = '../catalogs/DOUBLESEARCH.fits'
cat = fits.open(CATALOG)
data = cat[1].data

SAVE_CATALOG = '../catalogs/COLA1_O3_fitted_redshift.fits'
FIELD = 'COLA1'
noise_rescale = 0.8008


with fits.open(CATALOG) as hdul:
    orig_table = hdul[1].data
    orig_cols = orig_table.columns


IDlist = data.field('NUMBER_1')  # NUMBER for other fields than J0100


z_guesslist = data.field('z_O3doublet')
Nclumps_Y = data.field('Nclumps_spec')

redshift_O3doublet_A = np.zeros(len(IDlist))
redshift_O3doublet_A_err = np.zeros(len(IDlist))
redshift_O3doublet_B = np.zeros(len(IDlist))
redshift_O3doublet_B_err = np.zeros(len(IDlist))

fudge_A = np.zeros(len(IDlist))
fudge_B = np.zeros(len(IDlist))

fudge_A_err = np.zeros(len(IDlist))
fudge_B_err = np.zeros(len(IDlist))

# You will also have to add this columns by hand
# This column should contain values: 'A', 'B' or ''
# It indicates which grism modules to ignore to obtain the redshift
# You have to type this by hand using e. g. Topcat with modules with
# a lot of contamination that are useless
Modignore = data.field('Module_ignore')


for q in range(len(IDlist)):
    thisID = int(IDlist[q])
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
        try:
            flux_tot = data_1d.field('flux_tot_%s' % module)
            flux_tot_err = data_1d.field('flux_tot_%s_err' % module)
        except:
            continue

        sel_include = (obs_wav > 4920*(1+thisz))*(obs_wav < 5060*(1+thisz))


        if thisNclumps_spec == 1.:
            model = Model(gaussian_O3_withfudge)
            model.set_param_hint('a', min=0., max=200)
            model.set_param_hint('sigma', min=0.01, max=200)
            model.set_param_hint('c', min=-0.5, max=0.5)
            model.set_param_hint('redshift', min=thisz-0.02, max=thisz+0.02)
            model.set_param_hint('fudge', min=0.5, max=2.0)
            params = model.make_params(
                a=0.1, c=0., redshift=thisz, sigma=10., fudge=1.0)

        if thisNclumps_spec == 2.:
            model = Model(gaussian_O3_withfudge, prefix='m1_') + Model(gaussian_O3_withfudge, prefix='m2_')
            model.set_param_hint('m1_a', min=0., max=200)
            model.set_param_hint('m1_sigma', min=7., max=35)
            model.set_param_hint('m1_c', min=-0.5, max=0.5)
            model.set_param_hint('m1_redshift', min=thisz-0.02, max=thisz+0.02)
            model.set_param_hint('m1_fudge', min=0.95, max=1.05)

            model.set_param_hint('m2_a', min=0., max=200)
            model.set_param_hint('m2_sigma', min=7., max=35)
            model.set_param_hint('m2_c', min=-0.5, max=0.5)
            model.set_param_hint('m2_redshift', min=thisz-0.02, max=thisz+0.02)
            model.set_param_hint('m2_fudge', min=0.95, max=1.05)
            params = model.make_params(m1_a=0.05, m1_c=0., m1_redshift=thisz+0.005, m1_sigma=10.,
                                       m1_fudge=1.0, m2_a=0.05, m2_c=0., m2_redshift=thisz-0.005, m2_sigma=10., m2_fudge=1.0)

            params['m2_c'].vary = False

        if thisNclumps_spec == 3.:
            model = Model(gaussian_O3_withfudge, prefix='m1_') +\
                    Model(gaussian_O3_withfudge, prefix='m2_') +\
                    Model(gaussian_O3_withfudge, prefix='m3_')
            model.set_param_hint('m1_a', min=0., max=200)
            model.set_param_hint('m1_sigma', min=5., max=35)
            model.set_param_hint('m1_c', min=-0.5, max=0.5)
            model.set_param_hint('m1_redshift', min=thisz-0.04, max=thisz+0.02)
            model.set_param_hint('m1_fudge', min=0.95, max=1.05)

            model.set_param_hint('m2_a', min=0., max=200)
            model.set_param_hint('m2_sigma', min=5., max=35)
            model.set_param_hint('m2_c', min=-0.5, max=0.5)
            model.set_param_hint('m2_redshift', min=thisz-0.02, max=thisz+0.02)
            model.set_param_hint('m2_fudge', min=0.95, max=1.05)

            model.set_param_hint('m3_a', min=0., max=200)
            model.set_param_hint('m3_sigma', min=5., max=35)
            model.set_param_hint('m3_c', min=-0.5, max=0.5)
            model.set_param_hint('m3_redshift', min=thisz-0.02, max=thisz+0.02)
            model.set_param_hint('m3_fudge', min=0.95, max=1.05)
            params = model.make_params(m1_a=0.5, m1_c=0., m1_redshift=thisz+0.005, m1_sigma=10., m1_fudge=1.0, m2_a=0.5, m2_c=0.,
                                       m2_redshift=thisz-0.005, m2_sigma=10., m2_fudge=1.0, m3_a=0.5, m3_c=0., m3_redshift=thisz-0.00, m3_sigma=10., m3_fudge=1.0)

            params['m2_c'].vary = False
            params['m3_c'].vary = False

            params['m1_fudge'].vary = False
            params['m2_fudge'].vary = False
            params['m3_fudge'].vary = False

        result = model.fit(flux_tot[sel_include], x=obs_wav[sel_include], params=params,
                           weights=1./flux_tot_err[sel_include], nan_policy='propagate', method='ampgo')

        print(result.fit_report())
        if thisNclumps_spec == 1:
            redshift, redshifterr = result.params['redshift'].value, result.params['redshift'].stderr

        if thisNclumps_spec > 1:
            redshift1, redshift1err = result.params['m1_redshift'].value, result.params['m1_redshift'].stderr
            redshift2, redshift2err = result.params['m2_redshift'].value, result.params['m2_redshift'].stderr
            tot1 = result.params['m1_a']*result.params['m1_sigma']
            tot2 = result.params['m2_a']*result.params['m2_sigma']
            if tot1 > tot2:
                redshift, redshifterr = redshift1, redshift1err
            else:
                redshift, redshifterr = redshift2, redshift2err
            print('TOT1,TOT2', tot1, tot2, redshift1, redshift2)

        pyplot.plot(obs_wav[sel_include],
                    flux_tot[sel_include], color='tab:blue')
        pyplot.fill_between(obs_wav[sel_include], -flux_tot_err[sel_include],
                            flux_tot_err[sel_include], lw=0, alpha=0.4, color='tab:blue')

        xx = np.arange(4920*(1+thisz), 5060*(1+thisz), 1.)
        model_y = model.eval(result.params, x=xx)
        pyplot.plot(xx, model_y, lw=2, color='k')
        try:
            pyplot.title(f'z={redshift:0.5f}({redshifterr:0.5f})')
        except:
            pyplot.title(f'z={redshift}({redshifterr})')
        pyplot.savefig(
            SAVE_FOLDER+'redshift_fit_O3doublet_%s_%s_mod%s.png' % (FIELD, thisID, module))
        pyplot.clf()

        # Save the residuals
        residuals = np.interp(xx,  obs_wav[sel_include], flux_tot[sel_include],
                              left=np.nan, right=np.nan) - model_y
        res_save_name = SAVE_FOLDER+'redshift_fit_residuals_%s_%s_mod%s.pkl' % (FIELD, thisID, module)
        with open(res_save_name, 'wb') as file:
            pickle.dump(dict(xx=xx, residuals=residuals), file)

        if module == 'A':
            redshift_O3doublet_A[q] = redshift
            redshift_O3doublet_A_err[q] = redshifterr

        if module == 'B':
            redshift_O3doublet_B[q] = redshift
            redshift_O3doublet_B_err[q] = redshifterr


col1 = fits.Column(name='z_O3doublet_A_n', format='D',
                   array=redshift_O3doublet_A)
col2 = fits.Column(name='z_O3doublet_B_n', format='D',
                   array=redshift_O3doublet_B)

col3 = fits.Column(name='z_O3doublet_A_err_n', format='D',
                   array=redshift_O3doublet_A_err)
col4 = fits.Column(name='z_O3doublet_B_err_n', format='D',
                   array=redshift_O3doublet_B_err)


z_combined = np.zeros(len(redshift_O3doublet_A))

sel_A = redshift_O3doublet_A > 0
sel_B = redshift_O3doublet_B > 0
sel_AB = (redshift_O3doublet_A > 0)*(redshift_O3doublet_B > 0)

z_combined[sel_A] = redshift_O3doublet_A[sel_A]
z_combined[sel_B] = redshift_O3doublet_B[sel_B]
z_combined[sel_AB] = 0.5 * \
    (redshift_O3doublet_A[sel_AB]+redshift_O3doublet_B[sel_AB])


col5 = fits.Column(name='z_O3doublet_combined_n', format='D', array=z_combined)

col6 = fits.Column(name='fudge_A', format='D', array=fudge_A)
col7 = fits.Column(name='fudge_A_err', format='D', array=fudge_A_err)

col8 = fits.Column(name='fudge_B', format='D', array=fudge_B)
col9 = fits.Column(name='fudge_B_err', format='D', array=fudge_B_err)

new_cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8, col9])

hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols)

hdu.writeto(SAVE_CATALOG, overwrite=True)
