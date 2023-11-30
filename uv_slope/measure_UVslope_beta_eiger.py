from scipy.integrate import simps
from lmfit import Model
import numpy as np
import scipy.ndimage as snd
from scipy.interpolate import interp1d
import numpy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.io import fits


def get_obs_mag(filename, lamb, fluxdensity):
    d110 = numpy.loadtxt(filename)
    c = 2.998E18  # Angstrom/s

    wav_110 = d110[:, 0]  # angstrom
    trans_110 = d110[:, 1]
    trans_110 = snd.gaussian_filter(trans_110, sigma=3)

    dummy = np.arange(0.924, 2.27, 0.005) * 1E4
    trans_110 = trans_110/numpy.nanmax(trans_110)
    interp_transmission = interp1d(
        wav_110, trans_110, kind='linear', fill_value='extrapolate')
    interp_flux = interp1d(lamb, fluxdensity, kind='cubic')
    S = interp_flux(dummy)
    T = interp_transmission(dummy)

    lam = dummy

    I1 = simps(S*T*lam, lam)  # Denominator
    I2 = simps(T/lam, lam)  # Numerator
    fnu = I1/I2 / c  # Average flux density

    mAB = -2.5*np.log10(fnu) - 48.6  # AB magnitude

    return mAB


# norm is in erg/s/cm2/AA at lambda_0=1500 A
def continuum_and_lines(dummyx, norm, beta, redshift):
    norm = 1E-20*norm

    lamb = numpy.arange(900, 4000, 0.01)*(1+redshift)
    cont = norm*(lamb/(1500.*(1+redshift)))**beta
    # just exists in case we would want to add lines
    lines = numpy.zeros(numpy.shape(lamb))

    sedmodel = lines+cont
    fn_F115W = 10**(-0.4*(get_obs_mag('JWST_NIRCam.F115W.dat',
                    lamb, sedmodel)-8.9)) * 1E9  # nJy
    fn_F200W = 10**(-0.4*(get_obs_mag('JWST_NIRCam.F200W.dat',
                    lamb, sedmodel)-8.9)) * 1E9  # nJy

    return numpy.array([fn_F115W, fn_F200W])


def fit_UV_slope(cat):
    data = cat[1].data

    N_sources = len(data)

    beta_Arr = np.empty(N_sources)
    beta_err_Arr = np.empty(N_sources)
    norm_Arr = np.empty(N_sources)
    norm_err_Arr = np.empty(N_sources)

    for iii in range(N_sources):
        thisfnu_F115W = data['fnu_F115W_AUTO_apcor'][iii]
        thisfnu_F200W = data['fnu_F200W_AUTO_apcor'][iii]

        thisfnu_F115W_err = data['enu_F115W_aper_model'][iii]
        thisfnu_F200W_err = data['enu_F200W_aper_model'][iii]

        # Theres a couple sources without a flux measurements for some reason
        if np.any(np.isnan([thisfnu_F115W, thisfnu_F200W, thisfnu_F115W_err, thisfnu_F200W_err])):
            beta_Arr[iii] = np.nan
            beta_err_Arr[iii] = np.nan
            norm_Arr[iii] = np.nan
            norm_err_Arr[iii] = np.nan
            continue

        thisz = data['z_O3doublet_combined_n'][iii]

        #########

        this_flux_array = np.array([thisfnu_F115W, thisfnu_F200W])
        # Adding 5% of the flux as uncertainty, e.g. flux calibration uncertainties
        this_flux_err_array = np.array(
            [thisfnu_F115W_err, thisfnu_F200W_err]) + 0.05*this_flux_array

        print(int(data['NUMBER'][iii]), f'({iii + 1} / {N_sources})')
        print('Observed', this_flux_array, this_flux_err_array)

        dx = 2.  # this is just a dummy

        model = Model(continuum_and_lines)
        model.set_param_hint('norm', min=0.001, max=40.)
        model.set_param_hint('beta', min=-4., max=0.)
        model.set_param_hint('redshift', min=thisz-0.005, max=thisz+0.005)


        params = model.make_params(norm=0.5, beta=-2., redshift=thisz)
        params['redshift'].vary = False

        result = model.fit(this_flux_array, params, dummyx=dx, nan_policy='raise',
                        weights=1./this_flux_err_array, scale_covar=False)

        print(result.fit_report())
        print('Model', model.eval(result.params, dummyx=dx))

        beta_Arr[iii] = result.params['beta'].value
        beta_err_Arr[iii] = result.params['beta'].stderr
        norm_Arr[iii] = result.params['norm'].value
        norm_err_Arr[iii] = result.params['norm'].stderr

    return beta_Arr, beta_err_Arr, norm_Arr, norm_err_Arr



if __name__ == '__main__':
    # cat_path = '/home/alberto/cosmos/ista/COLA1/catalogs/COLA1_O3doublet_catalog_1.0.fits'
    cat_path = '/home/alberto/cosmos/ista/COLA1/catalogs/eiger/EIGER_5fields_O3doublets_SYSTEMS_31102023.fits'

    cat = fits.open(cat_path)

    fit_slope_results = fit_UV_slope(cat)


    col0 = fits.Column(name='NUMBER', format='D', array=cat[1].data['NUMBER'])
    col1 = fits.Column(name='beta', format='D', array=fit_slope_results[0])
    col2 = fits.Column(name='beta_err', format='D', array=fit_slope_results[1])
    col3 = fits.Column(name='norm', format='D', array=fit_slope_results[2])
    col4 = fits.Column(name='norm_err', format='D', array=fit_slope_results[3])


    orig_cols = cat[1].data.columns
    new_cols = fits.ColDefs([col0, col1, col2, col3, col4])

    hdu = fits.BinTableHDU.from_columns(orig_cols + new_cols)

    SAVE_CATALOG = 'UV_slopes_EIGER_field.fits'
    hdu.writeto(SAVE_CATALOG, overwrite=True)
