from scipy.integrate import simps
from lmfit import Model
import numpy as np
import scipy.ndimage as snd
from scipy.interpolate import interp1d
import numpy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


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
    # sel_zero=(lamb<3E4) + (lamb>4E4)
    lam = dummy

    I1 = simps(S*T*lam, lam)  # Denominator
    I2 = simps(T/lam, lam)  # Numerator
    fnu = I1/I2 / c  # Average flux density

    mAB = -2.5*np.log10(fnu) - 48.6  # AB magnitude

    return mAB


# norm is in erg/s/cm2/AA at lambda_0=1500 A
def continuum_and_lines(dummyx, norm, beta, redshift):
    norm = 1E-20*norm

    lamb = numpy.arange(1215, 3000, 0.01)*(1+redshift)
    cont = norm*(lamb/(1500.*(1+redshift)))**beta
    # just exists in case we would want to add lines
    lines = numpy.zeros(numpy.shape(lamb))

    # sel_1500 = (lamb < ((1500)*(1+redshift)+15.)) * \
    #     (lamb > ((1500)*(1+redshift)-15.))

    # mean_1500 = numpy.nanmean(cont[sel_1500])

    # fnu_1500 = mean_1500*(3.34E4*(1500.*(1+redshift))**2)  # erg/s/cm2/A

    sedmodel = lines+cont
    fn_F115W = 10**(-0.4*(get_obs_mag('JWST_NIRCam.F115W.dat',
                    lamb, sedmodel)-8.9)) * 1E9  # nJy
    fn_F150W = 10**(-0.4*(get_obs_mag('JWST_NIRCam.F150W.dat',
                    lamb, sedmodel)-8.9)) * 1E9  # nJy
    fn_F200W = 10**(-0.4*(get_obs_mag('JWST_NIRCam.F200W.dat',
                    lamb, sedmodel)-8.9)) * 1E9  # nJy

    return numpy.array([fn_F115W, fn_F150W, fn_F200W])


thisfnu_F115W = 222.75876
thisfnu_F115W_err = 5.006681561755274


thisfnu_F150W = 161.87422
thisfnu_F150W_err = 4.550855908814856


thisfnu_F200W = 182.78079
thisfnu_F200W_err = 2.6739856073287096

thisz = 6.59165

# NOTE, WE COULD ALSO JUST CHECK WHAT WE GET FROM F115W and F150W alone. Beta is defined as the slope at 1500 Angstrom rest-frame, and F200W is pretty far away from that


this_flux_array = np.array([thisfnu_F115W, thisfnu_F150W, thisfnu_F200W])
# Adding 5% of the flux as uncertainty, e.g. flux calibration uncertainties
this_flux_err_array = np.array(
    [thisfnu_F115W_err, thisfnu_F150W_err, thisfnu_F200W_err]) + 0.05*this_flux_array

print('COLA1')
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
