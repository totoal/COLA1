import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u


def SFR_XION_dust_corrected(Hgflux, Hgflux_err, Hbflux, Hbflux_err,
                            MUV, MUV_err, redshift, SFR_C_Hb=41.78,
                            SFR_C_UV=43.51, N_iter=5000, return_EBV=False):

    Dl = cosmo.luminosity_distance(redshift).to(u.cm).value

    Hb_obs = np.random.normal(Hbflux, Hbflux_err, size=N_iter)

    if Hgflux == 0.4 * Hbflux:
        this_HgHb_ratio = np.random.normal(0.4159, 0.075, size=N_iter)
        Hg_obs = this_HgHb_ratio * Hb_obs
    else:
        Hg_obs = np.random.normal(Hgflux, Hgflux_err, size=N_iter)
    thisMUV = np.random.normal(MUV, MUV_err, size=N_iter)

    HgHb = Hg_obs / Hb_obs

    tau = np.zeros_like(HgHb)
    where_HgHb = (HgHb < 0.473) & (HgHb > 0)
    tau[where_HgHb] = np.log(0.473 / HgHb[where_HgHb])

    thisEBV = 0.95 * tau

    # 3.76 is k_Hb for Cardelli 1989 law
    dustcorr_Hb = 10**(0.4*(thisEBV)*3.76)
    dustcorr_UV = 10**(0.4*(thisEBV)*8.68)

    L_Hb = Hb_obs * dustcorr_Hb * 4 * np.pi * Dl**2 * 1e-18

    fnu = 10**(-0.4 * (thisMUV + 48.6))  # erg/s/Hz/cm2 at 10 pc
    LUV = fnu * (4. * np.pi * (10. * 3.08568 * 10**18)**2)  # erg/s/Hz

    # k(lambda)  Calzetti k_l at 1500 is 10.333. Reddy: 8.68 -- here assume EBVstars=EBVgas
    cHb = 4.86e-13
    X_ion_top = L_Hb
    X_ion_down = LUV * dustcorr_UV * cHb

    SFRHB = 2.86 * L_Hb / 10**SFR_C_Hb
    SFRUV = LUV * dustcorr_UV / 10**SFR_C_UV * 299792458 / 1500e-10

    XION = np.log10(X_ion_top / X_ion_down)

    # median_HgHb = np.nanmedian(HgHb)
    # std_HgHb = np.nanstd(HgHb)
    # errup = [np.nanpercentile(HgHb, 84)-median_HgHb]
    # errdown = [median_HgHb-np.nanpercentile(HgHb, 16)]

    SFRHB_percs = np.nanpercentile(SFRHB, [16, 50, 84])
    SFRUV_percs = np.nanpercentile(SFRUV, [16, 50, 84])
    XION_percs = np.nanpercentile(XION, [16, 50, 84])


    # print('SFR_Hb_dust', np.nanmedian(SFRHB), -np.nanmedian(SFRHB) + np.nanpercentile(SFRHB, [16, 84]))
    # print('SFR_UV_dust', np.nanmedian(SFRUV), -np.nanmedian(SFRUV) + np.nanpercentile(SFRUV, [16, 84]))
    # print('Xion_dust', np.nanmedian(XION), -np.nanmedian(XION) + np.nanpercentile(XION, [16, 84]))

    if return_EBV:
        return thisEBV
    else:
        return SFRHB_percs, SFRUV_percs, XION_percs

if __name__ == '__main__':
    SFR_XION_dust_corrected(
                            2.117970076761039, 0.20597032970802598,
                            4.102164292177762, 0.1647917052521611,
                            -21.26, 0.07, 6.59165)
