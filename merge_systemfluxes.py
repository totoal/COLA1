from astropy.io import fits as pyfits
import numpy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


OUTPUTCAT = '../catalogs/COLA1_O3_fitted_flux.fits'
SAVECAT = '../catalogs/COLA1_O3CANDIDATES_aftermerge.fits'

cat = pyfits.open(OUTPUTCAT)
data = cat[1].data

ID = data.field('NUMBER_1')

F115W = data.field('F115W_AUTO_fnu')
F200W = data.field('F200W_AUTO_fnu')
F356W = data.field('F356W_AUTO_fnu')

F115W_e = data.field('F115W_AUTO_enu')
F200W_e = data.field('F200W_AUTO_enu')
F356W_e = data.field('F356W_AUTO_enu')


fO3_b = data.field('f_O3_4960')
fO3_r = data.field('f_O3_5008')
fHb = data.field('f_Hb')
fO3_b_e = data.field('f_O3_4960_err')
fO3_r_e = data.field('f_O3_5008_err')
fHb_e = data.field('f_Hb_err')


CHAIR = data.field('IS_GROUP_CHAIR')
GROUPID = data.field('GroupID')


uniqueGroups = numpy.unique(GROUPID)

for thisGroup in uniqueGroups:
    if thisGroup == 0:
        continue
    print(thisGroup)

    sel = GROUPID == thisGroup
    sel_lead = (GROUPID == thisGroup)*(CHAIR == True)

    sel_flux = sel

    F115W[sel_lead] = numpy.nansum(F115W[sel])
    F200W[sel_lead] = numpy.nansum(F200W[sel])
    F356W[sel_lead] = numpy.nansum(F356W[sel])

    fHb[sel_lead] = numpy.nansum(fHb[sel_flux])
    fO3_b[sel_lead] = numpy.nansum(fO3_b[sel_flux])
    fO3_r[sel_lead] = numpy.nansum(fO3_r[sel_flux])

    F115W_e[sel_lead] = numpy.nansum(F115W_e[sel]**2)**0.5
    F200W_e[sel_lead] = numpy.nansum(F200W_e[sel]**2)**0.5
    F356W_e[sel_lead] = numpy.nansum(F356W_e[sel]**2)**0.5

    fHb_e[sel_lead] = numpy.nansum(fHb_e[sel_flux]**2)**0.5
    fO3_b_e[sel_lead] = numpy.nansum(fO3_b_e[sel_flux]**2)**0.5
    fO3_r_e[sel_lead] = numpy.nansum(fO3_r_e[sel_flux]**2)**0.5


data.field('F115W_AUTO_fnu')[:] = F115W
data.field('F200W_AUTO_fnu')[:] = F200W
data.field('F356W_AUTO_fnu')[:] = F356W
data.field('F115W_AUTO_enu')[:] = F115W_e
data.field('F200W_AUTO_enu')[:] = F200W_e
data.field('F356W_AUTO_enu')[:] = F356W_e

data.field('f_Hb')[:] = fHb
data.field('f_O3_4960')[:] = fO3_b
data.field('f_O3_5008')[:] = fO3_r
data.field('f_Hb_err')[:] = fHb_e
data.field('f_O3_4960_err')[:] = fO3_b_e
data.field('f_O3_5008_err')[:] = fO3_r_e

pyfits.writeto(SAVECAT, data, overwrite=True)
