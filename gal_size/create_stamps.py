import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


def create_cutout(directimage, RA, DEC, lx, ly):
    hdu = fits.open(directimage)
    wcs = WCS(hdu['SCI'].header)
    positions = SkyCoord(RA, DEC, unit='deg')
    size = (ly, lx)
    cutout = Cutout2D(hdu['SCI'].data, position=positions, size=size, wcs=wcs)
    hdu['SCI'].data = cutout.data
    hdu['SCI'].header.update(cutout.wcs.to_header())

    return cutout.data, hdu['SCI'].header



# OPEN CATALOG
CATALOG = '/home/alberto/cosmos/ista/COLA1/catalogs/C1F_stars.fits'
with fits.open(CATALOG) as hdul:
    orig_table = hdul[1].data
    orig_cols = orig_table.columns


IDlist = orig_table.field('NUMBER')
F200W = orig_table.field('F200W_AUTO_fnu')
RA = orig_table.field('ALPHA_J2000_det')
DEC = orig_table.field('DELTA_J2000_det')

med_F200W = np.nanmedian(F200W)

lx, ly = 200, 200


IMG_PATH = '/home/alberto/cosmos/ista/COLA1/images'

for thisfilt in [200]:
    directimage = f'{IMG_PATH}/cola1_F{thisfilt}W.fits'
    whtimage = f'{IMG_PATH}/cola1_F{thisfilt}W.fits'
    stack = []
    for q in range(len(IDlist)):
        thisRA = RA[q]
        thisDEC = DEC[q]
        thisF200W = F200W[q]
        multiply = med_F200W/thisF200W
        data, header = create_cutout(directimage, thisRA, thisDEC, lx, ly)
       # datatime,header=create_cutout(whtimage,RA,DEC,lx,ly)
        stack.append(data/multiply)

    medstack = np.nanmedian(stack, axis=0)


    ####### Now bootstrapping to compute errors
    # (Lazy version)
    N_boots = 100
    stack = []
    
    stacks_boots = []

    for jjj in range(N_boots):
        boot_IDs = np.random.choice(range(len(IDlist)), size=len(IDlist), replace=True)
        for q in boot_IDs:
            thisRA = RA[q]
            thisDEC = DEC[q]
            thisF200W = F200W[q]
            multiply = med_F200W/thisF200W
            data, header = create_cutout(directimage, thisRA, thisDEC, lx, ly)
            stack.append(data/multiply)
        
        stacks_boots.append(np.nanmean(stack, axis=0))


    errstack = np.nanstd(stacks_boots, axis=0)
    medstack = np.nanmean(stacks_boots, axis=0)


    fits.writeto(f'{IMG_PATH}/star_cutouts/C1F_F{thisfilt}W_star_stack.fits', medstack,
                 overwrite=True)
    fits.writeto(f'{IMG_PATH}/star_cutouts/C1F_F{thisfilt}W_star_stack_err.fits', errstack,
                 overwrite=True)
    # fits.writeto('C1F_F%sW_wht.fits'%thisfilt,datatime,header,overwrite=True)
