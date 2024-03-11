import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 12})

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord


px_size = 0.03 # arcsec / px

x_COLA1 = 8739.578
y_COLA1 = 3480.65

# Load doublet and singlet catalogs
cat_d = fits.open('/home/alberto/cosmos/COLA1/catalogs/C1F_O3doublet_catalog_06022024.fits')[1].data
cat_s = fits.open('/home/alberto/cosmos/COLA1/catalogs/SINGLESEARCH.fits')[1].data

# Mask catalogs on redshift
mask_d = (cat_d['redshift'] > 5.7) & (cat_d['redshift'] < 6.7)
cat_d = cat_d[mask_d]
mask_s = (cat_s['z_O3_5008'] > 5.7) & (cat_s['z_O3_5008'] < 6.7)
cat_s = cat_s[mask_s]

# Get the WCS from one of the photometric images
wcs = WCS(fits.open('/home/alberto/cosmos/COLA1/images/cola1_F200W.fits')[0])
coords_d = SkyCoord(cat_d['ALPHA_J2000_det'], cat_d['DELTA_J2000_det'],
                    unit='deg')
x_d, y_d = skycoord_to_pixel(coords_d, wcs)
coords_s = SkyCoord(cat_s['RA'], cat_s['DEC'],
                    unit='deg')
x_s, y_s = skycoord_to_pixel(coords_s, wcs)


############### FIGURE BEGINS #################

fig, ax = plt.subplots()


RGBFILE = '/home/alberto/portatilHP/cosmos/ista/COLA1/images/stiff_bin1_cola1.tif'

# Load COSMOS2020 catalog
cosmos_cat = fits.open('/home/alberto/cosmos/FLAMES/COSMOS2020_CLASSIC_R1_v2.2_p3.fits')
mask_radec = ((cosmos_cat[1].data['ALPHA_J2000'] - 150.6474170147586)**2
              + (cosmos_cat[1].data['DELTA_J2000'] - 2.203869112743768)**2)**0.5 < 12.5/60
mask_zphot = (cosmos_cat[1].data['ez_z_phot'] > 5.7) & (cosmos_cat[1].data['ez_z_phot'] < 6.7)
cosmos_RA = cosmos_cat[1].data['ALPHA_J2000'][mask_radec & mask_zphot]
cosmos_DEC = cosmos_cat[1].data['DELTA_J2000'][mask_radec & mask_zphot]

img = Image.open(RGBFILE)
img = np.asarray(img)
img = img[::-1, :, :]  # for some reason it flips it

ax.imshow(img, origin='lower', aspect='equal', rasterized=True, zorder=9)


# Mark COSMOS2020 objects
marker_radius = 11 / px_size
for ra, dec in zip(cosmos_RA, cosmos_DEC):
    coords = SkyCoord(ra, dec, unit='deg')
    xx, yy = skycoord_to_pixel(coords, wcs)
    circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
                            lw=0.7, ec='g')
    ax.add_patch(circle_pos)

for xx, yy in zip(x_d, y_d):
    circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
                            lw=0.7, ec='r', zorder=99)
    ax.add_patch(circle_pos)
for xx, yy in zip(x_s, y_s):
    circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
                            lw=0.7, ec='r', zorder=99)
    ax.add_patch(circle_pos)


# COLA1 od
# for iii, (xx, yy) in enumerate(zip(x_d, y_d)):
#     if (cat_d['redshift'][iii] < 6.56) | (cat_d['redshift'][iii] > 6.63):
#         continue
#     circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
#                             lw=0.7, ec='aqua', zorder=99)
#     ax.add_patch(circle_pos)
# for iii, (xx, yy) in enumerate(zip(x_s, y_s)):
#     if (cat_s['z_O3_5008'][iii] < 6.56) | (cat_s['z_O3_5008'][iii] > 6.63):
#         continue
#     circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
#                             lw=0.7, ec='aqua', zorder=99)
#     ax.add_patch(circle_pos)

# Plot circle showing the instrument FoV of diameter 25 arcmim
fov_radius_px = 25 * 0.5 * 60 / px_size
circle_fov = plt.Circle((x_COLA1, y_COLA1), fov_radius_px, fill=False,
                        lw=2, ec='r')
ax.add_patch(circle_fov)

# Add text
ax.text(-16000, -23000,
        r'{\bf FLAMES FoV}' + '\n' + r'\bf{diameter $\mathbf{\sim 25^\prime}$}',
        fontsize=9, horizontalalignment='left', verticalalignment='bottom',
        color='r')


ax.set_ylim(y_COLA1 - fov_radius_px - 500, y_COLA1 + fov_radius_px + 500)
ax.set_xlim(x_COLA1 - fov_radius_px - 500, x_COLA1 + fov_radius_px + 500)

ax.set_aspect('equal')

ax.set_xticks([])
ax.set_yticks([])

ax.plot([], [], marker='o', mew=0.8, ls='', mec='r',
        mfc='none', label='C1F [OIII] emitters')
ax.plot([], [], marker='o', mew=0.8, ls='', mec='g',
        mfc='none', label=r'COSMOS2020')

ax.legend(fontsize=7, framealpha=1, loc='upper right')


ax.axis('off')


# Save it
savefig_path = '/home/alberto/cosmos/FLAMES/figures'
fig.savefig(f'{savefig_path}/C1F_sources_MEDUSA_FoV.pdf', bbox_inches='tight', pad_inches=0.05,
            # facecolor='w')
            facecolor='w', dpi=600)
plt.close()



########## SECOND FIGURE ############

fig, ax = plt.subplots()


RGBFILE = '/home/alberto/portatilHP/cosmos/ista/COLA1/images/stiff_bin1_cola1.tif'

img = Image.open(RGBFILE)
img = np.asarray(img)
img = img[::-1, :, :]  # for some reason it flips it

ax.imshow(img, origin='lower', aspect='equal', rasterized=True)

# Mark OIII emitters
marker_radius = 11 / px_size
for xx, yy in zip(x_d, y_d):
    circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
                            lw=0.8, ec='r')
    ax.add_patch(circle_pos)
for xx, yy in zip(x_s, y_s):
    circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
                            lw=0.8, ec='r')
    ax.add_patch(circle_pos)


# COLA1 od
for iii, (xx, yy) in enumerate(zip(x_d, y_d)):
    if (cat_d['redshift'][iii] < 6.56) | (cat_d['redshift'][iii] > 6.63):
        continue
    circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
                            lw=0.8, ec='aqua')
    ax.add_patch(circle_pos)
for iii, (xx, yy) in enumerate(zip(x_s, y_s)):
    if (cat_s['z_O3_5008'][iii] < 6.56) | (cat_s['z_O3_5008'][iii] > 6.63):
        continue
    circle_pos = plt.Circle((xx, yy), marker_radius, fill=False,
                            lw=0.8, ec='aqua')
    ax.add_patch(circle_pos)

# Dummies for the legend
ax.plot([], [], marker='o', mew=0.8, ls='', mec='r',
        mfc='none', label='All')
ax.plot([], [], marker='o', mew=0.8, ls='', mec='aqua',
        mfc='none', label=r'$z \sim 6.6$')
ax.legend(fontsize=11, ncol=2, loc='upper left', bbox_to_anchor=(0., 1.05, 1., .102),
          frameon=False)

# Mark 1 arcmin
ax.errorbar(13400, 350, xerr=60/px_size,
            ls='', fmt='', ecolor='yellow', elinewidth=2,
            capsize=2.5, capthick=2)
ax.text(13400, 450, r'{\bf 1$\mathbf{^\prime}$}', color='yellow',
        verticalalignment='bottom', horizontalalignment='center')

ax.set_yticks([])
ax.set_xticks([])

# Save it
savefig_path = '/home/alberto/cosmos/FLAMES/figures'
fig.savefig(f'{savefig_path}/C1F_sources_MEDUSA_C1F_zoom.pdf', bbox_inches='tight', pad_inches=0.05,
            facecolor='w', dpi=600)