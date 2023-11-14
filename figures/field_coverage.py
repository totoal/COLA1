import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 12})

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np

from astropy.io import fits
from astropy.visualization import AsinhStretch


############### FIGURE BEGINS #################

fig, ax = plt.subplots(2, 1, figsize=(12, 9.5),
                     height_ratios=[2, 1])

[ax0, ax1] = ax


RGBFILE = '/home/alberto/cosmos/ista/COLA1/images/stiff_bin1_cola1.tif'

img = Image.open(RGBFILE)
img = np.asarray(img)
img = img[::-1, :, :]  # for some reason it flips it

ax0.imshow(img, origin='lower', aspect='equal', rasterized=True)

## Axes ticks
xtick_labels = np.arange(-250, 250 + 50, 50)
ytick_labels = np.arange(-100, 100 + 50, 50)
# Convert to px coordinates
# The Zero is set to the COLA1 coordinates
xtick0 = 8739.578
ytick0 = 3480.65
xticks = [x / 0.03 + xtick0 for x in xtick_labels]
yticks = [y / 0.03 + ytick0 for y in ytick_labels]
ax0.set_xticks(xticks)
ax0.set_yticks(yticks)
ax0.set_xticklabels(xtick_labels)
ax0.set_yticklabels(ytick_labels)

ax0.set_xlabel('Angular separation [arcsecond]')
ax0.set_ylabel('Angular separation [arcsecond]')


# Load the error cube
path_to_errorcube = '/home/alberto/cosmos/ista/COLA1/images/COLA1_spec_ERR_cube_v2.fits'
errcube = fits.open(path_to_errorcube)[0]
# Shape of errcube is: (1200, 430, 175)
# The spatial dimensions are (430, 175), dimensions of the image compressed 40 times
# Then, the coordinates of COLA1 in this frame are:
COLA1_errcube_posx = xtick0 / 40
COLA1_errcube_posy = ytick0 / 40
# We take a slice in the y axis at the position of COLA1
errplane_COLA1 = errcube.data[int(COLA1_errcube_posy)]
errplane_COLA1[errplane_COLA1 == 0] = np.nan

stretch = AsinhStretch(a=0.1)
min_stretched = stretch([np.nanmin(errplane_COLA1)])

cbar = ax1.imshow(stretch(errplane_COLA1[:, ::-1].T) / min_stretched, aspect='auto',
                  vmin=1, vmax=4, rasterized=True, cmap='magma')

# Color bar inside ax1
cbar_ax = ax1.inset_axes(bounds=[0.006, 0.05, 0.012, 0.9])
plt.colorbar(cbar, cax=cbar_ax)
# cbar_ax properties
# Plot params
cbar_ax.tick_params(direction='in', left=False,  right=True, labelsize=10)

# plt.colorbar(cbar, cax=ax_colorbar)

# The units of the wavelenth (y axis labels)
full_wave = 30000 + np.arange(1240) * 9.75
errwave = full_wave[20:-20]
ytick_labels = np.arange(32000, 40000 + 2000, 2000)
yticks = np.interp(ytick_labels, errwave, np.arange(1200)[::-1])
ax1.set_yticks(yticks)
ax1.set_yticklabels(ytick_labels / 10000)

ax1.set_ylabel(r'Wavelength [$\mu$m]')

# Secondary y axis
secax1 = ax1.secondary_yaxis('right')
secax1.set_yticks(yticks)
secax1.set_yticklabels([f'{ytick / 5008.24 - 1:0.1f}' for ytick in ytick_labels])
secax1.set_ylabel(r'[OIII]$_{\lambda 5008}$ Redshift')

# ax1 x axis labels
xtick_labels = np.arange(-250, 250 + 50, 50)
xticks = [x / 0.03 / 40 + COLA1_errcube_posx for x in xtick_labels]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)

# Y limits
ax1.set_ylim(np.interp([31500, 40000], errwave, np.arange(1200)[::-1]))

ax1.set_xlabel('Angular separation [arcsecond]')

fig.subplots_adjust(hspace=0.0)

# Save it
savefig_path = '/home/alberto/cosmos/ista/COLA1/paper/figures'
fig.savefig(f'{savefig_path}/COLA1_field_coverage.pdf', bbox_inches='tight', pad_inches=0.05,
            # facecolor='w')
            facecolor='w', dpi=200)
plt.close()