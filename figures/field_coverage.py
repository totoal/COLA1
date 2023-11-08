import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 12})

from PIL import Image
# Image.MAX_IMAGE_PIXELS = None

import numpy as np

from astropy.io import fits


############### FIGURE BEGINS #################

fig = plt.figure(figsize=(12, 12))

# ax0 is for the color image
ax0 = fig.add_subplot(211)

RGBFILE = '/home/alberto/cosmos/ista/COLA1/images/stiff_bin1_cola1.tif'

img = Image.open(RGBFILE)
img = np.asarray(img)
img = img[::-1, :, :]  # for some reason it flips it

ax0.imshow(img, origin='lower', aspect='equal')

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



# ax1 is for the error cube
ax1 = fig.add_subplot(212)

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

ax1.imshow(errplane_COLA1[::-1, :].T)


fig.subplots_adjust(hspace=0.05)

# Save it
savefig_path = '/home/alberto/cosmos/ista/COLA1/paper/figures'
fig.savefig(f'{savefig_path}/COLA1_field_coverage.pdf', bbox_inches='tight', pad_inches=0.05,
            # facecolor='w', dpi=1200)
            facecolor='w')
plt.close()