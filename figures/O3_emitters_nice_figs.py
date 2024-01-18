import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 13})

from scipy.ndimage import gaussian_filter

from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch

from grism.eiger import get_wav_array

# Load O3 C1F catalog
cat = fits.open('/home/alberto/cosmos/ista/COLA1/catalogs/COLA1_O3doublet_catalog_1.3.fits')[1].data

for iii, NUMBER in enumerate(cat['NUMBER']):
    NUMBER = int(NUMBER)
    print(f'Plotting NUMBER: {NUMBER}')

    path_to_spec = f'/home/alberto/cosmos/ista/COLA1/spectra/SPECTRA_O3_FINAL'
    hdul = fits.open(f'{path_to_spec}/stacked_2D_COLA1_{NUMBER}.fits')

    stretch = AsinhStretch(a=0.3)

    emline_stack = stretch(hdul['EMLINE'].data)
    emline_A = stretch(hdul['EMLINEA'].data)
    emline_B = stretch(hdul['EMLINEB'].data)

    wav_Arr = get_wav_array(f'{path_to_spec}/stacked_2D_COLA1_{NUMBER}.fits')
    z_O3 = cat['redshift'][iii]
    wav_0_Arr = wav_Arr / (1 + z_O3)

    aux = np.arange(emline_stack.shape[0])
    mask_spec_height = (aux > 10) & (aux < 40)

    emline_stack = emline_stack[mask_spec_height]
    emline_A = emline_A[mask_spec_height]
    emline_B = emline_B[mask_spec_height]

    # Apply Gaussian smoothing to images
    sigma = 0.4
    emline_stack = gaussian_filter(emline_stack, sigma)
    emline_A = gaussian_filter(emline_A, sigma)
    emline_B = gaussian_filter(emline_B, sigma)

    # The cutoff image
    cutoff_path = '/home/alberto/cosmos/ista/COLA1/images/O3_candidate_cutoffs'
    img = mpimg.imread(f'{cutoff_path}/C1F_O3_emitter_{NUMBER}.tiff')

    # Define limits for the plots
    mask_O3_wav = (wav_0_Arr > 4840) & (wav_0_Arr < 6000)

    spec_im_ratio = img.shape[0] / len(mask_spec_height)


    ### MAKE FIGURE ###
    fig = plt.figure(figsize=(14, 4.5))

    # In the first row, the stacked spectra
    gs = fig.add_gridspec(3, 2, width_ratios=[1, spec_im_ratio * 0.2],
                        height_ratios=[1, 1, 1])
    ax01 = fig.add_subplot(gs[0, 0])
    ax11 = fig.add_subplot(gs[1, 0])
    ax21 = fig.add_subplot(gs[2, 0])

    axim = fig.add_subplot(gs[:-1, 1])
    axtext = fig.add_subplot(gs[-1, 1])

    [vmin, vmax] = ZScaleInterval(contrast=0.8).get_limits(emline_stack.flatten())

    ax21.imshow(emline_stack[:, mask_O3_wav], vmax=vmax, vmin=vmin, rasterized=True)

    ax01.imshow(emline_A[:, mask_O3_wav], vmax=vmax, vmin=vmin, rasterized=True)

    ax11.imshow(emline_B[:, mask_O3_wav], vmax=vmax, vmin=vmin, rasterized=True)

    # Remove ax ticks
    for ax in [ax01, ax11]:
        ax.set_xticks([])
        ax.set_xticklabels([])
    for ax in [ax01, ax11, ax21]:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Mark the lines
    line_w0_list = [4960.295, 5008.240, 4862.68]
    line_x_list = [np.argmin(np.abs(wav_0_Arr[mask_O3_wav] - xtw0)) for xtw0 in line_w0_list]
    for ax in [ax01, ax11, ax21]:
        ax.vlines(x=line_x_list, ymax=sum(mask_spec_height) - 0.5, ymin=22,
                ls='--', color='r', lw=2)
    
    line_w0_list = [4364.436, 4341.68]

    xticks_w0 = np.arange(4850, 5050, 25)
    xticks = [np.argmin(np.abs(wav_0_Arr[mask_O3_wav] - xtw0)) for xtw0 in xticks_w0]
    ax21.set_xticks(xticks)
    ax21.set_xticklabels(xticks_w0)

    # Mark the image
    # Angular pixel size = 0.03"
    axim.vlines(x=img.shape[0] // 2 - 1, ymax=img.shape[1],
                ymin=60, color='white', ls='--', lw=2)
    axim.hlines(y=img.shape[1] // 2 - 1, xmax=img.shape[1],
                xmin=60, color='white', ls='--', lw=2)

    central_x = img.shape[0] // 2 - 1
    axim.set_yticks([central_x - 33, central_x, central_x + 33])
    axim.set_yticklabels([-1, 0, 1])

    # Spec axes limits
    rest_wav_min = 5000 - 10000 / (1 + z_O3)
    rest_wav_max = 5000 + 700 / (1 + z_O3)
    xmin = np.argmin(np.abs(wav_0_Arr[mask_O3_wav] - rest_wav_min))
    xmax = 170
    ax01.set_xlim(xmin, xmax)
    ax11.set_xlim(xmin, xmax)
    ax21.set_xlim(xmin, xmax)

    # Axes labels
    ax21.set_xlabel(r'Rest-frame wavelength [\AA]', fontsize=15)

    ax21.set_ylabel(r'\bf Combined', fontsize=14)
    ax01.set_ylabel(r'\bf Module A', fontsize=14)
    ax11.set_ylabel(r'\bf Module B', fontsize=14)

    axim.imshow(img, rasterized=True, interpolation='nearest')

    axim.set_xticks([])
    axim.set_xticklabels([])

    axim.set_ylabel('Angular separation [arcsec]', fontsize=15)
    axim.yaxis.set_label_position('right') 
    axim.yaxis.tick_right()

    gs.update(wspace=-0.275, hspace=0.05)


    # Add text with the line names
    line_dict1 = {
        r'[OIII]$_{\lambda 4960}$': 4960.295,
        r'[OIII]$_{\lambda 5008}$': 5008.240,
        r'H$\beta$': 4862.68,
    }
    for linename, wavelength in line_dict1.items():
        wav_idx = np.argmin(np.abs(wav_0_Arr[mask_O3_wav] - wavelength))
        ax01.text(wav_idx, -1, linename, fontsize=13,
                horizontalalignment='center', verticalalignment='bottom')

    # Add text box with basic properties
    this_EW = cat["EW_O3"][iii] + cat["EW_Hb"][iii]
    this_EW_err_up = (cat['EW_O3_84'][iii]**2 + cat['EW_Hb_84'][iii]**2)**0.5
    this_EW_err_down = (cat['EW_O3_16'][iii]**2 + cat['EW_Hb_16'][iii]**2)**0.5
    if np.isfinite(this_EW):
        EW_err_str = r'$^{+' + str(int(this_EW_err_up)) + r'}_{-' + str(int(this_EW_err_down)) + r'}$'
    else:
        EW_err_str = ''

    textbox = (f'NUMBER: {int(cat["NUMBER"][iii])}\n'
               r'$z_{\rm [OIII]} =$' + f' {cat["redshift"][iii]:0.3f}\n'
               r'$M_{\rm UV} =$' + f' {cat["muv"][iii]:0.2f}'
               '\n' + r'EW$_0$([OIII] + H$\beta$) = ' + f'{this_EW:0.0f}' + EW_err_str + r' \AA')

    axtext.text(0.1, 0, textbox,
                horizontalalignment='left',
                verticalalignment='bottom')
    axtext.set_axis_off()


    savefig_path = '/home/alberto/cosmos/ista/COLA1/images/O3_candidate_inspection_imgs'
    fig.savefig(f'{savefig_path}/spectra_{NUMBER}.png', bbox_inches='tight', pad_inches=0.1,
                facecolor='w')
    plt.close()