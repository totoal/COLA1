import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 12})

from astropy.io import fits

import numpy as np


w_Lya = 1215.67

f_Lya = 5.9 * 1e-17 # erg/s/cm2
f_O3_5008_A = 24.60 # erg/s/cm2
f_O3_5008_B = 26.75 # erg/s/cm2

# Measured from grism module A
z_Lya_sys = 6.59165

# Load Lya line.
'''
>>> hdul[1].columns
ColDefs(
    name = 'vacuum_wavelength'; format = 'E'
    name = 'fluxdensity'; format = 'E'
    name = 'fluxdensity_err'; format = 'E'
    name = 'lambda_0_Lya'; format = 'E'
    name = 'velocity_Lya'; format = 'E'
)
'''
Lya_spec = fits.open('/home/alberto/cosmos/ista/COLA1/spectra/1D_COLA1_2020_v2_newest.fits')[1].data

# Load O3
COLA1_fluxes_dir = '/home/alberto/cosmos/ista/COLA1/code/COLA1_fluxes'
O3_5008_wav = np.load(f'{COLA1_fluxes_dir}/COLA1_9269_5008_modA_obswav.npy')
O3_5008_flx_A = np.load(f'{COLA1_fluxes_dir}/COLA1_9269_5008_modA_fluxtot.npy')
O3_5008_flx_B = np.load(f'{COLA1_fluxes_dir}/COLA1_9269_5008_modB_fluxtot.npy')

fig, ax = plt.subplots(figsize=(6, 4))

# Define relative wavelength limits
wav_lim_low, wav_lim_high = -2.1, 2.5


## Plot the Lya profile

# Define wavelength relative to the systemic Lya wav
Lya_wav_rel = Lya_spec['vacuum_wavelength'] / (1 + z_Lya_sys) - w_Lya
# Mask wavelength
mask_Lya_wav = (Lya_wav_rel > wav_lim_low - 2) & (Lya_wav_rel < wav_lim_high + 2)

norm = 1e-3 * (1 + z_Lya_sys)

ax.plot(Lya_wav_rel[mask_Lya_wav], Lya_spec['fluxdensity'][mask_Lya_wav] * norm,
        c='darkslategray', zorder=100, drawstyle='steps-mid')
ax.fill_between(Lya_wav_rel[mask_Lya_wav],
                (Lya_spec['fluxdensity'] + Lya_spec['fluxdensity_err'])[mask_Lya_wav] * norm,
                (Lya_spec['fluxdensity'] - Lya_spec['fluxdensity_err'])[mask_Lya_wav] * norm,
                color='darkslategray', alpha=0.35, zorder=99, lw=0,
                step='mid')

# Draw vertical lines in the redshifts
# ModA
z_O3_5008_A = 6.591601
z_O3_4960_A = 6.591655
z_Hbeta_A = 6.591662

# ModB
z_O3_5008_B = 6.590421
z_O3_4960_B = 6.590137
z_Hbeta_B = 6.589786


# Line styles
ax.axvline(w_Lya * ((1 + z_O3_5008_A) / (1 + z_Lya_sys) - 1),
          lw=1.5, ls='--', color='C0', zorder=53)
ax.axvline(w_Lya * ((1 + z_O3_4960_A) / (1 + z_Lya_sys) - 1),
          lw=1.5, ls='--', color='C1', zorder=51)
ax.axvline(w_Lya * ((1 + z_Hbeta_A) / (1 + z_Lya_sys) - 1),
          lw=1.5, ls='--', color='C2', zorder=52)

ax.axvline(w_Lya * ((1 + z_O3_5008_B) / (1 + z_Lya_sys) - 1),
          lw=1.5, ls='-.', color='C0', zorder=53)
ax.axvline(w_Lya * ((1 + z_O3_4960_B) / (1 + z_Lya_sys) - 1),
          lw=1.5, ls='-.', color='C1', zorder=52)
ax.axvline(w_Lya * ((1 + z_Hbeta_B) / (1 + z_Lya_sys) - 1),
          lw=1.5, ls='-.', color='C2', zorder=51)

# Dummy lines for legend
ax.axvline(100, color='C0', lw=2, label=r'[OIII]$_{\lambda 5008}$')
ax.axvline(100, color='C1', lw=2, label=r'[OIII]$_{\lambda 4960}$')
ax.axvline(100, color='C2', lw=2, label=r'H$\beta$')
ax.axvline(100, color='dimgray', lw=1.5, ls='--', label='Module A')
ax.axvline(100, color='dimgray', lw=1.5, ls='-.', label='Module B')


# Draw zero horizontal line
ax.axhline(0, c='k', ls='-', zorder=-999, lw=1)

# Legend
ax.legend(fontsize=10, markerfirst=True, alignment='right',
          title='Grism line centroids', title_fontsize=11.5,
          frameon=False, loc=1)


ax.set_xlim(wav_lim_low, wav_lim_high)

# Axes ticks
xticklabels = np.arange(-500, 600 + 100, 100).astype(int)
xticks = np.interp(xticklabels, Lya_wav_rel / w_Lya * 299792.458, Lya_wav_rel)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

# Axes labels
ax.set_xlabel(r'$\Delta v$ [km\,s$^{-1}$]', fontsize=15)
ax.set_ylabel(r'Flux [$10^{-17}$\,erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$]', fontsize=15)

savefig_path = '/home/alberto/cosmos/ista/COLA1/paper/figures'
fig.savefig(f'{savefig_path}/COLA1_Lya_profile.pdf', bbox_inches='tight', pad_inches=0.05,
            facecolor='w')
fig.clf()