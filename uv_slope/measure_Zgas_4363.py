import pyneb as pn
import matplotlib
from matplotlib import pyplot
import numpy
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

pyplot.rcParams['xtick.labelsize'] = 17
pyplot.rcParams['ytick.labelsize'] = 17
pyplot.rcParams['axes.labelsize'] = 16
pyplot.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['xtick.minor.size'] = 4
matplotlib.rcParams['xtick.major.width'] = 2.
matplotlib.rcParams['xtick.minor.width'] = 1.5
matplotlib.rcParams['ytick.major.size'] = 6
matplotlib.rcParams['ytick.minor.size'] = 4
matplotlib.rcParams['ytick.major.width'] = 2.
matplotlib.rcParams['ytick.minor.width'] = 1.5
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams.update({'font.size': 16, 'font.family': 'serif',
                           'mathtext.fontset': 'cm'})  # change for diff style


# # EIGER II  stack (fluxes in 1E42 erg/s)
# Hb = 1.01461
# Hb_err = 0.066013
# O3_b = 2.0126126961241386
# O3_b_err = 0.123618342265509
# O3_r = 6.1094564680632
# O3_r_err = 0.357493712526215

# O3 = 0.5*(2.98*O3_b + O3_r)
# O3_err = 0.5*((2.98*O3_b_err)**2 + O3_r**2)**0.5

# Hg = 0.5295767730417912
# Hg_err = 0.05097596923768

# O3_4363 = 0.23450238922051167
# O3_4363_err = 0.0435800316973457

# COLA1 fluxes
Hb = 4.102173791187896
Hb_err = 0.16479177254799196
O3_b = 8.485066991520519
O3_b_err = 0.20556481443953625
O3_r = 25.676083480108232
O3_r_err = 0.38816374682561117

O3 = 0.5*(2.98*O3_b + O3_r)
O3_err = 0.5*((2.98*O3_b_err)**2 + O3_r**2)**0.5

Hg = 2.117971605396823
Hg_err = 0.20597044573279671

O3_4363 = 0.755969222734515
O3_4363_err = 0.24577417533350124


HgHb = []
for q in range(1000):
    thisHb = numpy.random.normal(Hb, Hb_err)
    thisHg = numpy.random.normal(Hg, Hg_err)
    HgHb.append(thisHg/thisHb)

print('Balmer decrement, Hg/Hb', numpy.nanmedian(HgHb), -
      (numpy.nanmedian(HgHb)-numpy.nanpercentile(HgHb, [16, 84])))

Te = []
OH = []

O3atom = pn.Atom('O', 3)
H1atom = pn.RecAtom('H', 1)


# T=np.arange(0.1,3.5,0.1)*1E4
# O3em=O3atom.getEmissivity(T,300,wave=5007)
# Hbem=H1atom.getEmissivity(T,300,wave=4860)
# Haem=H1atom.getEmissivity(T,300,wave=4340)
# #pyplot.plot(T,numpy.log10(O3em),color='red')
# #pyplot.plot(T,numpy.log10(Hbem),color='blue')
# pyplot.plot(T,Haem/Hbem,color='k')
# pyplot.show()
# stop


for q in range(1000):
    thisO3 = numpy.random.normal(O3, O3_err)
    thisO3_4363 = numpy.random.normal(O3_4363, O3_4363_err)
    thisHb = numpy.random.normal(Hb, Hb_err)

    thisO32 = numpy.random.normal(10., 3)  # an assumption of the O32 ratio
    int_ratio = thisO3/thisO3_4363

    # assume density ne=300 cm-3
    T_1 = O3atom.getTemDen(int_ratio=int_ratio, den=300.,
                           wave1=5007, wave2=4363) / 1E4
    T_2 = -0.577+T_1*(2.065-0.498*T_1)

    OH_O3 = numpy.log10(1.25*thisO3/thisHb) + 6.2 + 1.251 / \
        T_1 - 0.55*numpy.log10(T_1) - 0.014*T_1
    OH_O2 = numpy.log10(1.25*thisO3/(thisHb*thisO32)) + 5.961 + 1.676/T_2 - \
        0.4*numpy.log10(T_2) - 0.034*T_2 + \
        numpy.log10(1+1.35*300*1E-4*T_2**-0.5)

    SUM = numpy.log10(10**(OH_O3-12) + 10**(OH_O2-12))+12.

    Te.append(T_1)
    OH.append(SUM)


print('Electron temperature', numpy.nanmedian(Te), -
      (numpy.nanmedian(Te)-numpy.nanpercentile(Te, [16, 84])))
print('Direct method metallicity', numpy.nanmedian(OH), -
      (numpy.nanmedian(OH)-numpy.nanpercentile(OH, [16, 84])))

# DONE


# """
# Sample PyNeb script
# Plots the [O III] 4363/5007 ratio as a function of Te for several Ne values
# """

# # Set high verbosity level to keep track of atom creation
# pn.log_.level = 1 # Set to 3 if you want all the atoms to be printed out

# # Create a collection of atoms - a bit overkill if we just need O III
# adict = pn.getAtomDict()

# # Lower verbosity level
# pn.log_.level = 2

# # Function to compute line ratio
# def line_ratio(atom, wave1, wave2, tem, den):
#     emis1 = adict[atom].getEmissivity(tem, den, wave = wave1)
#     emis2 = adict[atom].getEmissivity(tem, den, wave = wave2)
#     return emis1 / emis2

# # Define array of Te
# tem = np.arange(5000, 18000, 30)

# # Plot
# plt.figure(1)
# for den in [1e2, 1e3, 1e4, 1e5]:
#     plt.semilogy(tem, line_ratio('O3', 4363, 5007, tem, den), label = 'Ne={0:.0e}'.format(den))
# plt.xlabel('T$_e$ [K]')
# plt.ylabel(r'[OIII] 4363/5007 $\AA$')
# plt.legend(loc=2)

# plt.show()
