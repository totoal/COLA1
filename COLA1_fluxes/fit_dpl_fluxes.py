import sys
sys.path.insert(0, '..')

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 16})

import corner

from scipy import linalg

from astropy.io import fits

from multiprocessing import Pool
from autoemcee import ReactiveAffineInvariantSampler


##########################

def double_power_law(wav, norm, wav_break, beta1, beta2):
    exp1 = (wav_break - wav) * (beta1 - 1)
    exp2 = (wav_break - wav) * (beta2 - 1)

    return norm * (10. ** exp1 + 10. ** exp2)

# The fitting curve
def dpl_fit(*args):
    # return np.log10(double_power_law(*args))
    return double_power_law(*args)

################################################
### A set of functions to compute MCMC stuff ###
################################################

def chi2_fullmatrix(data_vals, inv_covmat, model_predictions):
    """
    Given a set of data points, its inverse covariance matrix and the
    corresponding set of model predictions, computes the standard chi^2
    statistic (using the full covariances)
    """

    y_diff = data_vals - model_predictions
    return np.dot(y_diff, np.dot(inv_covmat, y_diff))

################################################
################################################


def transform(theta):
    '''
    Transform features to match the priors
    '''
    theta_trans = np.empty_like(theta)
    
    # Flat Priors
    norm_range = [50, 100]
    wav_break_range = [1, 3]
    beta1_range = [-3, 1]
    beta2_range = [1, 4]

    theta_trans[0] = norm_range[0] + (norm_range[1] - norm_range[0]) * theta[0]
    theta_trans[1] = wav_break_range[0] + (wav_break_range[1] - wav_break_range[0]) * theta[1]
    theta_trans[2] = beta1_range[0] + (beta1_range[1] - beta1_range[0]) * theta[2]
    theta_trans[3] = beta2_range[0] + (beta2_range[1] - beta2_range[0]) * theta[3]

    return theta_trans

# Main function
def run_mcmc_fit():
    cat = fits.open('COLA1_O3_fitted_flux.fits')[1].data
    src = cat['NUMBER_1'] == 9269

    fluxes = np.asarray([cat['F115W_AUTO_fnu'][src],
                        cat['F150W_AUTO_fnu'][src],
                        cat['F200W_AUTO_fnu'][src]]).flatten()
    wavs = np.asarray([1.154, 1.501, 1.990])
    errs = np.asarray([cat['F115W_AUTO_enu'][src],
                       cat['F150W_AUTO_enu'][src],
                       cat['F200W_AUTO_enu'][src]]).flatten()

    # Define the name of the fit parameters
    paramnames = ['norm', 'wav_break', 'beta1', 'beta2']

    def log_like(theta):
        norm_0 = theta[0]
        wav_break_0 = theta[1]
        beta1_0 = theta[2]
        beta2_0 = theta[3]

        model_Arr = dpl_fit(wavs, norm_0, wav_break_0,
                            beta1_0, beta2_0)

        covmat = np.eye(3) * errs**2
        invcovmat = linalg.inv(covmat)

        chi2 = chi2_fullmatrix(fluxes, invcovmat, model_Arr)

        return -0.5 * chi2


    # Define the sampler
    sampler = ReactiveAffineInvariantSampler(paramnames,
                                             log_like,
                                             transform=transform)
    # Run the sampler
    sampler.run(max_ncalls=1e7, progress=False)
    # Print the results
    sampler.print_results()

    # Plot results
    fig = corner.corner(sampler.results['samples'], labels=paramnames,
                        show_titles=True, truths=sampler.results['posterior']['median'])
    fig.savefig(f'dpl_corner.pdf', pad_inches=0.1,
                bbox_inches='tight', facecolor='w')
    plt.close()

    
    # Save the chain
    flat_samples = sampler.results['samples']
    np.save(f'mcmc_dpl_fit_chain', flat_samples)

    # Obtain the fit parameters
    fit_params = sampler.results['posterior']['median']
    fit_params_perc84 = np.percentile(flat_samples, [84], axis=0)[0]
    fit_params_perc16 = np.percentile(flat_samples, [16], axis=0)[0]
    fit_params_err_up = fit_params_perc84 - fit_params
    fit_params_err_down = fit_params - fit_params_perc16

    return fit_params, fit_params_err_up, fit_params_err_down




if __name__ == '__main__':
    run_mcmc_fit()