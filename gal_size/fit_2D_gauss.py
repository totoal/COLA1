import numpy as np
import lmfit
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import stats



if __name__ == '__main__':
    star_stack_path = '/home/alberto/cosmos/ista/COLA1/images/star_cutouts/C1F_F200W_star_stack.fits'
    star_stack_err_path = '/home/alberto/cosmos/ista/COLA1/images/star_cutouts/C1F_F200W_star_stack_err.fits'

    stack_img = fits.open(star_stack_path)[0].data
    stack_err = fits.open(star_stack_err_path)[0].data

    model = lmfit.models.Gaussian2dModel()

    pixel_size = 0.03 # arcsec

    mesh_xxyy = np.meshgrid(np.arange(stack_img.shape[0]), np.arange(stack_img.shape[1]))
    xx = mesh_xxyy[0].flatten() * pixel_size
    yy = mesh_xxyy[1].flatten() * pixel_size

    params = model.guess(stack_img.flatten(), xx, yy)
    result = model.fit(stack_img.flatten(), x=xx, y=yy, params=params,)
                    #    weights=stack_err.flatten()**-2)

    lmfit.report_fit(result)

    
    # Plot to check the fits

    fig, ax = plt.subplots()

    plot_xx = np.arange(stack_img.shape[0]) * pixel_size

    ax.plot(plot_xx, stack_img[99, :])
    ax.plot(plot_xx,
            stats.norm.pdf(plot_xx, result.params['centerx'], result.params['sigmax'])
            / result.params['amplitude'])

    ax.set_xlim(2.5, 3.5)

    plt.show()


    fig, ax = plt.subplots()

    plot_xx = np.arange(stack_img.shape[1]) * pixel_size

    ax.plot(plot_xx, stack_img[:, 100])
    ax.plot(plot_xx,
            stats.norm.pdf(plot_xx, result.params['centery'], result.params['sigmay'])
            / result.params['amplitude'])

    ax.set_xlim(2.5, 3.5)

    plt.show()