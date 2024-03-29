import numpy as np
import lmfit
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal



def f(x):
    return x


if __name__ == '__main__':
    # star_stack_path = '/home/alberto/cosmos/COLA1/images/star_cutouts/C1F_F200W_star_stack.fits'
    # star_stack_err_path = '/home/alberto/cosmos/COLA1/images/star_cutouts/C1F_F200W_star_stack_err.fits'

    IMG_PATH = '/home/alberto/cosmos/COLA1/images'
    stacks_boots = np.load(f'{IMG_PATH}/star_cutouts/C1F_F200W_star_stack_list.npy')
    stacks_boots_std = np.load(f'{IMG_PATH}/star_cutouts/C1F_F200W_star_stack_std_list.npy')

    # stack_img = fits.open(star_stack_path)[0].data
    # stack_err = fits.open(star_stack_err_path)[0].data
    sigma_list = []
    for stack_img, stack_err in zip(stacks_boots, stacks_boots_std):
        boxside = 15
        
        mask_center = np.zeros_like(stack_img).astype(bool)
        mask_center[99 - boxside : 99 + boxside, 99 - boxside : 99 + boxside] = True

        stack_img = stack_img[mask_center].reshape(boxside*2, boxside*2)
        stack_err = stack_err[mask_center].reshape(boxside*2, boxside*2)

        # Virtually mask regions where flux is too low
        mask_low_flx = stack_img 

        img_var = stack_err.flatten() ** 2


        model = lmfit.models.Gaussian2dModel()

        pixel_size = 0.03 # arcsec

        mesh_xxyy = np.meshgrid(np.arange(stack_img.shape[0]),
                                np.arange(stack_img.shape[1]))
        xx = mesh_xxyy[0].flatten() * pixel_size
        yy = mesh_xxyy[1].flatten() * pixel_size

    #     model.set_param_hint('centerx', min=2.95, max=3.05)
    #     model.set_param_hint('centery', min=2.95, max=3.05)
    #     model.set_param_hint('sigmay', min=0.01, max=0.1)
    #     model.set_param_hint('sigmax', min=0.01, max=0.1)

    #     params = model.guess(stack_img.flatten(), xx, yy)
        params = lmfit.Parameters()
        params.add('centerx', value=0.8, min=0.0, max=2)
        params.add('centery', value=0.8, min=0.0, max=2)
        params.add('sigmax', value=0.04, min=0.001, max=0.1)
        params.add('sigmay', value=0.04, min=0.001, max=0.1)
        params.add('amplitude', value=1, min=0, max=100)

        result = model.fit(stack_img.flatten(), x=xx, y=yy, params=params,)
                        #    weights=img_var**-1)

        lmfit.report_fit(result)

        # Plot to check the fits
    #     fig, ax = plt.subplots()

    #     plot_xx = np.arange(stack_img.shape[0]) * pixel_size

    #     ax.errorbar(plot_xx, stack_img[stack_img.shape[1]//2, :], yerr=stack_err[stack_img.shape[1]//2, :])
    #     ax.plot(plot_xx,
    #             stats.norm.pdf(plot_xx, result.params['centerx'], result.params['sigmax'])
    #             * (result.params['amplitude'] / ((2 * np.pi) ** 0.5 * result.params['sigmax'])))

    # #     ax.set_xlim(2.5, 3.5)

    #     plt.show()


    #     fig, ax = plt.subplots()

    #     plot_xx = np.arange(stack_img.shape[1]) * pixel_size

    #     ax.errorbar(plot_xx, stack_img[:, stack_img.shape[1]//2+1], yerr=stack_err[:, stack_img.shape[1]//2+1])
    #     ax.plot(plot_xx,
    #             stats.norm.pdf(plot_xx, result.params['centery'], result.params['sigmay'])
    #             * (result.params['amplitude'] / ((2 * np.pi) ** 0.5 * result.params['sigmay'])))

    # #     ax.set_xlim(2.5, 3.5)

    #     plt.show()



    #     plt.imshow(np.log(stack_img))
    #     plt.colorbar()

    #     plot_xx = np.linspace(xx.min(), xx.max(), 1000)
    #     x, y = np.meshgrid(plot_xx, plot_xx)
    #     pos = np.dstack([x, y])
    #     rv = multivariate_normal(mean=[result.params['centerx'].value, result.params['centery'].value],
    #                             cov=[[result.params['sigmax'].value ** 2, 0],
    #                                 [0, result.params['sigmay'].value ** 2]])
    #     Z = rv.pdf(pos)
    #     levels = 1 - np.array([0.5, 0.68, 0.95, 0.99])
    #     levels.sort()
    #     plt.contour(x / pixel_size, y / pixel_size, Z, levels=levels, colors='k', zorder=99)

    #     plt.show()

        sigma_list.append((result.params['sigmax'].value + result.params['sigmax'].value) * 0.5)

    print(np.nanmedian(sigma_list))
    print(np.nanpercentile(sigma_list, [16, 84]) - np.nanmedian(sigma_list))