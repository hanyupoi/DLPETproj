from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale

import numpy as np
import matplotlib.pyplot as plt

# Iterative Reconstruction Algorithm -> solving step by step
# This code is explaining how the MLEM (Maximum Likelihood Expectation Maximization) uses
# statistical methods. It breaks down the problem of recon into smaller steps and starts
# by guessing the image space - iteratively adjusting the image to look like the ground truth
# repeat this until threshold reached
#
# Equation -> x^(k+1) = ((x^k) / A^T normiert) * A^T * (m/A*x^k)


plt.ion()
# controlling the radioactive concentration if it's a PET simulation (not necessary)
activity_level = 0.1
true_object = shepp_logan_phantom()

# true object just rescaled from 400x400 -> 200x200
true_object = rescale(activity_level * true_object, 0.5)

fig,axs = plt.subplots(2,3, figsize=(12,7))
axs[0,0].imshow(true_object, cmap='Greys_r')
axs[0,0].set_title('Object')

# generate simulated sinogram data
azi_angles = np.linspace(0.0, 180.0, 180, endpoint=False)   # 0-180 degree with 180 steps
sinogram = radon(true_object, azi_angles, circle=False)     # radon() doing forward projection, do this for several viewing angles
axs[0,1].imshow(sinogram.T, cmap='Greys_r')
axs[0,1].set_title('Sinogram')


# iteration zero (k=0)
mlem_recon = np.ones(true_object.shape) # give the image shape of the true object to define/guess the size
sino_ones = np.ones(sinogram.shape) #
sens_image = iradon(sino_ones, azi_angles, circle=False, filter_name=None)


# it recons in the beginning very fast but takes a long time to get a clear crisp image :(
for iter in range(100):
    fp = radon(mlem_recon, azi_angles, circle=False)    # forward projection of mlem_recon at iteration k -> A*x^k
    ratio = sinogram/ (fp + 0.000001)   # m / A * x^k to make sure it's not divided by zero
    correction = iradon(ratio, azi_angles, circle=False, filter_name=None) / sens_image
    # create the transpose of A because of no filter, if we would use a filter then it would create the inverse for example using in FBP
    # sens_image measures the sensitivity / how many back projected lines are impacting upon each pixel
    # line stands for A^T / A^T normiert -> A^T is a sinogram filled with ones

    axs[1,0].imshow(mlem_recon, cmap='Greys_r')
    axs[1,0].set_title('MLEM')

    axs[1,1].imshow(fp.T, cmap='Greys_r')
    axs[1,1].set_title('Forwad projection')

    axs[0,2].imshow(ratio.T, cmap='Greys_r')
    axs[0,2].set_title('Ratio Sinogram')


    mlem_recon = mlem_recon * correction

    axs[1,2].imshow(correction, cmap='Greys_r')
    axs[1,2].set_title('BP of ratio')

    axs[1,0].imshow(mlem_recon, cmap='Greys_r')
    axs[1,0].set_title('MLEM recon iteration=%d' %(iter+1))
    plt.show()
    plt.pause(0.5)

plt.show(block=True)
