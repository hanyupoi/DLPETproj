import numpy as np
from skimage.transform import radon, iradon, rescale
from skimage.data import shepp_logan_phantom
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def compute_anatomical_weights(anatomical_image, sigma):
    """
    Compute anatomical weights based on intensity differences in the anatomical image.
    """
    weights = np.zeros_like(anatomical_image)
    rows, cols = anatomical_image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [
                anatomical_image[i - 1, j], anatomical_image[i + 1, j],
                anatomical_image[i, j - 1], anatomical_image[i, j + 1]
            ]
            weights[i, j] = sum(
                np.exp(-((anatomical_image[i, j] - neighbor) ** 2) / (2 * sigma ** 2)) for neighbor in neighbors
            )
    return weights


def mlem_with_anatomical_regularization(sinogram, angles, num_iters, anatomical_weights, regularization_weight):
    """
    MLEM reconstruction with anatomical regularization.
    """
    # Initial guess
    reconstruction = np.ones((sinogram.shape[0], sinogram.shape[0]))
    sensitivity = iradon(np.ones_like(sinogram), theta=angles, circle=True, filter_name=None)

    for _ in range(num_iters):
        # Forward projection and ratio
        fp = radon(reconstruction, theta=angles, circle=True)
        fp = rescale(fp, sinogram.shape)  # Ensure matching shapes
        ratio = sinogram / (fp + 1e-8)  # Avoid division by zero
        correction = iradon(ratio, theta=angles, circle=True, filter_name=None) / sensitivity

        # Compute regularization term
        reg_term = regularization_weight * anatomical_weights * reconstruction

        # Update reconstruction with regularization
        reconstruction *= (correction + reg_term)

    return reconstruction


# Simulate the phantom and anatomical image
activity_level = 0.1
true_object = shepp_logan_phantom()
true_object = rescale(activity_level * true_object, (128, 128), anti_aliasing=True)

# Generate a simulated anatomical image by blurring the true object
anatomical_image = gaussian_filter(true_object, sigma=5)

# Compute anatomical weights
sigma = 0.1
anatomical_weights = compute_anatomical_weights(anatomical_image, sigma)

# Generate sinogram
angles = np.linspace(0.0, 180.0, 180, endpoint=False)
sinogram = radon(true_object, angles, circle=True)

print("1")

# Perform MLEM reconstruction without regularization
reconstruction_no_reg = mlem_with_anatomical_regularization(sinogram, angles, num_iters=10,
                                                            anatomical_weights=np.zeros_like(anatomical_image),
                                                            regularization_weight=0)
print("2")

# Perform MLEM reconstruction with anatomical regularization
regularization_weight = 0.01
reconstruction_with_reg = mlem_with_anatomical_regularization(sinogram, angles, num_iters=10,
                                                              anatomical_weights=anatomical_weights,
                                                              regularization_weight=regularization_weight)

print("3")

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(true_object, cmap='gray')
plt.title("Original Phantom")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(anatomical_image, cmap='gray')
plt.title("Simulated Anatomical Image")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(reconstruction_no_reg, cmap='gray')
plt.title("Reconstruction Without Regularization")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(reconstruction_with_reg, cmap='gray')
plt.title("Reconstruction With Regularization")
plt.axis("off")

plt.tight_layout()
plt.show()
