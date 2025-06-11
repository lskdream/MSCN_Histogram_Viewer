# calc_mscn.py
# Compute MSCN (Mean Subtracted Contrast Normalized) coefficients and plot histogram

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal

# Gaussian kernel generator
def gaussian_kernel2d(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# Local mean computation
def local_mean(image, kernel):
    return scipy.signal.convolve2d(image, kernel, 'same', boundary='symm')

# Local deviation computation
def local_deviation(image, local_mu, kernel):
    sigma_sq = scipy.signal.convolve2d(image**2, kernel, 'same', boundary='symm')
    return np.sqrt(np.abs(sigma_sq - local_mu**2))

# Compute MSCN coefficients
def compute_mscn(image):
    kernel = gaussian_kernel2d(7, 7 / 6)
    mu = local_mean(image, kernel)
    sigma = local_deviation(image, mu, kernel)
    mscn = (image - mu) / (sigma + 1e-8)
    return mscn

# Plot MSCN histogram
def plot_histogram(mscn, bins=100):
    plt.figure(figsize=(6,4))
    plt.hist(mscn.ravel(), bins=bins, density=True, alpha=0.7, color='blue')
    plt.title('MSCN Coefficient Histogram')
    plt.xlabel('MSCN value')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute MSCN and plot histogram')
    parser.add_argument('--img', type=str, required=True, help='Path to grayscale image')
    args = parser.parse_args()

    # Load image in grayscale
    image = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {args.img}")
    image = image.astype(np.float32) / 255.0

    mscn = compute_mscn(image)
    plot_histogram(mscn)
