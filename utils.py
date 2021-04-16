import math

import numpy as np
import torch


def load_num_batches(loader, num_batches):
    """A generator that returns num_batches batches from the loader, irrespective of the length
    of the dataset."""
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return


def nats_to_bits_per_dim(nats, c, h, w):
    """Convert negative log likelihood in nats to bits per dimension for the given number of
    channels and image size."""
    return nats / (math.log(2) * c * h * w)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


class Preprocess:
    """A pre-processing transformation that converts the pixel values to the expected range, re-samples 
    the data to a different quantization precision if needed, and dequantizes the data with uniform dequantization."""
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_bins = 2 ** self.num_bits

    def __call__(self, img):
        if img.dtype == torch.uint8:
            img = img.float()  # Already in [0,255]
        else:
            img = img * 255.  # [0,1] -> [0,255]

        if self.num_bits != 8:
            img = torch.floor(img / 2 ** (8 - self.num_bits))  # [0, 255] -> [0, num_bins - 1]

        # Uniform dequantization.
        img = img + torch.rand_like(img)

        return img

    def inverse(self, inputs):
        # Discretize the pixel values.
        inputs = torch.floor(inputs)
        # Convert to a float in [0, 1].
        inputs = inputs * (256 / self.num_bins) / 255
        inputs = torch.clamp(inputs, 0, 1)
        return inputs


def pca(cov):
    """Compute eigenvectors and eigenvalues of the given covariance matrix, and return both sorted
    by the eigenvalue, largest to smallest."""
    pca_w, pca_v = np.linalg.eig(cov)
    idx = np.argsort(pca_w)[::-1]
    pca_w = pca_w[idx]
    pca_v = pca_v[:, idx]
    return pca_w, pca_v
