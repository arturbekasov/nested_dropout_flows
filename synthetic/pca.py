import argparse

import numpy as np

from synthetic.data import generate_data, TRAIN_SEED, VALIDATION_SEED
from utils import pca

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_points', type=int, default=10000)
    parser.add_argument('--num_eval_points', type=int, default=10000)

    args = parser.parse_args()

    train_data = generate_data(num_points=args.num_train_points,
                               seed=TRAIN_SEED)
    eval_data = generate_data(num_points=args.num_eval_points,
                              seed=VALIDATION_SEED)

    estimated_cov = np.cov(train_data, rowvar=False)
    pca_w, pca_v = pca(estimated_cov)

    for keep in [3, 2, 1]:
        pca_v = pca_v[:, :keep]

        # Project and re-construct
        val_data_proj = eval_data @ pca_v
        val_data_reconstr = val_data_proj @ pca_v.T

        mse = np.mean((eval_data - val_data_reconstr) ** 2)
        print(f"mse ({keep}) {mse:.3f}")
