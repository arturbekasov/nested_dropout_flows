import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from synthetic.data import generate_data, TRAIN_SEED, VALIDATION_SEED
from synthetic.flow import create_flow
from utils import tensor2numpy, pca

rc_params = {
    'text.usetex': True,
    'font.size': 10,
}
matplotlib.rcParams.update(rc_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_ckpt', type=str, required=True)
    parser.add_argument('-o', '--output_pdf', type=str)
    parser.add_argument('--flow_type', type=str, choices=['qr_flow', 'lu_flow'], default='qr_flow')
    parser.add_argument('--num_train_points', type=int, default=10000)
    parser.add_argument('--num_eval_points', type=int, default=10000)

    args = parser.parse_args()

    train_data = generate_data(num_points=args.num_train_points,
                               seed=TRAIN_SEED)
    eval_data = generate_data(num_points=args.num_eval_points,
                              seed=VALIDATION_SEED)

    flow = create_flow(args.flow_type)
    flow.load_state_dict(torch.load(args.input_ckpt))

    flow.eval()

    estimated_cov = np.cov(train_data, rowvar=False)
    _, pca_v = pca(estimated_cov)
    c = eval_data @ pca_v[:, 0]

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')

    with torch.no_grad():
        proj = flow.transform_to_noise(torch.FloatTensor(eval_data))
        proj = tensor2numpy(proj)
    s = ax.scatter(proj[:, 0], proj[:, 1], c=c.flat, alpha=0.3)
    s.set_rasterized(True)

    if args.output_pdf:
        fig.tight_layout()
        fig.savefig(args.output_pdf, bbox_inches='tight', dpi=150)
    else:
        plt.show()
