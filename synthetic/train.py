import argparse
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.distributions import Geometric
from torch.utils.data import TensorDataset, DataLoader

from synthetic.data import generate_data, TRAIN_SEED
from synthetic.flow import create_flow
from utils import load_num_batches


def train(flow, *, mse_coef, train_data, num_iters, geometric_prob):
    flow.train()

    train_dataset = TensorDataset(torch.FloatTensor(train_data))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=500)
    optimizer = optim.Adam(flow.parameters())

    idx_distr = Geometric(geometric_prob)

    for it, (batch,) in enumerate(load_num_batches(train_loader, num_batches=num_iters)):
        optimizer.zero_grad()

        noise, logabsdet = flow._transform.forward(batch)

        # Compute likelihood.
        log_prob = flow._distribution.log_prob(noise) + logabsdet
        log_prob_loss = -torch.mean(log_prob)

        # Compute reconstruction MSE.
        batch_size = batch.shape[0]
        mask_indices = idx_distr.sample([batch_size])[:, None] + 1
        mask = (torch.arange(3).expand(batch_size, -1) < mask_indices).float()
        reconstr_batch, _ = flow._transform.inverse(noise * mask)
        mse = torch.mean((batch - reconstr_batch).pow(2))

        loss = log_prob_loss + mse_coef * mse
        loss.backward()

        optimizer.step()

        if it % (num_iters // 10) == 0:
            print(f'it {it} loss {loss.item():.3f} '
                  f'(log_prob {log_prob_loss.item():.3f} '
                  f'mse {mse.item():.3f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('--num_iters', type=int, default=30000)
    parser.add_argument('--geometric_prob', type=float, default=0.33)
    parser.add_argument('--mse_coef', type=float, default=0.)
    parser.add_argument('--flow_type', type=str, choices=['qr_flow', 'lu_flow'], default='qr_flow')
    parser.add_argument('--seed', type=int, default=14042021)
    parser.add_argument('--num_train_points', type=int, default=10000)
    parser.add_argument('--num_repeats', type=int, default=1)

    args = parser.parse_args()

    train_data = generate_data(num_points=args.num_train_points,
                               seed=TRAIN_SEED)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for repeat in range(1, args.num_repeats+1):
        flow = create_flow(args.flow_type)

        train(flow,
              mse_coef=args.mse_coef,
              train_data=train_data,
              num_iters=args.num_iters,
              geometric_prob=args.geometric_prob)

        if args.num_repeats > 1:
            torch.save(flow.state_dict(), output_dir/f'ckpt_{repeat}.pt')
        else:
            torch.save(flow.state_dict(), output_dir/'ckpt.pt')

