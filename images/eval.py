import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nflows import distributions, flows
from torch.utils.data import DataLoader

from data import create_dataset
from transform import create_transform
from utils import nats_to_bits_per_dim


def eval(output_dir, eval_dataset, flow, dim_order, use_gpu, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    c, h, w = eval_dataset.img_shape

    flow = flow.to(device)
    flow.eval()

    # Disable gradients for the rest of the run
    torch.set_grad_enabled(False)

    eval_loader = DataLoader(eval_dataset.dataset,
                             batch_size=1000,
                             shuffle=False)

    print(f'Evaluating likelihood')

    # Evaluate test likelihood
    bpd = []
    for batch, _ in eval_loader:
        batch = batch.to(device)
        log_prob = flow.log_prob(batch)
        bpd.append(nats_to_bits_per_dim(nats=-log_prob, c=c, h=h, w=w))
    bpd = torch.cat(bpd)
    bpd_mean = torch.mean(bpd).item()
    bpd_std = torch.std(bpd).item()
    print(f"BPD: {bpd_mean:.3f} +/- {bpd_std:.3f}")
    np.save(output_dir / 'bpd.npy', (bpd_mean, bpd_std))

    print(f'Evaluating reconstruction error')

    # Evaluate reconstruction error
    if dim_order == 'original':
        indices = torch.arange(c * h * w)
    elif dim_order == 'reversed':
        indices = reversed(torch.arange(c * h * w))
    elif dim_order == 'random':
        indices = torch.arange(c * h * w)[torch.randperm(c * h * w)]
    else:
        raise RuntimeError('Unkown order')

    values = []
    for keep in range(1, 1024, 10):
        mask = (indices < keep).float().to(device)
        mse = []
        for batch, _ in eval_loader:
            batch = batch.to(device)
            noise, _ = flow._transform(batch)
            reconstr_noise = noise * mask[None, :]
            batch_reconstr = flow._transform.inverse(reconstr_noise)[0]
            mse.append(torch.mean((batch - batch_reconstr).pow(2).view(batch.shape[0], -1), dim=1))
        mse = torch.mean(torch.cat(mse, dim=0))
        print(f"MSE({keep}): {mse.item():.3f}")
        values.append((keep, mse.item()))
    np.save(output_dir / 'mse.npy', np.array(values))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=314833845)
    parser.add_argument('--valid_proportion', type=float, default=1.)
    parser.add_argument('--dim_order', type=str, choices=['original', 'reversed', 'random'],
                        default='reversed')
    parser.add_argument('--use_test_data', type=bool, default=False)

    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--data_config', type=str,
                        default=script_dir / 'config' / 'data_config.json')
    parser.add_argument('--flow_config', type=str,
                        default=script_dir / 'config' / 'flow_config.json')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    print('Loading data')

    with open(args.data_config) as fp:
        data_config = json.load(fp)

    if args.use_test_data:
        eval_dataset = create_dataset(root=args.data_dir,
                                      split='test',
                                      **data_config)
    else:
        # Use validation data
        valid_indices = torch.load(run_dir / 'valid_indices.pt')
        valid_indices = valid_indices[:int(len(valid_indices) * args.valid_proportion)]
        eval_dataset = create_dataset(root=args.data_dir,
                                      split='valid',
                                      valid_indices=valid_indices,
                                      **data_config)

    print('Creating a flow')

    with open(args.flow_config) as fp:
        flow_config = json.load(fp)

    c, h, w = eval_dataset.img_shape
    distribution = distributions.StandardNormal((c * h * w,))
    transform = create_transform(c, h, w,
                                 num_bits=data_config['num_bits'],
                                 **flow_config)
    flow = flows.Flow(transform, distribution)

    # Load checkpoint
    flow_ckpt = run_dir / 'latest_flow.pt'
    flow.load_state_dict(torch.load(flow_ckpt))
    print(f'Flow checkpoint loaded: {flow_ckpt}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    eval(
        output_dir=output_dir,
        eval_dataset=eval_dataset,
        flow=flow,
        dim_order=args.dim_order,
        use_gpu=args.use_gpu,
        seed=args.seed
    )

if __name__ == '__main__':
    main()