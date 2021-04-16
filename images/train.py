import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.distributions import Geometric
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from nflows import flows, distributions

from transform import create_transform
from data import ImageDataset, create_dataset
from utils import load_num_batches, nats_to_bits_per_dim

# Silence TF re. some package not being installed.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def train(flow,
          train_dataset: ImageDataset,
          run_dir,
          *,
          num_iters,
          batch_size,
          reconstr_coef,
          geometric_prob,
          use_gpu,
          num_evals,
          lr,
          flow_ckpt,
          optimizer_ckpt,
          first_iter):
    c, h, w = train_dataset.img_shape

    run_dir = Path(run_dir)
    run_dir.mkdir(exist_ok=True)

    torch.save(train_dataset.valid_indices, run_dir / 'valid_indices.pt')  # Save for evaluation

    train_loader = DataLoader(dataset=train_dataset.dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    eval_batch, _ = next(iter(DataLoader(dataset=train_dataset.dataset,
                                         batch_size=10,
                                         shuffle=True)))

    if flow_ckpt is not None:
        flow.load_state_dict(torch.load(flow_ckpt))
        print(f'Flow checkpoint loaded: {flow_ckpt}')

    flow = flow.to(device)

    optimizer = optim.Adam(flow.parameters(), lr=lr)
    if optimizer_ckpt is not None:
        optimizer.load_state_dict(torch.load(optimizer_ckpt))
        print(f'Optimizer state loaded: {optimizer_ckpt}')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        last_epoch=first_iter - 1,
        T_max=num_iters
    )

    idx_distr = Geometric(geometric_prob)

    summary_writer = SummaryWriter(run_dir)

    for it, (batch, _) in enumerate(load_num_batches(loader=train_loader,
                                                     num_batches=num_iters - first_iter),
                                    start=first_iter):
        flow.train()

        batch = batch.to(device)

        optimizer.zero_grad()

        batch_noise, logabsdet = flow._transform(batch)
        log_prob = flow._distribution.log_prob(batch_noise)
        log_prob += logabsdet
        log_prob_loss = -nats_to_bits_per_dim(nats=torch.mean(log_prob), c=c, h=h, w=w)

        summary_writer.add_scalar('log_prob_loss', log_prob_loss.item(), it)

        if reconstr_coef == 0:
            loss = log_prob_loss
        else:
            batch_size = batch.shape[0]
            mask_indices = idx_distr.sample([batch_size])[:, None] + 1
            mask = (reversed(torch.arange(c * h * w)).expand(batch_size, -1) < mask_indices).float()
            mask = mask.to(device)

            reconstr_noise = batch_noise * mask
            batch_reconstr, _ = flow._transform.inverse(reconstr_noise)
            reconstr_loss = torch.mean((batch - batch_reconstr).pow(2))

            summary_writer.add_scalar('reconstr_loss', reconstr_loss.item(), it)

            loss = log_prob_loss + reconstr_coef * reconstr_loss

        summary_writer.add_scalar('loss', loss.item(), it)
        summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], it)

        loss.backward()

        optimizer.step()
        scheduler.step()

        if it % (num_iters // num_evals) == 0:
            print(f'Training: it {it}/{num_iters} loss {loss.item():.4f}')
            flow.eval()

            # Checkpoint flow and optimizer
            torch.save(flow.state_dict(), run_dir / f'latest_flow.pt')
            torch.save(optimizer.state_dict(), run_dir / f'latest_optim.pt')

            with torch.no_grad():
                # Sample
                images = flow.sample(64)
                images = train_dataset.preprocess_fn.inverse(images)
                images_grid = make_grid(images, nrow=8)
                summary_writer.add_image('samples', images_grid, global_step=it)

                # Reconstructions
                mask_indices = torch.tensor([1, 2, 3, 4, 5, 10, 50, 100,
                                             250, 500, 750, 1000, 1025])[:, None]
                nrow = mask_indices.shape[0]
                mask = (reversed(torch.arange(c * h * w)).expand(mask_indices.shape[0], -1)
                        < mask_indices).float()
                mask = mask.to(device)

                eval_noise, _ = flow._transform(eval_batch.to(device))
                reconstr_noise = eval_noise[:, None, :] * mask[None, ...]

                reconstr_noise = reconstr_noise.view(-1, reconstr_noise.shape[-1])
                images = flow._transform.inverse(reconstr_noise)[0]
                images = train_dataset.preprocess_fn.inverse(images)
                images_grid = make_grid(images, nrow=nrow)
                summary_writer.add_image('reconstructions', images_grid, global_step=it)

    # Last checkpoint
    torch.save(flow.state_dict(), run_dir / f'latest_flow.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--geometric_prob', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_evals', type=int, default=200)
    parser.add_argument('--num_iters', type=int, default=100000)
    parser.add_argument('--reconstr_coef', type=float, default=0.0)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=314833845)

    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--data_config', type=str,
                        default=script_dir/'config'/'data_config.json')
    parser.add_argument('--flow_config', type=str,
                        default=script_dir/'config'/'flow_config.json')

    # For restarting runs.
    parser.add_argument('--flow_ckpt', type=str)
    parser.add_argument('--optimizer_ckpt', type=str)
    parser.add_argument('--first_iter', type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Loading data')
    with open(args.data_config) as fp:
        data_config = json.load(fp)
    train_dataset = create_dataset(root=args.data_dir,
                                   split='train',
                                   **data_config)
    c, h, w = train_dataset.img_shape

    print('Creating a flow')
    with open(args.flow_config) as fp:
        flow_config = json.load(fp)
    distribution = distributions.StandardNormal((c * h * w,))
    transform = create_transform(c, h, w, num_bits=data_config['num_bits'], **flow_config)
    flow = flows.Flow(transform, distribution)

    print('Training')
    train(flow=flow,
          train_dataset=train_dataset,
          run_dir=args.run_dir,
          num_iters=args.num_iters,
          batch_size=args.batch_size,
          reconstr_coef=args.reconstr_coef,
          geometric_prob=args.geometric_prob,
          use_gpu=args.use_gpu,
          num_evals=args.num_evals,
          lr=args.learning_rate,
          flow_ckpt=args.flow_ckpt,
          optimizer_ckpt=args.optimizer_ckpt,
          first_iter=args.first_iter)

if __name__ == '__main__':
    main()