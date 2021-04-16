import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nflows import distributions, flows
from torchvision.utils import save_image

from data import create_dataset
from transform import create_transform


def sample(flow, eval_dataset, output_dir, use_gpu, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    c, h, w = eval_dataset.img_shape
    eval_image, _ = eval_dataset.dataset[2]
    eval_image = eval_image[None, ...]

    flow = flow.to(device)
    flow.eval()

    # Disable gradients for the rest of the run
    torch.set_grad_enabled(False)

    # Samples
    images = flow.sample(18)
    images = eval_dataset.preprocess_fn.inverse(images)
    save_image(images, output_dir / 'samples.pdf', nrow=6)

    # Reconstructions
    mask_indices = torch.linspace(1, 510, 17).long().flip(0)[:, None]
    mask = (reversed(torch.arange(c * h * w)).expand(mask_indices.shape[0], -1)
            < mask_indices).float()
    mask = mask.to(device)

    eval_noise, _ = flow._transform(eval_image.to(device))
    reconstr_noise = eval_noise * mask
    images = flow._transform.inverse(reconstr_noise)[0]
    images = torch.cat([eval_image, images.cpu()])
    images = eval_dataset.preprocess_fn.inverse(images)
    save_image(images, output_dir / 'reconstructions.pdf', nrow=6)

    # Samples from the manifold
    noise = flow._distribution.sample(12)[:, None, :]
    mask_indices = torch.tensor([2, 4, 8, 16, 32, 64, 128, 256, 512])[:, None]
    mask = (reversed(torch.arange(c * h * w)).expand(mask_indices.shape[0], -1)
            < mask_indices).float()
    mask = mask.to(device)
    noise = (noise * mask[None, ...]).permute([1, 0, 2]).reshape(-1, 1024)
    images = flow._transform.inverse(noise)[0]
    images = eval_dataset.preprocess_fn.inverse(images)
    save_image(images, output_dir / 'manifold_samples.pdf', nrow=12)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=314833845)

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

    # Load validation data
    valid_indices = torch.load(run_dir / 'valid_indices.pt')
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

    print('Sampling')
    sample(flow=flow,
           eval_dataset=eval_dataset,
           output_dir=output_dir,
           use_gpu=args.use_gpu,
           seed=args.seed)

if __name__ == '__main__':
    main()