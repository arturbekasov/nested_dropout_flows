import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from synthetic.data import generate_data, TEST_SEED, VALIDATION_SEED
from synthetic.flow import create_flow


def evaluate(flow, eval_data):
    flow.eval()

    eval_data = TensorDataset(torch.FloatTensor(eval_data))
    eval_data = DataLoader(dataset=eval_data,
                            batch_size=500,
                            shuffle=False)

    with torch.no_grad():
        # Evaluate test likelihood
        total_log_prob = 0
        for (batch,) in eval_data:
            total_log_prob += torch.mean(flow.log_prob(batch))
        total_log_prob /= len(eval_data)
        total_log_prob = total_log_prob.item()

        # Evaluate reconstruction error
        mse_results = {}
        for keep in [3, 2, 1]:
            mse = []
            for (batch,) in eval_data:
                noise, _ = flow._transform.forward(batch)
                mask = (torch.arange(3).expand(500, -1) < keep).float()
                reconstr_batch, _ = flow._transform.inverse(noise * mask)
                mse.append(torch.mean((batch - reconstr_batch).pow(2), dim=1))
            mse = torch.mean(torch.cat(mse, dim=0))
            mse = mse.item()
            if keep == 3:
                assert mse < 1e-6
            else:
                mse_results[keep] = mse

    return total_log_prob, mse_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('--flow_type', type=str, choices=['qr_flow', 'lu_flow'], default='qr_flow')
    parser.add_argument('--seed', type=int, default=14042021)
    parser.add_argument('--num_eval_points', type=int, default=10000)
    parser.add_argument('--test', type=bool, default=False)

    args = parser.parse_args()

    if args.test:
        print('Using test data')
        eval_data = generate_data(num_points=args.num_eval_points,
                                  seed=TEST_SEED)
    else:
        print('Using validation data')
        eval_data = generate_data(num_points=args.num_eval_points,
                                  seed=VALIDATION_SEED)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    ckpts = list(input_dir.glob('ckpt*.pt'))

    assert len(ckpts) > 0
    print(f'No. of checkpoints found: {len(ckpts)}')

    all_log_prob = []
    all_mse_2 = []
    all_mse_1 = []

    for ckpt in ckpts:
        flow = create_flow(args.flow_type)
        flow.load_state_dict(torch.load(ckpt))
        log_prob, mse_results = evaluate(flow, eval_data)

        all_log_prob.append(log_prob)
        all_mse_2.append(mse_results[2])
        all_mse_1.append(mse_results[1])

    log_prob_mean, log_prob_std = np.mean(all_log_prob), np.std(all_log_prob)
    mse_2_mean, mse_2_std = np.mean(all_mse_2), np.std(all_mse_2)
    mse_1_mean, mse_1_std = np.mean(all_mse_1), np.std(all_mse_1)

    print('Results:')
    print(f'log_prob: {log_prob_mean:.3f} +/- {2*log_prob_std:.3f}')
    print(f'mse(2): {mse_2_mean:.3f} +/- {2*mse_2_std:.3f}')
    print(f'mse(1): {mse_1_mean:.3f} +/- {2*mse_1_std:.3f}')