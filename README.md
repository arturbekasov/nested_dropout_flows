# Nested Dropout Flows

Code and experiments for the paper:
> Artur Bekasov, Iain Murray, _Ordering Dimensions with Nested Dropout Normalizing Flows_.
> [[arXiv]](https://arxiv.org/abs/2006.08777)

Presented at the Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models [[INNF+]](https://invertibleworkshop.github.io), ICML 2020.

## Dependencies 

[`nflows`](https://github.com/bayesiains/nflows) package provides flow implementations.

See `requirements.txt` for other dependencies. To install all at once:
```bash
pip install -r requirements.txt
```

## Usage

### Synthetic experiments

`synthetic` directory contains code for experiments with the synthetic 3D dataset. 

To train a model:
```bash
python synthetic/train.py -o run_dir
```

To evaluate a trained model:
```bash
python synthetic/eval.py -i run_dir
```

### Image experiments

`images` directory contains code for experiments with [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) images. 

To train a model:
```bash
python images/train.py\
  --data_dir ...\     # Where to store downloaded data
  --run_dir run_dir\  # Where to store checkpoints
  --reconstr_coef ... # Hyperparameters
```

To evaluate a trained model:
```bash
python images/eval.py\
  --data_dir ...\     # Where to store downloaded data
  --run_dir run_dir\  # run_dir used for train.py 
  --output_dir ...    # Where to store the artifacts 
```

`images/eval.py` outputs:
- `bpd.npy`: negative log likelihood in bits-per-dimension.
- `mse.npy`: reconstruction MSE, varying the number of dimensions dropped. 

To sample from a trained model:
```bash
python images/sample.py\
  --data_dir ...\     # Where to store downloaded data
  --run_dir run_dir\  # run_dir used for train.py 
  --output_dir ...    # Where to store the artifacts 
```
