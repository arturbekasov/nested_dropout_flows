from collections import namedtuple

import torch
from torch.utils.data import Subset
from torchvision import transforms as tvt, datasets

from utils import Preprocess

ImageDataset = namedtuple('ImageDataset', ['dataset',
                                           'img_shape',
                                           'preprocess_fn',
                                           'valid_indices'])


def create_dataset(root,
                   dataset,
                   num_bits,
                   pad,
                   valid_size,
                   valid_indices=None,
                   split='train'):
    assert split in ['train', 'valid', 'test']

    preprocess = Preprocess(num_bits)
    c, h, w = (1, 28 + 2 * pad, 28 + 2 * pad)

    if dataset == 'fashion-mnist':
        transforms = []

        if split == 'train':
            transforms += [tvt.RandomHorizontalFlip()]

        transforms += [
            tvt.Pad((pad, pad)),
            tvt.ToTensor(),
            preprocess
        ]

        dataset = datasets.FashionMNIST(
            root=root,
            train=(split in ['train', 'valid']),
            transform=tvt.Compose(transforms),
            download=True
        )
    else:
        raise RuntimeError('Unknown dataset')

    if split == 'train':
        num_train = len(dataset)
        indices = torch.randperm(num_train).tolist()
        # valid_size = int(np.floor(valid_frac * num_train))
        train_indices, valid_indices = indices[valid_size:], indices[:valid_size]
        dataset = Subset(dataset, train_indices)
    elif split == 'valid':
        dataset = Subset(dataset, valid_indices)

    print(f'Using {split} data split of size {len(dataset)}')

    return ImageDataset(dataset=dataset,
                        img_shape=(c, h, w),
                        preprocess_fn=preprocess,
                        valid_indices=valid_indices)