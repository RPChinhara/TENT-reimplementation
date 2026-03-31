from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def build_cifar10_transforms(train):
    steps = []
    if train:
        steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(steps)


def build_cifar10_dataloaders(
    data_dir,
    batch_size,
    eval_batch_size,
    num_workers=2,
    download=False,
):
    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=build_cifar10_transforms(train=True),
        download=download,
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=build_cifar10_transforms(train=False),
        download=download,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


class CIFAR10CDataset(Dataset):
    def __init__(self, corruption, severity, data_dir, num_examples=10000):
        root = Path(data_dir) / "CIFAR-10-C"
        corruption_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"

        if not corruption_path.exists():
            raise FileNotFoundError(f"Missing corruption file: {corruption_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Severity must be in [1, 5], got {severity}")

        self.images = np.load(corruption_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        start = (severity - 1) * 10000
        count = min(num_examples, 10000)
        self.indices = range(start, start + count)
        self.transform = build_cifar10_transforms(train=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_index = self.indices[index]
        image = np.array(self.images[sample_index], copy=True)
        target = int(self.labels[sample_index])
        image = self.transform(image)
        return image, target


def build_cifar10c_loader(
    corruption,
    severity,
    data_dir,
    batch_size,
    num_examples=10000,
    num_workers=2,
):
    dataset = CIFAR10CDataset(
        corruption=corruption,
        severity=severity,
        data_dir=data_dir,
        num_examples=num_examples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
