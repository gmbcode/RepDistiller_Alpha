"""
dataset_patch.py
----------------
Drop-in patch for RepDistiller's dataset/cifar100.py to add --subset_frac support.

HOW TO USE:
    Copy this file into the RepDistiller root, then at the top of train_student.py
    (right after the existing dataset import), add:

        from dataset_patch import patch_get_cifar100_dataloaders_sample
        patch_get_cifar100_dataloaders_sample()

OR just manually apply the diff below to dataset/cifar100.py.

This adds a `subset_frac` kwarg to get_cifar100_dataloaders_sample().
"""

import numpy as np


def get_cifar100_dataloaders_sample_patched(
    batch_size=128,
    num_workers=8,
    k=4096,
    mode="exact",
    is_sample=True,
    percent=1.0,
    subset_frac=1.0,     # <-- NEW: fraction of training data to use
):
    """
    Patched version of get_cifar100_dataloaders_sample that supports
    subset_frac for fast hyperparameter sweeps.

    subset_frac=1.0  -> use all 50,000 training samples  (default, paper setting)
    subset_frac=0.2  -> use 10,000 training samples
    subset_frac=0.1  -> use  5,000 training samples  (very fast, signal may degrade)
    """
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import torch
    from torch.utils.data import DataLoader, Subset
    from dataset.cifar100 import CIFAR100InstanceSample   # original class

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        ),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        ),
    ])

    train_set = CIFAR100InstanceSample(
        root="./data",
        download=True,
        train=True,
        transform=train_transform,
        k=k,
        mode=mode,
        is_sample=is_sample,
        percent=percent,
    )

    # ── Stratified subsetting ───────────────────────────────────────────
    if subset_frac < 1.0:
        targets = np.array(train_set.targets)
        classes = np.unique(targets)
        keep_indices = []
        rng = np.random.default_rng(seed=42)   # reproducible

        for cls in classes:
            cls_indices = np.where(targets == cls)[0]
            n_keep = max(1, int(len(cls_indices) * subset_frac))
            chosen = rng.choice(cls_indices, size=n_keep, replace=False)
            keep_indices.extend(chosen.tolist())

        rng.shuffle(keep_indices)
        keep_indices = np.asarray(keep_indices)

        # Slice data/targets in-place and rebuild class index tables so
        # returned sample indices remain contiguous in [0, n_data - 1]
        # for ContrastMemory indexing.
        train_set.data = train_set.data[keep_indices]
        train_set.targets = [int(targets[i]) for i in keep_indices]
        n_data = len(train_set.targets)

        num_classes = 100
        cls_positive = [[] for _ in range(num_classes)]
        for i, lbl in enumerate(train_set.targets):
            cls_positive[lbl].append(i)

        cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j != i:
                    cls_negative[i].extend(cls_positive[j])

        cls_positive = [np.asarray(cls_positive[i]) for i in range(num_classes)]
        cls_negative = [np.asarray(cls_negative[i]) for i in range(num_classes)]
        if 0 < percent < 1:
            n = int(len(cls_negative[0]) * percent)
            cls_negative = [np.random.permutation(cls_negative[i])[0:n]
                            for i in range(num_classes)]

        train_set.cls_positive = np.asarray(cls_positive)
        train_set.cls_negative = np.asarray(cls_negative)

        print(f"[subset] Using {n_data} / {len(targets)} training samples "
              f"({subset_frac*100:.0f}%, stratified)")
    else:
        n_data = len(train_set)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_set = datasets.CIFAR100(
        root="./data", download=True, train=False, transform=test_transform
    )
    val_loader = DataLoader(
        test_set,
        batch_size=int(batch_size / 2),
        shuffle=False,
        num_workers=int(num_workers / 2),
        pin_memory=True,
    )

    return train_loader, val_loader, n_data


def patch_get_cifar100_dataloaders_sample():
    """Monkey-patch the dataset module at runtime."""
    import dataset.cifar100 as _mod
    _mod.get_cifar100_dataloaders_sample = get_cifar100_dataloaders_sample_patched
    print("[dataset_patch] Patched get_cifar100_dataloaders_sample with subset_frac support")
