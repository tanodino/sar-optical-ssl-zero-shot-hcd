"""
Paired SAR / optical pre-training data.

Data format (produced by preprocess_corpus.py): two numpy arrays of shape (N, C, H, W),
one per modality, with the same N and row i of both arrays = the same co-registered location.

Notes:
* Arrays are opened with `mmap_mode="r"`: nothing is copied into RAM up-front, the OS page cache does the work.
* Training is defined in **steps**: `infinite_batches` reshuffles at every pass over the data with a
  seeded generator, so a given seed gives the same batch order; the last incomplete batch is dropped
  (constant batch size = constant number of in-batch negatives).
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class PairedArrays(Dataset):
    """Dataset over two memory-mapped arrays. __getitem__(i) -> (sar_i, opt_i) as float32 tensors (C, H, W)."""

    def __init__(self, sar_path: str, opt_path: str, max_samples: Optional[int] = None):
        self.sar = np.load(sar_path, mmap_mode="r")
        self.opt = np.load(opt_path, mmap_mode="r")
        if self.sar.shape[0] != self.opt.shape[0]:
            raise ValueError(f"different number of samples: SAR {self.sar.shape}, optical {self.opt.shape}")
        if self.sar.shape[2:] != self.opt.shape[2:]:
            raise ValueError(f"different spatial size: SAR {self.sar.shape}, optical {self.opt.shape}")
        self.n = self.sar.shape[0] if max_samples is None else min(max_samples, self.sar.shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(np.array(self.sar[i], dtype=np.float32)),
                torch.from_numpy(np.array(self.opt[i], dtype=np.float32)))

    @property
    def sar_channels(self) -> int:
        return self.sar.shape[1]

    @property
    def opt_channels(self) -> int:
        return self.opt.shape[1]

    @property
    def img_size(self) -> int:
        assert self.sar.shape[2] == self.sar.shape[3], "square tiles expected"
        return self.sar.shape[2]


class EpochShuffleBatchSampler(Sampler):
    """Yields full batches of indices forever; a new permutation (from a seeded generator) at every pass.

    `start_batch` lets a resumed run skip the batches already consumed, so that resuming gives the same
    batch sequence as an uninterrupted run with the same seed.
    """

    def __init__(self, n: int, batch_size: int, seed: int, start_batch: int = 0):
        if n < batch_size:
            raise ValueError(f"dataset ({n}) smaller than one batch ({batch_size})")
        self.n, self.batch_size, self.seed, self.start_batch = n, batch_size, seed, start_batch
        self.batches_per_pass = n // batch_size      # drop_last

    def __iter__(self) -> Iterator[list]:
        b = self.start_batch
        while True:
            epoch, k = divmod(b, self.batches_per_pass)
            g = torch.Generator().manual_seed(self.seed * 100_003 + epoch)   # deterministic per pass
            perm = torch.randperm(self.n, generator=g)
            for j in range(k, self.batches_per_pass):
                # Sorted indices make the memory-mapped reads more sequential; order inside a batch is irrelevant.
                yield sorted(perm[j * self.batch_size:(j + 1) * self.batch_size].tolist())
            b = (epoch + 1) * self.batches_per_pass

    def __len__(self) -> int:  # infinite sampler; length of one pass, for information only
        return self.batches_per_pass


def infinite_batches(dataset: PairedArrays, batch_size: int, seed: int, start_batch: int = 0,
                     num_workers: int = 6) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Endless iterator of (sar (B, C1, H, W), opt (B, C2, H, W)) CPU batches (pinned memory)."""
    sampler = EpochShuffleBatchSampler(len(dataset), batch_size, seed, start_batch)
    # A dedicated generator: creating the loader iterator draws a base seed, and drawing it from the GLOBAL
    # torch RNG would shift the random stream (masks, augmentations) after a resume (found by the unit test).
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=4 if num_workers > 0 else None,
                        generator=torch.Generator().manual_seed(seed + start_batch))
    return iter(loader)
