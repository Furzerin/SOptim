"""CIFAR-10 loaders: a standard torchvision pipeline and an in-memory batched one."""
import torch


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def torchvision_loader(root, train, batch_size, num_workers=2, drop_last=None):
    """Per-sample PIL transforms in worker processes (the usual baseline)."""
    import torchvision
    import torchvision.transforms as T

    tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    tf = T.Compose(tf + [T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                                       persistent_workers=num_workers > 0,
                                       drop_last=train if drop_last is None else drop_last)


class FastCIFAR10Loader:
    """Whole dataset as one uint8 tensor; crop/flip/normalize applied per batch with tensor ops.

    Same augmentation distribution as `torchvision_loader` (zero-pad 4 + random crop + hflip).
    """

    def __init__(self, root, train, batch_size, seed=0, drop_last=None):
        import torchvision

        ds = torchvision.datasets.CIFAR10(root, train=train, download=True)
        self.images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # uint8 [N,3,32,32]
        self.labels = torch.tensor(ds.targets)
        self.train, self.batch_size = train, batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.gen = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(MEAN).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(STD).view(1, 3, 1, 1) * 255

    def __len__(self):
        n = len(self.labels)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _augment(self, x):
        B = x.shape[0]
        x = torch.nn.functional.pad(x, (4, 4, 4, 4))
        dy, dx = torch.randint(0, 9, (2, B, 1), generator=self.gen)
        ar = torch.arange(32)
        rows, cols = (dy + ar)[:, :, None], (dx + ar)[:, None, :]
        x = x[torch.arange(B)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(B, generator=self.gen) < 0.5
        return torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)

    def __iter__(self):
        n = len(self.labels)
        idx = torch.randperm(n, generator=self.gen) if self.train else torch.arange(n)
        for i in range(len(self)):
            b = idx[i * self.batch_size:(i + 1) * self.batch_size]
            x = self.images[b]
            if self.train:
                x = self._augment(x)
            yield (x.float() - self.mean) / self.std, self.labels[b]
