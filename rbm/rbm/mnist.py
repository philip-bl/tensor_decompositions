import logging

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

IMG_SIDE_LEN = 28
NUM_LABELS = 10

transform = transforms.Compose(
    (
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor: (tensor > 0).float().flatten()),
    )
)


class HomogenousBinaryMNIST(Dataset):
    def __init__(self, dataset_root: str, train: bool):
        torchvision_dataset = MNIST(dataset_root, train, transforms.ToTensor())
        self.classes = torchvision_dataset.classes
        self.class_to_idx = torchvision_dataset.class_to_idx
        self.data = torch.cat(
            (
                rearrange((torchvision_dataset.data > 0).float(), "b h w -> b (h w)"),
                F.one_hot(torchvision_dataset.targets, len(self.classes)).float(),
            ),
            dim=1,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]

    @staticmethod
    def extract_images(tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.unsqueeze(0) if tensor.ndimension() == 1 else tensor
        assert batch.ndimension() == 2
        return rearrange(
            batch[:, : IMG_SIDE_LEN ** 2], "b (h w) -> b h w", h=IMG_SIDE_LEN
        )
