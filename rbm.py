import os.path
from tempfile import mkdtemp
from typing import *
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.distributions import Bernoulli
import torch.utils.tensorboard as tb

from einops import rearrange

logger = logging.getLogger()

IMG_SIDE_LEN = 28
NUM_LABELS = 10

dataset_root = os.path.expanduser("~/archive/datasets/mnist/pytorch_root/")

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

    def __len__() -> int:
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


train_dataset = HomogenousBinaryMNIST(dataset_root, train=True)


class HomogenousBinaryRBM(nn.Module):
    def __init__(self, visible_num_vars: int, hidden_num_vars: int):
        super().__init__()
        self.visible_num_vars = visible_num_vars
        self.hidden_num_vars = hidden_num_vars
        self.W = nn.Parameter(torch.randn(visible_num_vars, hidden_num_vars) * 0.33)

    def energy(self, visible: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return -torch.einsum("nm,bn,bm->b", self.W, visible, hidden)

    def unnormalized_likelihood(
        self, visible: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp(-self.energy(visible, hidden))

    def gibbs_sample(
        self,
        batch_size: int,
        num_steps: int,
        tb_writer: Optional[tb.SummaryWriter] = None,
        tb_tag: Optional[str] = None,
    ) -> torch.Tensor:
        assert batch_size > 0
        assert num_steps >= 2
        with torch.no_grad():
            hidden = Bernoulli(0.01).sample((batch_size, self.hidden_num_vars))
            hidden_history = []
            visible_history = []
            for step in range(num_steps):
                # calculate the conditional distribution p(visible|hidden)
                # it'll be a multivariate bernoulli distribution
                p_vis_given_hid = Bernoulli(
                    torch.sigmoid(torch.einsum("nm,bm->bn", self.W, hidden))
                )
                visible = p_vis_given_hid.sample()
                if tb_writer is not None:
                    tb_writer.add_histogram(
                        f"{tb_tag}_expectation_of_visible", p_vis_given_hid.mean, step
                    )
                    tb_writer.add_scalar(
                        f"{tb_tag}_mean_of_expectation_of_visible",
                        p_vis_given_hid.mean.mean(),
                        step,
                    )
                    tb_writer.add_image(
                        f"{tb_tag}_expectation_of_visible_of_image_part",
                        train_dataset.extract_images(p_vis_given_hid.mean[0]).squeeze(),
                        step,
                        dataformats="HW",
                    )
                    if step > 0:
                        tb_writer.add_histogram(
                            f"{tb_tag}_expectation_of_hidden",
                            p_hid_given_vis.mean,
                            step,
                        )
                        tb_writer.add_scalar(
                            f"{tb_tag}_mean of expectation of hidden",
                            p_hid_given_vis.mean.mean(),
                            step,
                        )
                    tb_writer.add_scalar(f"{tb_tag}_hidden at 0,0", hidden[0, 0], step)
                hidden_history.append(hidden)
                visible_history.append(visible)
                p_hid_given_vis = Bernoulli(
                    torch.sigmoid(torch.einsum("nm,bn->bm", self.W, visible))
                )
                hidden = p_hid_given_vis.sample()
            # do Rubin-Gelman convergence diagnostic
            # https://stats.stackexchange.com/questions/99375/gelman-and-rubin-convergence-diagnostic-how-to-generalise-to-work-with-vectors
            # and BMoML Sk lecture 12 slides
            last_half = torch.cat(
                (torch.stack(hidden_history), torch.stack(visible_history)), dim=2
            )[round(len(hidden_history) / 2) :]
            # shape: chain length × batch size × (visible+hidden) size

            half_chain_len = last_half.shape[0]
            variables_size = last_half.shape[-1]
            within_chain_var = torch.mean(torch.var(last_half, dim=0), dim=0)
            assert within_chain_var.shape == (variables_size,)
            between_chain_var = (
                torch.var(torch.mean(last_half, dim=0), dim=0) * half_chain_len
            )
            assert between_chain_var.shape == within_chain_var.shape
            weighted_sum_of_vars = (
                within_chain_var * (half_chain_len - 1) / half_chain_len
                + between_chain_var / half_chain_len
            )
            gelman_rubin_statistic = torch.sqrt(weighted_sum_of_vars / within_chain_var)
            threshold = 1.2
            num_unconverged_components = torch.sum(gelman_rubin_statistic > 1.2)
            log_str = f"num_unconverged_components / variables_size = {num_unconverged_components} / {variables_size}"
            if tb_writer is not None:
                tb_writer.add_histogram(
                    f"{tb_tag}_gelman_rubin_statistic", gelman_rubin_statistic
                )
                tb_writer.add_histogram(f"{tb_tag}_within_chain_var", within_chain_var)
                tb_writer.add_histogram(
                    f"{tb_tag}_between_chain_var", between_chain_var
                )
                tb_writer.add_histogram(
                    f"{tb_tag}_weighted_sum_of_vars", weighted_sum_of_vars
                )
                tb_writer.add_text(f"{tb_tag}", log_str)
            if num_unconverged_components > 0:
                logger.warn(log_str)
            return visible, hidden


hidden_num_vars = 2

model = HomogenousBinaryRBM(train_dataset.data.shape[1], hidden_num_vars)
tb_dir = mkdtemp()
print(f"tb_dir = {tb_dir}")
tb_writer = tb.SummaryWriter(tb_dir)
visible, hidden = model.gibbs_sample(
    batch_size=15, num_steps=30, tb_writer=tb_writer, tb_tag="babbys_first_gibbs"
)
