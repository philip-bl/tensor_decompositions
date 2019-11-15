import os.path
from tempfile import mkdtemp
from typing import *
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.distributions import Bernoulli
import torch.utils.tensorboard as tb

# import tensorboardX as tb

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


train_dataset = HomogenousBinaryMNIST(dataset_root, train=True)


def gen_all_binary_vectors(length: int) -> torch.Tensor:
    return (
        (torch.arange(2 ** length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1
    ).float()


class HomogenousBinaryRBM(nn.Module):
    def __init__(
        self,
        visible_num_vars: int,
        hidden_num_vars: int,
        visible_bias_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.visible_num_vars = visible_num_vars
        self.hidden_num_vars = hidden_num_vars
        self.W = nn.Parameter(torch.randn(visible_num_vars, hidden_num_vars) * 1e-2)
        self.visible_bias = nn.Parameter(torch.randn(self.visible_num_vars) * 1e-2)
        self.hidden_bias = nn.Parameter(torch.randn(self.hidden_num_vars) * 1e-2)
        if visible_bias_init is not None:
            self.visible_bias.data = visible_bias_init

    def unnormalized_log_likelihood(
        self, visible: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.einsum("nm,bn,bm->b", self.W, visible, hidden)
            + torch.einsum("n,bn->b", self.visible_bias, visible)
            + torch.einsum("m,bm->b", self.hidden_bias, hidden)
        )

    def unnormalized_likelihood(
        self, visible: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp(self.unnormalized_log_likelihood(visible, hidden))

    def log_unnormalized_marginal_likelihood_of_visible(
        self, visible: torch.Tensor
    ) -> torch.Tensor:
        foo = torch.einsum("bn,n->b", visible, self.visible_bias)
        bar = self.hidden_bias + torch.einsum("nm,bn->bm", self.W, visible)
        buzz = torch.sum(torch.log(1.0 + torch.exp(bar)), dim=1)
        assert buzz.shape == foo.shape
        result = foo + buzz
        assert torch.all(torch.isfinite(result))
        return result

    def log_unnormalized_marginal_likelihood_of_hidden(
        self, hidden: torch.Tensor
    ) -> torch.Tensor:
        """This method and log_unnormalized_marginal_likelihood_of_visible are
        mirrored copy pastes of each other."""
        foo = torch.einsum("bm,m->b", hidden, self.hidden_bias)
        bar = self.visible_bias + torch.einsum("nm,bm->bn", self.W, hidden)
        buzz = torch.sum(torch.log(1.0 + torch.exp(bar)), dim=1)
        assert buzz.shape == foo.shape
        result = foo + buzz
        assert torch.all(torch.isfinite(result))
        return result

    def log_normalization_constant(self) -> torch.Tensor:
        """returns ln(Z), such that unnormalized_likelihood / Z = likelihood."""
        assert self.hidden_num_vars <= 20
        hidden = gen_all_binary_vectors(self.hidden_num_vars)
        return torch.logsumexp(
            self.log_unnormalized_marginal_likelihood_of_hidden(hidden), dim=0
        )

    def log_marginal_likelihood_of_visible(
        self,
        visible: torch.Tensor,
        log_normalization_constant: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if log_normalization_constant is None:
            log_normalization_constant = self.log_normalization_constant()
        assert log_normalization_constant.shape == ()
        return (
            self.log_unnormalized_marginal_likelihood_of_visible(visible)
            - log_normalization_constant
        )

    def gibbs_sample(
        self,
        batch_size: int,
        num_steps: int,
        tb_writer: Optional[tb.SummaryWriter] = None,
        tb_tag_many_records: Optional[str] = None,
        tb_tag_one_record: Optional[str] = None,
    ) -> torch.Tensor:
        assert batch_size > 0
        assert num_steps >= 2
        with torch.no_grad():
            hidden = Bernoulli(0.15).sample((batch_size, self.hidden_num_vars))
            hidden_history = []
            visible_history = []
            for step in range(num_steps):
                # calculate the conditional distribution p(visible|hidden)
                # it'll be a multivariate bernoulli distribution
                p_vis_given_hid = Bernoulli(
                    torch.sigmoid(
                        self.visible_bias + torch.einsum("nm,bm->bn", self.W, hidden)
                    )
                )
                visible = p_vis_given_hid.sample()
                if tb_tag_many_records is not None:
                    tb_writer.add_histogram(
                        f"{tb_tag_many_records}_expectation_of_visible",
                        p_vis_given_hid.mean,
                        step,
                    )
                    tb_writer.add_scalar(
                        f"{tb_tag_many_records}_mean_of_expectation_of_visible",
                        p_vis_given_hid.mean.mean(),
                        step,
                    )
                    tb_writer.add_image(
                        f"{tb_tag_many_records}_expectation_of_visible_of_image_part",
                        train_dataset.extract_images(p_vis_given_hid.mean[0]).squeeze(),
                        step,
                        dataformats="HW",
                    )
                    if step > 0:
                        tb_writer.add_histogram(
                            f"{tb_tag_many_records}_expectation_of_hidden",
                            p_hid_given_vis.mean,
                            step,
                        )
                        tb_writer.add_scalar(
                            f"{tb_tag_many_records}_mean of expectation of hidden",
                            p_hid_given_vis.mean.mean(),
                            step,
                        )
                    tb_writer.add_scalar(
                        f"{tb_tag_many_records}_hidden at 0,0", hidden[0, 0], step
                    )
                hidden_history.append(hidden)
                visible_history.append(visible)
                p_hid_given_vis = Bernoulli(
                    torch.sigmoid(
                        self.hidden_bias + torch.einsum("nm,bn->bm", self.W, visible)
                    )
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
            within_chain_var = torch.mean(torch.var(last_half, dim=0), dim=0) + 1e-15
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
            assert torch.all(torch.isfinite(gelman_rubin_statistic))
            threshold = 1.2
            num_unconverged_components = torch.sum(gelman_rubin_statistic > 1.2)
            log_str = f"num_unconverged_components / variables_size = {num_unconverged_components} / {variables_size}"
            if tb_tag_one_record is not None:
                tb_writer.add_text(f"{tb_tag_one_record}", log_str)
                tb_writer.add_histogram(
                    f"{tb_tag_one_record}_gelman_rubin_statistic",
                    gelman_rubin_statistic,
                    step,
                )
                tb_writer.add_histogram(
                    f"{tb_tag_one_record}_within_chain_var", within_chain_var, step
                )
                tb_writer.add_histogram(
                    f"{tb_tag_one_record}_between_chain_var", between_chain_var, step
                )
                tb_writer.add_histogram(
                    f"{tb_tag_one_record}_weighted_sum_of_vars",
                    weighted_sum_of_vars,
                    step,
                )
            if num_unconverged_components > 0:
                logger.warning(log_str)
            return visible, hidden


hidden_num_vars = 12

visible_base_rate = train_dataset.data.mean(dim=0).clamp(1e-30, 1 - 1e-30)
model = HomogenousBinaryRBM(
    train_dataset.data.shape[1],
    hidden_num_vars,
    torch.log(visible_base_rate) - torch.log(1 - visible_base_rate),
)
print(f"ln(Z) = {model.log_normalization_constant()}")
tb_dir = mkdtemp()
print(f"tb_dir = {tb_dir}")
optimizer = SGD(model.parameters(), 5e-2, weight_decay=1e-10)
with tb.SummaryWriter(tb_dir) as tb_writer:
    for iteration in range(10001):
        if iteration % 1000 == 0:
            gibbs_sampling_tag_one_record = "gibbs_sampling_result"
            visible, _ = model.gibbs_sample(
                batch_size=128,
                num_steps=6000,
                tb_writer=tb_writer,
                tb_tag_many_records=f"gibbs_at_start_of_iter_{iteration}",
                tb_tag_one_record=gibbs_sampling_tag_one_record,
            )
            tb_writer.add_image(
                gibbs_sampling_tag_one_record,
                train_dataset.extract_images(visible[0]).squeeze(),
                iteration,
                dataformats="HW",
            )
        train_ll = model.log_marginal_likelihood_of_visible(train_dataset.data)
        train_mean_ll = torch.mean(train_ll)
        tb_writer.add_scalar("train_mean_ll", train_mean_ll, iteration)
        optimizer.zero_grad()
        (-train_mean_ll).backward()
        optimizer.step()
        if iteration % 100 == 0:
            print(f"Iteration {iteration} done, mean_ll = {train_mean_ll.item()}")
