import itertools
import logging
from typing import *

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
from einops import rearrange
from torch.distributions import Bernoulli

from .mnist import HomogenousBinaryMNIST


logger = logging.getLogger()


def gen_all_binary_vectors(length: int) -> torch.Tensor:
    return (
        (torch.arange(2 ** length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1
    ).float()


def log_1_plus_exp(x: torch.Tensor) -> torch.Tensor:
    """Applies elementwise the function f(x) = ln(1 + exp(x_i))."""
    result = torch.logsumexp(torch.stack((x, torch.zeros_like(x)), dim=-1), dim=-1)
    assert result.shape == x.shape
    return result


class HomogenousBinaryRBM(nn.Module):
    def __init__(
        self,
        visible_num_vars: int,
        hidden_num_vars: int,
        visible_bias_init: Optional[torch.Tensor] = None,
        dataset_for_visible_bias_init: Optional[HomogenousBinaryMNIST] = None,
    ):
        super().__init__()
        self.visible_num_vars = visible_num_vars
        self.hidden_num_vars = hidden_num_vars
        self.W = nn.Parameter(torch.randn(visible_num_vars, hidden_num_vars) * 1e-4)
        self.visible_bias = nn.Parameter(torch.randn(self.visible_num_vars) * 1e-4)
        self.hidden_bias = nn.Parameter(torch.randn(self.hidden_num_vars) * 1e-4)
        if dataset_for_visible_bias_init is not None:
            assert visible_bias_init is None
            visible_bias_init = self.calc_visible_bias_init(
                dataset_for_visible_bias_init, 1e-30
            )
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

    def log_conditional_visible_likelihood(
        self,
        what: torch.Tensor,
        condition: torch.Tensor,
        all_what: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates ln p(what|condition) = ln p(what, condition) - ln \sum_{what'} p(what', condition). The sum is
        performed over all_what. If all_what is not passed, the sum is performed over all binary vectors of
        matching length. all_what must be binary and contain what.
        It is assumed that condition is the first part of visible, and
        what is the last part of visible. Sure it can be done any other way in principle,
        but fuck writing additional code."""
        # how to calculate it
        # concat all_what with condition while respecting the batch - get a "visible"
        # calculate unnormalized log likelihood of "visible"
        # calculate the result using logsumexp
        batch_size, what_size = what.shape
        num_all_what = all_what.shape[0]
        condition_size = condition.shape[1]
        assert condition_size + what_size == self.visible_num_vars
        if all_what is None:
            all_what = gen_all_binary_vectors(what_size)
        all_what_duplicated = torch.stack(tuple(itertools.repeat(all_what, batch_size)))
        assert all_what_duplicated.shape == (batch_size, num_all_what, what_size)
        condition_duplicated = rearrange(
            torch.stack(tuple(itertools.repeat(condition, num_all_what))),
            "a b n -> b a n",
            a=num_all_what,
            b=batch_size,
        )
        visible = torch.cat((condition_duplicated, all_what_duplicated), dim=2)
        assert visible.shape == (batch_size, num_all_what, self.visible_num_vars)
        unnormalized_ll_of_all = rearrange(
            self.unnormalized_log_likelihood(rearrange(visible, "b a n -> (b a) n")),
            "(b a)-> b a",
            b=batch_size,
        )
        # now we look for index of each what[b] in all_what
        what_match = torch.all(what.unsqueeze(0) == all_what.unsqueeze(1), dim=-1)
        # what_match[a, b] is True iff all_what[a] is the same as what[b]
        assert torch.all(what_match.sum(dim=0) == 1)
        unnormalized_ll = torch.einsum(
            "ab,ba->b", what_match.float(), unnormalized_ll_of_all
        )
        # unnormalized_ll[b] equals unnormalized_ll(condition[b], what[b])
        unnormalized_marginal_ll_of_condition = unnormalized_ll_of_all.logsumexp(dim=1)
        assert unnormalized_ll.shape == unnormalized_marginal_ll_of_condition.shape
        return unnormalized_ll - unnormalized_marginal_ll_of_condition

    def unnormalized_likelihood(
        self, visible: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp(self.unnormalized_log_likelihood(visible, hidden))

    def log_unnormalized_marginal_likelihood_of_visible(
        self, visible: torch.Tensor
    ) -> torch.Tensor:
        foo = torch.einsum("bn,n->b", visible, self.visible_bias)
        bar = self.hidden_bias + torch.einsum("nm,bn->bm", self.W, visible)
        buzz = torch.sum(log_1_plus_exp(bar), dim=1)
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
        buzz = torch.sum(log_1_plus_exp(bar), dim=1)
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
        hidden_base_rate: float = 0.5,
    ) -> torch.Tensor:
        assert batch_size > 0
        assert num_steps >= 2
        with torch.no_grad():
            hidden = Bernoulli(hidden_base_rate).sample(
                (batch_size, self.hidden_num_vars)
            )
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
                        HomogenousBinaryMNIST.extract_images(
                            p_vis_given_hid.mean[0]
                        ).squeeze(),
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
                tb_writer.add_text(f"{tb_tag_one_record}", log_str, step)
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

    @staticmethod
    def calc_visible_bias_init(
        train_dataset: HomogenousBinaryMNIST, eps: float
    ) -> torch.Tensor:
        visible_base_rate = train_dataset.data.mean(dim=0).clamp(eps, 1 - eps)
        return torch.log(visible_base_rate) - torch.log(1 - visible_base_rate)

# TODO: add visualization (as an image) of hidden during gibbs sampling
# TODO: add visualization (as an image) of what each hidden unit does
