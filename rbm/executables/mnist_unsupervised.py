import os.path
from tempfile import mkdtemp

import torch
import torch.utils.tensorboard as tb
from torch.optim import SGD

from rbm.mnist import HomogenousBinaryMNIST
from rbm.rbm import HomogenousBinaryRBM


dataset_root = os.path.expanduser("~/archive/datasets/mnist/pytorch_root/")
train_dataset = HomogenousBinaryMNIST(dataset_root, train=True)
hidden_num_vars = 12
model = HomogenousBinaryRBM(
    train_dataset.data.shape[1],
    hidden_num_vars,
    dataset_for_visible_bias_init=train_dataset,
)

print(f"ln(Z) = {model.log_normalization_constant()}")
tb_dir = mkdtemp()
print(f"tb_dir = {tb_dir}")
optimizer = SGD(model.parameters(), 5e-2, weight_decay=1e-9)
with tb.SummaryWriter(tb_dir) as tb_writer:
    for iteration in range(300001):
        if iteration % 1000 == 0:
            gibbs_sampling_tag_one_record = "gibbs_sampling_result"
            visible, _ = model.gibbs_sample(
                batch_size=128,
                num_steps=600,
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
