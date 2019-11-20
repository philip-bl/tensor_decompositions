import logging
import os.path
from tempfile import mkdtemp

import torch
import torch.utils.tensorboard as tb
from torch.optim import SGD

from rbm.mnist import HomogenousBinaryMNIST, IMG_SIDE_LEN
from rbm.rbm import HomogenousBinaryRBM

logger = logging.getLogger(__name__)

dataset_root = os.path.expanduser("/mnt/hdd_1tb/datasets/mnist/dataset_source")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"device = {device}")

train_dataset = HomogenousBinaryMNIST(dataset_root, train=True)
hidden_num_vars = 12
model = HomogenousBinaryRBM(
    train_dataset.data.shape[1],
    hidden_num_vars,
    dataset_for_visible_bias_init=train_dataset,
).to(device)

# we can check whether log_conditional_visible_likelihood does utter nonsense
# by loading an unsupervised trained model, making predictions by using it and
# comparing their quality with predictions made by a randomly initialized model
# (I've checked, looks plausible)
# model.load_state_dict(torch.load(os.path.expanduser("~/archive/experiments/2019-11_rbm_binary_small_mnist/h=12_trainmll=-162.83.pth")))
input_num_vars = IMG_SIDE_LEN ** 2

optimizer = SGD(model.parameters(), 1.0, weight_decay=1e-9)
tb_dir = mkdtemp()
print(f"tb_dir = {tb_dir}")
mnist_all_possible_y = torch.eye(len(train_dataset.classes)).to(device)
condition = train_dataset.data[:, :input_num_vars].to(device)
what = train_dataset.data[:, input_num_vars:].to(device)
with tb.SummaryWriter(tb_dir) as tb_writer:
    for iteration in range(300001):
        train_mean_conditional_visible_ll = torch.mean(
            model.log_conditional_visible_likelihood(
                what, condition, mnist_all_possible_y
            )
        )
        tb_writer.add_scalar(
            "train_mean_conditional_visible_log_likelihood",
            train_mean_conditional_visible_ll,
            iteration,
        )
        train_geometric_mean_conditional_visible_l = torch.exp(train_mean_conditional_visible_ll)
        tb_writer.add_scalar(
            "train_geometric_mean_conditional_visible_likelihood", train_geometric_mean_conditional_visible_l,
            
            iteration,
        )
        optimizer.zero_grad()
        (-train_mean_conditional_visible_ll).backward()
        optimizer.step()
        if iteration % 1 == 0:
            print(f"Iteration {iteration} done, train_geometric_mean_conditional_visible_likelihood = {train_geometric_mean_conditional_visible_l.item()}")
