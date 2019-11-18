import os.path
from tempfile import mkdtemp

import torch
import torch.utils.tensorboard as tb
from torch.optim import SGD

from rbm.mnist import HomogenousBinaryMNIST, IMG_SIDE_LEN
from rbm.rbm import HomogenousBinaryRBM


dataset_root = os.path.expanduser("~/archive/datasets/mnist/pytorch_root/")
train_dataset = HomogenousBinaryMNIST(dataset_root, train=True)
hidden_num_vars = 12
model = HomogenousBinaryRBM(
    train_dataset.data.shape[1],
    hidden_num_vars,
    dataset_for_visible_bias_init=train_dataset,
)

# we can check whether log_conditional_visible_likelihood does utter nonsense
# by loading an unsupervised trained model, making predictions by using it and
# comparing their quality with predictions made by a randomly initialized model
# (I've checked, looks plausible)
model.load_state_dict(torch.load(os.path.expanduser("~/archive/experiments/2019-11_rbm_binary_small_mnist/h=12_trainmll=-162.83.pth")))
input_num_vars = IMG_SIDE_LEN**2
batch = train_dataset[:128]
condition = batch[:, :input_num_vars]
what = batch[:, input_num_vars:]
foo = model.log_conditional_visible_likelihood(what, condition)

print(f"ln(Z) = {model.log_normalization_constant()}")
optimizer = SGD(model.parameters(), 5e-2, weight_decay=1e-9)
tb_dir = mkdtemp()
print(f"tb_dir = {tb_dir}")
# with tb.SummaryWriter(tb_dir) as tb_writer:
# TODO WRITE THIS
