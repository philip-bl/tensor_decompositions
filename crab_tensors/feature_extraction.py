import numpy as np
import scipy.linalg as la

import torch

import tensorly as tl
from tensorly.decomposition import partial_tucker
import tensorly.tenalg as ta

class TuckerFeatureExtractor(object):
    def __init__(self, input_shape, output_shape):
        assert len(input_shape) == len(output_shape)
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)
        self.modes_to_compress = [
            mode+1 for mode, in_dim, out_dim
            # add plus one coz 0 mode will be samples
            in zip(
                range(len(input_shape)),
                input_shape,
                output_shape
            )
            if in_dim != out_dim
        ]
        self.ranks = [
            output_shape[mode-1]
            for mode in self.modes_to_compress
        ]
        self.was_fit = False
    
    def fit_transform(self, X):
        assert tuple(X.shape[1:]) == self.input_shape
        core, self.factors = partial_tucker(
            X, self.modes_to_compress, self.ranks
        )
        assert tuple(core.shape[1:]) == self.output_shape
        assert core.shape[0] == X.shape[0]
        self.factors_inverses = [__class__._pseudoinverse(factor) for factor in self.factors]
        self.was_fit = True
        return core
    
    def transform(self, X):
        assert tuple(X.shape[1:]) == self.input_shape
        return ta.multi_mode_dot(X, self.factors_inverses, self.modes_to_compress)
    
    def reconstruct(self, core):
        assert tuple(core.shape[1:]) == self.output_shape
        return ta.multi_mode_dot(core, self.factors, self.modes_to_compress)
    
    @staticmethod
    def _pseudoinverse(matrix):
        if isinstance(matrix, np.ndarray):
            return la.pinv(matrix)
        assert isinstance(matrix, torch.Tensor)
        return torch.pinverse(matrix)


def calc_error(orig, approx):
    orig = tl.to_numpy(orig)
    approx = tl.to_numpy(approx)
    abs_errors = np.abs(orig - approx)
    print(f"Mean absolute error = {abs_errors.mean()}")
    rel_errors = abs_errors / (np.abs(orig) + np.finfo(orig.dtype).eps)
    print(f"Mean relative error = {rel_errors.mean()}")
