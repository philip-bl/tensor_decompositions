import torch
import torch.nn as nn

def free_cuda_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

class ShapyLinear(nn.Module):
    """Can model any affine function from the set of tensors of any (fixed) shape `in_shape` to
    the set of tensors of any (fixed) shape `out_shape`.
    
    In forward method the first modes of `inputs` are interpreted as indices of samples,
    then come the modes corresponding to `in_shape`. The affine function is applied to each sample."""
    def __init__(self, in_shape, out_shape):
        """:param in_shape: shape of one input sample
        :param out_shape: shape of one output sample"""
        super().__init__()
        in_shape = tuple(in_shape)
        out_shape = tuple(out_shape)
        self.input_num_modes = len(in_shape)
        
        weight_data = torch.zeros(*in_shape, *out_shape, requires_grad=True)
        nn.init.xavier_normal_(weight_data)
        self.weight = nn.Parameter(weight_data)
        
        self.bias = nn.Parameter(torch.zeros(*out_shape, requires_grad=True))
    
    @property
    def weight_contraction_modes(self):
        """Indices of modes of `self.weight` over which tensor contraction with input is performed."""
        return tuple(range(self.input_num_modes))
    
    def forward(self, inputs):
        # calculate how many modes of `inputs` represent indices of samples
        num_sample_modes = inputs.ndimension() - self.input_num_modes
        
        # calculate over what modes of `inputs` we perform tensor contraction
        inputs_contraction_modes = tuple(range(num_sample_modes, inputs.ndimension()))
        
        foo = torch.tensordot(inputs, self.weight, dims=(inputs_contraction_modes, self.weight_contraction_modes))
        return foo + self.bias
    
    def __repr__(self):
        return f"ShapyLinear(input_num_modes={self.input_num_modes}, weight.shape={tuple(self.weight.shape)})"
