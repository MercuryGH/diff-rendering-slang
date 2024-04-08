import torch

def wrap_float_tensor(data, requires_grad=False):
    tensor = torch.tensor(data).type(torch.float).cuda()
    tensor.requires_grad = requires_grad
    return tensor

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def calc_grid_size(shape, threads_dim=16):
    return ((shape[0] + threads_dim - 1)//threads_dim, (shape[1] + threads_dim - 1)//threads_dim, 1)
