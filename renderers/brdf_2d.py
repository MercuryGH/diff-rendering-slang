import torch
import timeit
import numpy as np
from torch.autograd import Function
from renderers.module import light_kernel

def calc_grid_size(shape, threads_dim=16):
    return ((shape[0] + threads_dim - 1)//threads_dim, (shape[1] + threads_dim - 1)//threads_dim, 1)

BLOCK_SIZE = (16, 16, 1)

class Brdf2d(Function):
    @staticmethod
    def forward(ctx, width, height, ref_brdf, input_params, half_res_brdf):
        original_shape = (width, height, 3)
        output = torch.zeros(original_shape, dtype=torch.float).cuda()

        grid_size = calc_grid_size(original_shape)

        light_kernel.brdf(
            input=ref_brdf,
            output=output,
            input_params=input_params,
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=grid_size
        )

        if ctx is not None:
            ctx.input_params = input_params
            ctx.save_for_backward(half_res_brdf, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        half_res_brdf, lighting_from_full_res_brdf = ctx.saved_tensors
        input_params = ctx.input_params

        grad_half_res_brdf = torch.zeros_like(half_res_brdf)
        grad_output = grad_output.contiguous()

        width, height = grad_output.shape[:2]
        original_shape = (width, height, 1)
        loss_output = torch.zeros(original_shape).cuda()
        grid_size = calc_grid_size(original_shape)

        start = timeit.default_timer()

        light_kernel.brdf_loss.bwd(
            # invariant input
            input_params=input_params,
            # inputs from forward
            reference=lighting_from_full_res_brdf,
            # differentiable inputs
            input=(half_res_brdf, grad_half_res_brdf),
            # output
            output=(loss_output, grad_output)
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=grid_size
        )

        end = timeit.default_timer()

        # print("Backward pass: %f seconds" % (end - start))

        return None, None, None, None, grad_half_res_brdf
