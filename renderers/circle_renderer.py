import torch
import timeit
import numpy as np
from torch.autograd import Function
from renderers.module import rasterizer2d

class CircleRenderer(Function):
    @staticmethod
    def forward(ctx, width, height, camera, sigma, pos, radius, color):
        output = torch.zeros((width, height, 3), dtype=torch.float).cuda()

        rasterizer2d.rasterize_circle(
            camera=camera,
            pos=pos,
            radius=radius,
            color=color,
            output=output
        ).launchRaw(
            blockSize=(16, 16, 1),
            gridSize=((width + 15)//16, (height + 15)//16, 1)
        )

        ctx.camera = camera
        ctx.sigma = sigma
        ctx.save_for_backward(pos, radius, color, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        pos, radius, color, output = ctx.saved_tensors
        camera = ctx.camera
        sigma = ctx.sigma

        grad_pos = torch.zeros_like(pos)
        grad_radius = torch.zeros_like(radius)
        grad_color = torch.zeros_like(color)
        grad_output = grad_output.contiguous()

        width, height = grad_output.shape[:2]

        start = timeit.default_timer()

        rasterizer2d.rasterize_circle.bwd(
            # invariant input
            camera=camera,
            # differentiable inputs
            pos=(pos, grad_pos),
            radius=(radius, grad_radius),
            color=(color, grad_color),
            # outputs
            output=(output, grad_output)
        ).launchRaw(
            blockSize=(16, 16, 1),
            gridSize=((width + 15)//16, (height + 15)//16, 1)
        )

        end = timeit.default_timer()

        print("Backward pass: %f seconds" % (end - start))

        return None, None, None, None, grad_pos, grad_radius, grad_color
