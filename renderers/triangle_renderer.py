import torch
import timeit
import numpy as np
from torch.autograd import Function
from renderers.module import rasterizer2d

class TriangleRenderer(Function):
    @staticmethod
    def forward(ctx, width, height, camera, sigma, vertices, color):
        output = torch.zeros((width, height, 3), dtype=torch.float).cuda()
        rasterizer2d.rasterize_triangle(
            camera=camera,
            vertices=vertices,
            color=color,
            output=output
        ).launchRaw(
            blockSize=(16, 16, 1),
            gridSize=((width + 15)//16, (height + 15)//16, 1)
        )

        ctx.camera = camera
        ctx.sigma = sigma
        ctx.save_for_backward(vertices, color, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        vertices, color, output = ctx.saved_tensors
        camera = ctx.camera
        sigma = ctx.sigma

        grad_vertices = torch.zeros_like(vertices)
        grad_color = torch.zeros_like(color)
        grad_output = grad_output.contiguous()

        width, height = grad_output.shape[:2]

        start = timeit.default_timer()

        rasterizer2d.rasterize_triangle.bwd(
            camera=camera,
            vertices=(vertices, grad_vertices),
            color=(color, grad_color),
            output=(output, grad_output)
        ).launchRaw(
            blockSize=(16, 16, 1),
            gridSize=((width + 15)//16, (height + 15)//16, 1)
        )

        end = timeit.default_timer()

        print("Backward pass: %f seconds" % (end - start))

        return None, None, None, None, grad_vertices, grad_color
