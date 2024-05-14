import torch
import timeit
import math
import numpy as np
from torch.autograd import Function
from renderers.module import soft_ras
from utils.util import calc_grid_size

from renderers.camera import PerspectiveCamera

BLOCK_SIZE = (16, 16, 1)

camera = PerspectiveCamera(
    eye=[0.0, 0.0, 3.0],
    dir=[0.0, 0.0, -1.0],
    up=[0.0, 1.0, 0.0],
    fov=45 / 180 * math.pi,
    aspect=1,
    near=0.1,
    far=100,
)


class SoftRas(Function):
    @staticmethod
    def forward(ctx, width, height, face_vertices, params):
        ctx.save_for_backward(face_vertices)
        ctx.params = params
        original_shape = (width, height, 3)
        output = torch.zeros(original_shape, dtype=torch.float).cuda()

        grid_size = calc_grid_size(original_shape)

        soft_ras.main(
            camera=(
                camera.eye,
                camera.dir,
                camera.up,
                camera.fov,
                float(width) / float(height),
                camera.near,
                camera.far,
            ),
            face_vertices=face_vertices,
            output=output,
            params=params,  # directly pass the RenderParams instance
        ).launchRaw(blockSize=BLOCK_SIZE, gridSize=grid_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (face_vertices,) = ctx.saved_tensors
        params = ctx.params
        grad_face_vertices = torch.zeros_like(face_vertices)
        grad_output = grad_output.contiguous()

        width, height = grad_output.shape[:2]

        soft_ras.backward_stub(
            vertices=(face_vertices, grad_face_vertices),
            output_grad=(grad_output, None),
            params=params,  # Pass the stored params
        ).launchRaw(
            blockSize=BLOCK_SIZE, gridSize=((width + 15) // 16, (height + 15) // 16, 1)
        )

        return None, None, grad_face_vertices, None
