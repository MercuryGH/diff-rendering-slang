import torch
import timeit
import numpy as np
from torch.autograd import Function
from renderers.module import soft_ras

from renderers.camera import PerspectiveCamera
from renderers.transform import Transform
from utils.mesh import Mesh
from renderers.light import PointLight

BLOCK_SIZE = (16, 16, 1)


def divide_and_round_up(a: int, b: int):
    return (a + b - 1) // b


class SoftRas(Function):
    @staticmethod
    def forward(
        ctx,
        camera: PerspectiveCamera,
        mesh: Mesh,
        transform: Transform,
        image: torch.Tensor,
        params,
    ):
        ctx.save_for_backward(mesh.face_vertices)
        ctx.params = params
        # (y, x, 3) to align with the behavior of plt.imshow
        original_shape = (camera.height, camera.width, 3)
        output = torch.zeros(original_shape, dtype=torch.float).cuda()

        soft_ras.Main(
            camera=camera.serialize(),
            mesh=mesh.serialize(),
            transform=transform.serialize(),
            texture0={"image": image},
            output=output,
            params=params,  # directly pass the RenderParams instance
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=(
                divide_and_round_up(camera.width, BLOCK_SIZE[0]),
                divide_and_round_up(camera.height, BLOCK_SIZE[1]),
                1,
            ),
        )

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
