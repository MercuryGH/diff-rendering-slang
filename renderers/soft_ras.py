import torch
import timeit
import numpy as np
from torch.autograd import Function
from renderers.module import soft_ras

from renderers.camera import PerspectiveCamera
from renderers.transform import Transform
from utils.mesh import Mesh
from renderers.light import PointLight
from renderers.material import CookTorrance

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
        light: PointLight,
        material: CookTorrance,
        image: torch.Tensor,
        bg_color: torch.Tensor,
        params,
    ):
        # (y, x, 3) to align with the behavior of plt.imshow
        original_shape = (camera.height, camera.width, 3)
        output = torch.zeros(original_shape, dtype=torch.float).cuda()

        soft_ras.Main(
            camera=camera.serialize(),
            mesh=mesh.serialize(),
            transform=transform.serialize(),
            light=light.serialize(),
            material=material.serialize(),
            texture0={"image": image},
            output=output,
            bg_color=bg_color,
            params=params,  # directly pass the RenderParams instance
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=(
                divide_and_round_up(camera.width, BLOCK_SIZE[0]),
                divide_and_round_up(camera.height, BLOCK_SIZE[1]),
                1,
            ),
        )

        ctx.params = params
        ctx.camera = camera
        ctx.mesh = mesh
        ctx.transform = transform
        ctx.light = light
        ctx.material = material
        ctx.image = image
        ctx.save_for_backward(bg_color, output)

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        bg_color, output = ctx.saved_tensors

        grad_bg_color = torch.zeros_like(bg_color)
        grad_output = grad_output.contiguous()

        soft_ras.Main.bwd(
            camera=ctx.camera.serialize(),
            mesh=ctx.mesh.serialize(),
            transform=ctx.transform.serialize(),
            light=ctx.light.serialize(),
            material=ctx.material.serialize(),
            texture0={"image": ctx.image},
            output=(output, grad_output),
            bg_color=(bg_color, grad_bg_color),
            params=ctx.params,  # Pass the stored params
        ).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=(
                divide_and_round_up(output.shape[1], BLOCK_SIZE[0]),
                divide_and_round_up(output.shape[0], BLOCK_SIZE[1]),
                1,
            ),
        )

        return None, None, None, None, None, None, grad_bg_color, None
