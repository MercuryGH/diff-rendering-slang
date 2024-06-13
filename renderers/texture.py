import slangtorch
from torch.autograd import Function
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# from resources.resource import img, normal, roughness

texture_module = slangtorch.loadModule("shaders/soft_ras/texture.slang", verbose=True)

BLOCK_SIZE = (16, 16, 1)


def divide_and_round_up(a: int, b: int):
    return (a + b - 1) // b


class TestTexture(Function):
    @staticmethod
    def forward(ctx, image):
        ctx.save_for_backward(image)

        print(image.shape)

        output = torch.zeros(
            (image.shape[0], image.shape[1], 3), dtype=torch.float
        ).cuda()

        texture_module.TestTexture(texture={"image": image}, output=output).launchRaw(
            blockSize=BLOCK_SIZE,
            gridSize=(
                divide_and_round_up(image.shape[1], BLOCK_SIZE[0]),
                divide_and_round_up(image.shape[0], BLOCK_SIZE[1]),
                1,
            ),
        )

        return output


img = np.divide(cv2.imread(R"resources/diffuse.jpg")[:,:,[2,1,0]], 255)

dispatcher = TestTexture()
print(img.shape)
result = dispatcher.apply(torch.tensor(img, dtype=torch.float).cuda())

plt.imsave("output_test_texture.png", result.cpu().numpy()) # type: ignore
plt.show()