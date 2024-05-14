import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from renderers.soft_ras import SoftRas
from resources.resource import capsule_obj
from utils.util import wrap_float_tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def single_forward(
    renderer, width, height, face_vertices, RenderParams
) -> torch.Tensor:
    # Create an instance of RenderParams
    # params = RenderParams(width, height, sigma)
    # Pass it to the forward function
    output = renderer.apply(width, height, face_vertices, RenderParams)
    return output


def main():
    renderer = SoftRas()
    width = height = 512
    sigma = 1e-4  # Choose an appropriate sigma value

    face_vertices = capsule_obj.face_vertices
    params = {
        "width": width,
        "height": height,
        "sigma": sigma,
        "epsilon": 1e-3,
        "gamma": 1e-4,
        "distance_epsilon": 1e-5,
        "fg_color": [0.7, 0.8, 0.9],
        "bg_color": [0.9, 0.9, 0.9],
    }
    output = single_forward(renderer, width, height, face_vertices, params)
    print(output[0, 0, :])
    plt.rcParams["figure.figsize"] = (width, height)
    plt.imshow(output.cpu().numpy())
    plt.show()


if __name__ == "__main__":
    main()
