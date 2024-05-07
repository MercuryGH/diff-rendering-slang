import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from renderers.soft_ras import SoftRas
from resources.resource import spot_obj
from utils.util import wrap_float_tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# class RenderParams:
#     def __init__(self, width, height, sigma):
#         self.width = width
#         self.height = height
#         self.sigma = sigma
#     def to_dict(self):
#         return {'width': self.width, 'height': self.height, 'sigma': self.sigma}

def single_forward(renderer, width, height, face_vertices, RenderParams):
    # Create an instance of RenderParams
    # params = RenderParams(width, height, sigma)
    # Pass it to the forward function
    output = renderer.apply(width, height, face_vertices, RenderParams)  # output is a tensor
    return output


def main():
    renderer = SoftRas()
    width = height = 512
    sigma = 0.5  # Choose an appropriate sigma value

    face_vertices = spot_obj.face_vertices
    params = {'width': width, 'height': height, 'sigma': sigma} 
    output = single_forward(renderer, width, height, face_vertices, params)
    plt.rcParams['figure.figsize'] = (20, 20)
    plt.imshow(output.cpu().numpy())
    plt.show()


if __name__ == '__main__':
    main()