import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from renderers.soft_ras import SoftRas
from resources.resource import spot_obj
from utils.util import wrap_float_tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def single_forward(renderer: SoftRas, width, height, face_vertices):
    output = renderer.forward(ctx=None, width=width, height=height, face_vertices=face_vertices)
    return output

def main():
    renderer = SoftRas()
    width = height = 20

    face_vertices = spot_obj.face_vertices
    # print(face_vertices.size()) # torch.Size([1, 5856, 3, 3])

    output = single_forward(renderer, width, height, face_vertices)
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.imshow(output.cpu().numpy())
    plt.show()

if __name__ == '__main__':
    main()
