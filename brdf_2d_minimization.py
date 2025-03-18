import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import cv2
from renderers.brdf_2d import Brdf2d
from resources.resource import img, normal, roughness
from utils.util import wrap_float_tensor, set_grad

def training_loop(i,
    optimizer, full_res_brdf, renderer,
    width, height, view_vector, half_res_brdf
    ):

    # Note: this is not a uniform distribution
    def random_hemi_vector():
        a, b, c = random.normalvariate(0, 1), random.normalvariate(0, 1), random.normalvariate(0, 1)
        l = (np.sqrt(a**2 + b**2 + c**2) + 0.0001)
        a, b, c = a/l, b/l, c/l
        c = abs(c)
        return (a, b, c)

    light_vector = random_hemi_vector()

    input_params = (*light_vector, *view_vector)

    if i % 100 == 0:
        print("Iteration %d" % i)

    output_loss = renderer.apply(width, height, full_res_brdf, input_params, half_res_brdf)

    # register loss grad hook if necessary
    # output_loss.register_hook(set_grad(output_loss))

    # Compute the loss.
    loss = torch.mean(output_loss)

    # Backward pass: compute the gradients.
    loss.backward()

    # Update the parameters.
    optimizer.step()

    # Zero the gradients.
    optimizer.zero_grad()

def single_forward(renderer: Brdf2d, width, height, input_params, brdf):
    output = renderer.forward(ctx=None, width=width, height=height, input_params=input_params, ref_brdf=brdf, half_res_brdf=None)
    return output

def train():
    renderer = Brdf2d()

    # Setup our training loop.
    lr = 0.001
    n_itr = 10000

    brdf_input_torch = torch.tensor(np.concatenate((img, normal, roughness[...,[0]]), axis=-1), dtype=torch.float).cuda().contiguous()
    full_res_brdf = brdf_input_torch.clone()

    width = brdf_input_torch.shape[0]
    height = brdf_input_torch.shape[1]

    half_res_brdf = torch.tensor(cv2.resize(brdf_input_torch.cpu().numpy(), None, fx=0.5, fy=0.5), dtype=torch.float).cuda()
    half_res_brdf_before_optim = half_res_brdf.clone()

    view_vector = (0.0, 0.0, 1.0)

    # Setup our optimizer.
    half_res_brdf = wrap_float_tensor(half_res_brdf, True)
    optimizer = torch.optim.Adam([half_res_brdf], lr=lr)

    for i in range(0, n_itr):
        training_loop(i, optimizer, full_res_brdf, renderer=renderer, width=width, height=height, view_vector=view_vector, half_res_brdf=half_res_brdf)

    # View the result.
    output_shape = (width, height, 3)
    light_vector = (0.2, -0.2, 0.6)

    # Initialize the output buffers and final display parameters.
    input_params = (*light_vector, *view_vector)
    lighting_from_full_res_brdf = torch.zeros(output_shape).cuda()
    lighting_from_half_res_brdf = torch.zeros(output_shape).cuda()
    lighting_before_optim = torch.zeros(output_shape).cuda()

    # Run the lighting passes.
    lighting_from_full_res_brdf = single_forward(renderer, width, height, input_params, full_res_brdf)
    lighting_from_half_res_brdf = single_forward(renderer, width, height, input_params, half_res_brdf)
    lighting_before_optim = single_forward(renderer, width, height, input_params, half_res_brdf_before_optim)

    # Plotting.
    def display(x):
        return np.clip(2*x, 0, 1)

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.imshow(display((np.block([[[lighting_before_optim.cpu().numpy()],
                                [lighting_from_half_res_brdf.cpu().numpy()]],
                                [[lighting_from_full_res_brdf.cpu().numpy()],
                                [0.5*np.abs((lighting_from_full_res_brdf - lighting_from_half_res_brdf).cpu().numpy())]]]))))
    plt.text(10, 20, "Naively downsampled BRDF", fontsize='x-large', color='white', fontweight='bold')
    plt.text(150, 20, "Optimized BRDF", fontsize='x-large', color='white', fontweight='bold')
    plt.text(10, 140, "Reference", fontsize='x-large', color='white', fontweight='bold')
    plt.text(150, 150, "Optimized and reference\ndifference", fontsize='x-large', color='white', fontweight='bold')
    plt.axis("off")
    plt.show()
    plt.rcParams['figure.figsize'] = (12, 12)

    half_res_brdf = wrap_float_tensor(half_res_brdf, False)

    a, b, c = (np.vstack((half_res_brdf_before_optim.cpu()[..., 0:3], half_res_brdf.cpu()[..., 0:3])),
            np.vstack((half_res_brdf_before_optim.cpu()[..., 3:6], half_res_brdf.cpu()[..., 3:6])),
            np.vstack((half_res_brdf_before_optim.cpu()[..., [6,6,6]], half_res_brdf.cpu()[..., [6,6,6]])))
    plt.imshow(np.hstack((a,b,c)))
    plt.axis("off")
    plt.show()

def main():
    train()

if __name__ == '__main__':
    main()

'''
TODO:

实现 toon shader 的 shader params 自动调参，最小化结果与参考图的差值
输入的相机、模型参数应当完全一致，shader 参数供学习
'''
