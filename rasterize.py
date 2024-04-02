import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from renderers.module import camera
from renderers.circle_renderer import CircleRenderer
from renderers.triangle_renderer import TriangleRenderer
from utils.util import wrap_float_tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def render_target_result(vertices, color, camera, sigma):
    # Render a simple target image.
    targetVertices = wrap_float_tensor(vertices)
    targetColor = wrap_float_tensor(color)
    targetImage = TriangleRenderer().apply(1024, 1024, camera, sigma, targetVertices, targetColor)
    return targetImage

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def training_loop(i,
    optimizer: torch.optim.Adam, target_image, circle_renderer: CircleRenderer, plot_axes,
    sigma, pos, radius, color
    ):
    print("Iteration %d" % i)

    output_image = circle_renderer.apply(1024, 1024, camera, sigma, pos, radius, color)

    output_image.register_hook(set_grad(output_image))

    # Compute the loss.
    loss = torch.mean((output_image - target_image) ** 2)

    # Backward pass: compute the gradients.
    loss.backward()

    # Update the parameters.
    optimizer.step()

    if i % 10 == 0:
        plot_axes[0].clear()
        plot_axes[0].imshow(output_image.permute(1, 0, 2).detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])
        plot_axes[1].clear()
        plot_axes[1].imshow(output_image.grad[:,:,1].T.detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])
        plot_axes[2].clear()
        plot_axes[2].imshow(target_image.permute(1, 0, 2).detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])

    # Zero the gradients.
    optimizer.zero_grad()

import matplotlib.animation as animation
from functools import partial

def main():
    circle_renderer = CircleRenderer()

    SIGMA = 0.02

    target_vertices = [[0.7,-0.3], [-0.3,0.2], [-0.9,-0.9]]
    target_color = [0.3, 0.8, 0.3]
    target_image = render_target_result(target_vertices, target_color, camera, SIGMA)

    # Setup our training loop.
    lr = 5e-3
    n_itr = 200

    # Initialize our parameters.
    pos = wrap_float_tensor([[-0.5, -0.5]], True)
    radius = wrap_float_tensor(0.5, True)
    color = wrap_float_tensor([0.8, 0.3, 0.3], True)

    # Setup our optimizer.
    optimizer = torch.optim.Adam([pos, radius, color], lr=lr)

    # Setup plot
    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    plot_axes = [ax1, ax2, ax3]

    ani = animation.FuncAnimation(fig, partial(
        training_loop,
        # itr params
        optimizer=optimizer,
        target_image=target_image,
        circle_renderer=circle_renderer,
        plot_axes=plot_axes,
        # shader params
        sigma=SIGMA,
        pos=pos,
        radius=radius,
        color=color
    ), frames=n_itr, interval=10)

    writer = animation.FFMpegWriter(fps=30)
    ani.save('rasterize.mp4', writer=writer)

if __name__ == '__main__':
    main()
