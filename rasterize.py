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

    print("pos = ", pos)
    print("radius = ", radius)
    print("color = ", color)

    output_image = circle_renderer.apply(1024, 1024, camera, sigma, pos, radius, color)

    # output_image.register_hook(set_grad(output_image))
    output_image.retain_grad() 

    # Compute the loss.
    loss = torch.mean((output_image - target_image) ** 2)

    # Backward pass: compute the gradients.
    loss.backward()

    # Update the parameters.
    optimizer.step()

    if i % 1 == 0:
        def clear_all_frame(ax):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in plot_axes:
            clear_all_frame(ax)
        plot_axes[0].imshow(output_image.permute(1, 0, 2).detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])
        plot_axes[1].imshow(output_image.grad[:,:,1].T.detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])
        plot_axes[2].imshow(target_image.permute(1, 0, 2).detach().cpu().numpy(), origin='lower', extent=[-1, 1, -1, 1])

    # Zero the gradients.
    optimizer.zero_grad()

import matplotlib.animation as animation
from functools import partial
from PIL import Image

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
    fig = plt.figure(dpi=100)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    axes = [plt.Axes(fig, [i/3.0, 0.0, 1/3.0, 1.0]) for i in range(3)]
    for ax in axes:
        ax.set_axis_off()
        fig.add_axes(ax)
    plot_axes = axes

    # ax1 = fig.add_subplot(131)
    # ax1.tick_params(axis='both', labelsize=8)
    # ax2 = fig.add_subplot(132)
    # ax2.tick_params(axis='both', labelsize=8)
    # ax3 = fig.add_subplot(133)
    # ax3.tick_params(axis='both', labelsize=8)
    # plot_axes = [ax1, ax2, ax3]

    if not os.path.exists('temp_frames'):
        os.makedirs('temp_frames')

    def update_frame(i):
        training_loop(i, optimizer, target_image, circle_renderer, 
                     plot_axes, SIGMA, pos, radius, color)
        
        frame_path = f'temp_frames/frame_{i:04d}.png'
        fig.savefig(frame_path, bbox_inches='tight', pad_inches=0)
        
        img = Image.open(frame_path)
        bbox = img.getbbox()
        cropped = img.crop(bbox)
        cropped.save(frame_path)

    for i in range(n_itr):
        update_frame(i)

    # ani = animation.FuncAnimation(fig, partial(
    #     training_loop,
    #     # itr params
    #     optimizer=optimizer,
    #     target_image=target_image,
    #     circle_renderer=circle_renderer,
    #     plot_axes=plot_axes,
    #     # shader params
    #     sigma=SIGMA,
    #     pos=pos,
    #     radius=radius,
    #     color=color
    # ), frames=n_itr, interval=10)

    frames = [Image.open(f'temp_frames/frame_{i:04d}.png') for i in range(n_itr)]
    frames[0].save('rasterize.gif',
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=50,  # 控制帧率（50ms = 20fps）
                   loop=0,
                   disposal=2)  # 设置处置方法保证帧间清除


    # writer = animation.FFMpegWriter(fps=30)
    # ani.save('rasterize.mp4', writer=writer)
    # ani.save('rasterize.gif', writer=animation.PillowWriter(fps=30), savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0})

if __name__ == '__main__':
    main()
