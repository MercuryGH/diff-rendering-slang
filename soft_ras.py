import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from renderers.soft_ras import SoftRas
from resources.resource import spot_obj, cube_obj, uniform_albedo, grid_texture
from utils.util import wrap_float_tensor, set_grad
from renderers.camera import PerspectiveCamera
from renderers.light import PointLight
from utils.vectors import Vector3
from renderers.transform import Transform, rotate_to_quaternion
from renderers.material import CookTorrance
from IPython.display import HTML

WIDTH = 512
HEIGHT = 384

camera = PerspectiveCamera(
    eye=Vector3(3.0, 3.0, 4.0),
    # eye=Vector3(7.0, -7.0, 5.0),
    dir=Vector3(-0.7, -0.7, -1.0),
    up=Vector3(0.0, 1.0, 0.0),
    fov=60.0 / 180.0 * np.pi,
    near=0.1,
    far=100,
    width=WIDTH,
    height=HEIGHT,
)

# model = cube_obj
model = spot_obj

def training_loop(i,
    optimizer: torch.optim.Adam,
    target_image: torch.Tensor,
    renderer: SoftRas,
    plot_axes,
    bg_color: torch.Tensor,
    metallic: torch.Tensor,
):
    print("Iteration %d" % i)

    transform = Transform(
        rotation=rotate_to_quaternion(Vector3(0.0, 1.0, 1.0), 0.0)
        * rotate_to_quaternion(Vector3(1.0, 0.0, 0.0), 0.0),
        position=Vector3(0.0, 0.0, 0.0),
        scaling=Vector3(1.5, 1.5, 1.5),
    )

    point_light = PointLight(
        position=Vector3(1.0, 2.0, 3.0),
        color=Vector3(1.0, 1.0, 1.0),
        attenuation=Vector3(1.0, 0.0, 0.0),
    )

    cook_torrance = CookTorrance(
        roughness=0.6,
        metallic=0.1, # not actually works since the real metallic is being optimized
    )

    print("bg_color = ", bg_color)
    print("metallic = ", metallic)

    output_image: torch.Tensor = renderer.apply(  # type: ignore
        camera,
        model,
        transform,
        point_light,
        cook_torrance,
        wrap_float_tensor(uniform_albedo),
        bg_color,
        metallic,
        {
            "sigma": 1e-6,
            "epsilon": 1e-3,
            "gamma": 1e-4,
            "distance_epsilon": 1e-5,
            "ambient_light": [0.1, 0.1, 0.1],
            "bg_color": bg_color.tolist(),
            "gamma_correction": True,
        },
    )

    # output_image.register_hook(set_grad(output_image))
    output_image.retain_grad() 

    # Compute the loss.
    loss = torch.mean((output_image - target_image) ** 2)
    print(f"Loss: {loss}")

    # Backward pass: compute the gradients.
    loss.backward()

    # Update the parameters.
    optimizer.step()

    # Zero the gradients.
    optimizer.zero_grad()

    if i % 1 == 0:
        def clear_all_frame(ax):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in plot_axes:
            clear_all_frame(ax)
        plot_axes[0].imshow(output_image.detach().cpu().numpy(), origin='lower')
        plot_axes[1].imshow(output_image.grad[:,:,1].T.permute(1, 0).detach().cpu().numpy(), origin='lower')
        plot_axes[2].imshow(target_image.detach().cpu().numpy(), origin='lower')

from functools import partial
import matplotlib.animation as animation
from PIL import Image

def main():
    # Setup plot
    fig = plt.figure(dpi=100)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    ax1 = fig.add_axes([0.0, 0.0, 0.33, 1.0])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.33, 0.0, 0.34, 1.0])
    ax3 = fig.add_axes([0.67, 0.0, 0.33, 1.0])

    plot_axes = [ax1, ax2, ax3]

    target_ambient_texture = np.full((1, 1, 3), (0x66 / 0xff, 0xcc / 0xff, 1.0))
    target_bg_color = [0.4, 0.4, 0.4]
    target_metallic = 0.75

    target_image: torch.Tensor = SoftRas().apply(  # type: ignore
        camera,
        model,
        Transform(
            rotation=rotate_to_quaternion(Vector3(0.0, 1.0, 1.0), 0.0)
            * rotate_to_quaternion(Vector3(1.0, 0.0, 0.0), 0.0),
            position=Vector3(0.0, 0.0, 0.0),
            scaling=Vector3(1.5, 1.5, 1.5),
        ),
        PointLight(
            position=Vector3(1.0, 2.0, 3.0),
            color=Vector3(1.0, 1.0, 1.0),
            attenuation=Vector3(1.0, 0.0, 0.0),
        ),
        CookTorrance(
            roughness=0.6,
            metallic=target_metallic,
        ),
        wrap_float_tensor(target_ambient_texture),
        # wrap_float_tensor(grid_texture),
        wrap_float_tensor(target_bg_color),
        wrap_float_tensor(target_metallic),
        {
            "sigma": 1e-6,
            "epsilon": 1e-3,
            "gamma": 1e-4,
            "distance_epsilon": 1e-5,
            "ambient_light": [0.1, 0.1, 0.1],
            "bg_color": [0.4, 0.4, 0.4],
            "gamma_correction": True,
        },
    )

    # save target image
    plt.imsave("target.png", target_image.detach().cpu().numpy(), origin="lower")

    # Setup our training loop.
    lr = 1e-2
    n_itr = 100

    # Initialize our parameters.
    bg_color = wrap_float_tensor([0.1, 0.2, 0.3], True)    
    metallic = wrap_float_tensor(0.1, True)

    # Setup our optimizer.
    optimizer = torch.optim.Adam([bg_color, metallic], lr=lr)

    renderer = SoftRas()

    if not os.path.exists('temp_frames'):
        os.makedirs('temp_frames')

    def update_frame(i):
        training_loop(i, optimizer, target_image, renderer, 
                     plot_axes, bg_color, metallic)
        
        frame_path = f'temp_frames/frame_{i:04d}.png'
        fig.savefig(frame_path, bbox_inches='tight', pad_inches=0)
        
        img = Image.open(frame_path)
        bbox = img.getbbox()
        cropped = img.crop(bbox)
        cropped.save(frame_path)

    for i in range(n_itr):
        update_frame(i)

    frames = []
    for i in range(n_itr):
        frame = Image.open(f'temp_frames/frame_{i:04d}.png')
        
        if i == 0:
            base_size = frame.size
        frame = frame.resize(base_size)
        
        frames.append(frame)

    frames[0].save('soft_ras.gif',
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,  # 100ms = 10fps
                   loop=0,
                   disposal=2)

    # ani = animation.FuncAnimation(fig, partial(  # type: ignore
    #         training_loop,
    #         # itr params
    #         optimizer=optimizer,
    #         target_image=target_image,
    #         renderer=renderer,
    #         plot_axes=plot_axes,
    #         # shader params
    #         bg_color=bg_color,
    #         metallic=metallic
    #     ), frames=n_itr, interval=10
    # )

    # writer = animation.FFMpegWriter(fps=30)
    # ani.save('soft_ras.mp4', writer=writer)


if __name__ == "__main__":
    main()
