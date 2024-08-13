import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from renderers.soft_ras import SoftRas
from resources.resource import spot_obj, tex_image, cube_obj, fox_gltf
from utils.util import wrap_float_tensor
from renderers.camera import PerspectiveCamera
from renderers.light import PointLight
from utils.vectors import Vector3
from renderers.transform import Transform, rotate_to_quaternion
from renderers.material import CookTorrance
from IPython.display import HTML

WIDTH = 512
HEIGHT = 384


def update_frame(frame: int, img):
    T = 32
    angle = 2 * np.pi * (frame % T) / T
    transform = Transform(
        rotation=rotate_to_quaternion(Vector3(0.0, 1.0, 1.0), angle)
        * rotate_to_quaternion(Vector3(1.0, 0.0, 0.0), angle),
        position=Vector3(0.0, 0.0, 0.0),
        scaling=Vector3(1.5, 1.5, 1.5),
    )

    camera = PerspectiveCamera(
        eye=Vector3(0.0, 0.0, 3.0),
        dir=Vector3(0.0, 0.0, -1.0),
        up=Vector3(0.0, 1.0, 0.0),
        fov=60.0 / 180.0 * np.pi,
        near=0.1,
        far=100,
        width=WIDTH,
        height=HEIGHT,
    )

    point_light = PointLight(
        position=Vector3(1.0, 2.0, 3.0),
        color=Vector3(1.0, 1.0, 1.0),
        attenuation=Vector3(1.0, 0.0, 0.0),
    )

    material = CookTorrance(
        roughness=0.6,
        metallic=0.1,
    )

    renderer = SoftRas()

    params = {
        "sigma": 1e-6,
        "epsilon": 1e-3,
        "gamma": 1e-4,
        "distance_epsilon": 1e-5,
        # "fg_color": [0.7, 0.8, 0.9],
        "ambient_light": [0.01, 0.01, 0.01],
        "bg_color": [0.4, 0.4, 0.4],
        "gamma_correction": True,
    }

    output: torch.Tensor = renderer.apply(
        camera,
        cube_obj,
        transform,
        point_light,
        material,
        wrap_float_tensor(tex_image),
        params,
    )  # type: ignore

    # print(output.shape)
    # print(output[512, 512])
    # print(output.max())
    # plt.rcParams["figure.figsize"] = (camera.width, camera.height)
    # plt.imshow(output.cpu().numpy(), origin="lower")
    # plt.imsave("output.png", output.cpu().numpy(), origin="lower")
    # plt.show()

    new_image = output.cpu().numpy()
    img.set_array(new_image)

    return (img,)


def set_grad(var):
    def hook(grad):
        var.grad = grad

    return hook


def training_loop(
    i: int,
    optimizer: torch.optim.Adam,
    target_image: torch.Tensor,
    renderer: SoftRas,
    bg_color: torch.Tensor,
    plot_axes,
):
    print("Iteration %d" % i)

    output_image: torch.Tensor = renderer.apply(  # type: ignore
        PerspectiveCamera(
            eye=Vector3(0.0, 0.0, 10.0),
            dir=Vector3(0.0, 0.0, -1.0),
            up=Vector3(0.0, 1.0, 0.0),
            fov=60.0 / 180.0 * np.pi,
            near=0.1,
            far=100,
            width=WIDTH,
            height=HEIGHT,
        ),
        cube_obj,
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
            metallic=0.1,
        ),
        wrap_float_tensor(tex_image),
        bg_color,
        {
            "sigma": 1e-6,
            "epsilon": 1e-3,
            "gamma": 1e-4,
            "distance_epsilon": 1e-5,
            "ambient_light": [0.01, 0.01, 0.01],
            "bg_color": bg_color.tolist(),
            "gamma_correction": True,
        },
    )

    # output_image.requires_grad_(True)

    output_image.register_hook(set_grad(output_image))

    # Compute the loss.
    loss = torch.mean((output_image - target_image) ** 2)

    print(f"Loss: {loss}")

    # Backward pass: compute the gradients.
    loss.backward(retain_graph=True)

    # Update the parameters.
    optimizer.step()

    plt.imsave(f"results/output{i}.png", output_image.detach().cpu().numpy(), origin="lower")

    print(f"Saved frame #{i}")

    # if True:
    #     plot_axes[0].clear()
    #     plot_axes[0].imshow(
    #         output_image.detach().cpu().numpy(), origin="lower"
    #     )
    #     plot_axes[1].clear()
    #     plot_axes[1].imshow(
    #         output_image.grad[:, :, 1].T.detach().cpu().numpy(), origin="lower"
    #     )
    #     plot_axes[2].clear()
    #     plot_axes[2].imshow(
    #         target_image.detach().cpu().numpy(), origin="lower"
    #     )

    # Zero the gradients.
    optimizer.zero_grad()

    return plot_axes


def main():
    # Setup plot
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    plot_axes = [ax1, ax2, ax3]

    target_image: torch.Tensor = SoftRas().apply(  # type: ignore
        PerspectiveCamera(
            eye=Vector3(0.0, 0.0, 10.0),
            dir=Vector3(0.0, 0.0, -1.0),
            up=Vector3(0.0, 1.0, 0.0),
            fov=60.0 / 180.0 * np.pi,
            near=0.1,
            far=100,
            width=WIDTH,
            height=HEIGHT,
        ),
        cube_obj,
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
            metallic=0.1,
        ),
        wrap_float_tensor(tex_image),
        wrap_float_tensor([0.4, 0.4, 0.4], True),
        {
            "sigma": 1e-6,
            "epsilon": 1e-3,
            "gamma": 1e-4,
            "distance_epsilon": 1e-5,
            "ambient_light": [0.01, 0.01, 0.01],
            "bg_color": [0.4, 0.4, 0.4],
            "gamma_correction": True,
        },
    )

    # save target image
    plt.imsave("target.png", target_image.detach().cpu().numpy(), origin="lower")

    # Initialize our parameters.
    bg_color = wrap_float_tensor([0.1, 0.2, 0.3], True)    

    # Setup our optimizer.
    optimizer = torch.optim.Adam([bg_color], lr=1e-2)

    ani = animation.FuncAnimation(
        fig,
        training_loop,
        fargs=(
            optimizer,
            target_image,
            SoftRas(),
            bg_color,
            plot_axes,
        ),
        frames=32,
        interval=100,
        blit=True,
    )

    # initial_image = np.zeros((HEIGHT, WIDTH, 3))
    # img = ax.imshow(initial_image, animated=True, origin="lower")
    # ani = animation.FuncAnimation(
    #     fig, update_frame, fargs=(img,), frames=32, interval=100, blit=True
    # )

    ani.save("output.gif", writer="imagemagick")  # Save as GIF


if __name__ == "__main__":
    main()
