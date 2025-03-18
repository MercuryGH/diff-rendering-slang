# Differential Rendering Experiment with Slang

This project is a set of experiments carried out using Slang, PyTorch with their auto-diff feature.

## Dependencies

- Slang（[https://github.com/shader-slang/slang](https://github.com/shader-slang/slang)）
- SlangPy（[https://github.com/shader-slang/slangpy](https://github.com/shader-slang/slangpy)）
- Slang-Torch（[https://github.com/shader-slang/slang-torch](https://github.com/shader-slang/slang-torch)）

## Contents

### Exp 1: PBR texture auto-LOD

We render a plane with a specific PBR texture combination `(diffuse, normal, roughness)` in  `./resources`, and then render it again with

* naively `/2` downsampled textures (shown in upper row),
* optimized `/2` downsampled textures (shown in lower row):

![optimized_texture](./img/optimized_texture.png)

We can see the optimized textures preserve more high-frequency details in a lower resolution. Thus the same for the rendering result below:

![optimized_rendering_result](./img/optimized_rendering_result.png)

The optimization process is done with a random lighting and a fixed camera.

### Exp 2: Auto tuning SDF shader parameters

We minimize the difference between a fixed triangle and a flexible circle.

![rasterize](./img/rasterize.gif)

The geometry is represented using 2D SDF functions. The optimized parameters are the circle's `(position, radius, color)`.

### Exp 3: Differentiable renderer and auto parameter tuning

> The forward differentiable rasterization algorithm is based on the paper [*Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning*](https://arxiv.org/abs/1904.01786).

We render a model with a differentiable renderer. It currently supports:

* glTF model import
* Texture import and sampling
* Punctual lighting and PBR shading (in `shaders/soft_ras/`)

e.g. forward rendering of a textured cube:

![cube](./img/cube.png)

This renderer is a playground for auto parameter tuning. e.g., fix the diffuse value, optimize `(bg_color, metallic)` for a spot model:

![soft_ras](./img/soft_ras.gif)

**Note:** This renderer relys on the **auto differentiation** feature of Slang. When `#face` of the model becomes large, the auto-diff process fails, manual differentiation is required.
