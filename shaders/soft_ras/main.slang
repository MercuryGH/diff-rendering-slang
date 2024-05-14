// Call stack:
// main::SoftRasterizer()
// SoftRasterizer::forward()
// srf::soft_rasterize()
// SoftRasFunction::apply()
// SoftRasFunction::forward() & backward()
// (CUDA) forward_soft_rasterize_cuda() and backward_soft_rasterize_cuda()

import camera;
import math;

struct RenderParams
{
    uint width;
    uint height;
    float sigma;
    float epsilon;
    float gamma;

    float3 fg_color;
    float3 bg_color;
};

[Differentiable]
float getInputElement(
    DiffTensorView<float> input,
    int3 loc // (bs, face, vertex)
)
{
    loc = min(loc, int3(input.size(0) - 1, input.size(1) - 1, input.size(2) - 1));
    loc = max(loc, int3(0, 0, 0));
    return input.load(loc);
}

[Differentiable]
float3 loadFloat3FromTensor(
    DiffTensorView<float> input,
    int3 loc // (bs, face, vertex)
)
{
    loc = min(loc, int3(input.size(0) - 1, input.size(1) - 1, input.size(2) - 1));
    loc = max(loc, int3(0, 0, 0));
    return float3(input[loc.x, loc.y, loc.z, 0], input[loc.x, loc.y, loc.z, 1], input[loc.x, loc.y, loc.z, 2]);
}

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void main(
    PerspectiveCamera camera,
    DiffTensorView<float> face_vertices,
    DiffTensorView<float> output,
    RenderParams params) // control soft rasterization
{
    uint3 global_idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint2 image_size = uint2(output.size(0), output.size(1));

    if (global_idx.x >= image_size.x || global_idx.y >= image_size.y)
        return;

    uint i = global_idx.y * image_size.x + global_idx.x;

    float2 pixel_coord = global_idx.xy + 0.5;

    const float3 fg_color = params.fg_color; // TODO: replace it with proper texture mapping
    const float3 bg_color = params.bg_color;

    const float epsilon = params.epsilon;
    const float gamma = params.gamma;

    let mvp = MVP(camera.getViewMatrix(), camera.getProjMatrix(), float2(image_size));

    float bg_weight = exp(epsilon / gamma);
    float total_weight = bg_weight;

    float3 color = bg_weight * bg_color;

    [MaxIters(3)]
    for (uint bs = 0; bs < face_vertices.size(0); bs++) // for each batch
    {
        [MaxIters(600)]                                                 // TODO: remove this by using separate kernel
        for (uint i_face = 0; i_face < face_vertices.size(1); i_face++) // for each face
        {
            float3 A = loadFloat3FromTensor(face_vertices, int3(bs, i_face, 0));
            float3 B = loadFloat3FromTensor(face_vertices, int3(bs, i_face, 1));
            float3 C = loadFloat3FromTensor(face_vertices, int3(bs, i_face, 2));

            /// project vertices to screen space
            float3 A_ = mvp.worldToScreen(A);
            float3 B_ = mvp.worldToScreen(B);
            float3 C_ = mvp.worldToScreen(C);

            float3 barycentric = computeBarycentric2D(A_, B_, C_, pixel_coord);

            float hit = soft_triangle(pixel_coord, A_.xy, B_.xy, C_.xy, params.sigma);

            /// do barycentric interpolation for inverse z under perspective camera
            float hit_inv_z = barycentric.x * A.z + barycentric.y * B.z + barycentric.z * C.z;

            color += hit * exp(hit_inv_z / gamma) * fg_color;
            total_weight += hit * exp(hit_inv_z / gamma);
        }
    }

    color /= total_weight;

    output.storeOnce(uint3(global_idx.xy, 0), color.x);
    output.storeOnce(uint3(global_idx.xy, 1), color.y);
    output.storeOnce(uint3(global_idx.xy, 2), color.z);
}

