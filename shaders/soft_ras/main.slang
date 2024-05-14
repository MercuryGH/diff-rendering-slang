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

    float distance_epsilon;

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
    float softmax_sum = bg_weight;
    float softmax_max = params.epsilon;

    float3 color = bg_weight * bg_color;

    // if (global_idx.x == 0 && global_idx.y == 0)
    // {
    //     printf("inital color: %f %f %f\n", color.x, color.y, color.z);
    // }

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

            float d = distanceToTriangle(pixel_coord, A_.xy, B_.xy, C_.xy);

            if (d > 0 && d * d > params.distance_epsilon * params.sigma)
                continue; // skip faces far away from the pixel to avoid NaN

            const float hit = sigmoid(-_sign(d) * d * d / params.sigma);

            /// do barycentric interpolation for inverse z under perspective camera
            const float hit_inv_z = barycentric.x * A.z + barycentric.y * B.z + barycentric.z * C.z;

            if (hit_inv_z < -1.0 || hit_inv_z > 1.0)
            {
                continue; // hit point is outside the view frustum
            }

            float exp_delta_inv_z = 1.0;
            if (hit_inv_z > softmax_max)
            {
                exp_delta_inv_z = exp((hit_inv_z - softmax_max) / params.gamma);
                softmax_max = hit_inv_z;
            }
            const float exp_z = exp(hit_inv_z / gamma);
            softmax_sum = exp_delta_inv_z * softmax_sum + exp_z * hit;
            color = exp_delta_inv_z * color + exp_z * hit * fg_color;

            // if (global_idx.x == 0 && global_idx.y == 0)
            // {
            //     printf("hit: %f\n", hit);
            //     printf("hit_inv_z: %f\n", hit_inv_z);
            //     printf("softmax_sum: %f\n", softmax_sum);
            // }
        }
    }

    color /= softmax_sum;

    // if (global_idx.x == 0 && global_idx.y == 0)
    // {
    //     printf("bg_weight: %f\n", bg_weight);
    //     printf("softmax_sum: %f\n", softmax_sum);
    //     printf("color: %f %f %f\n", color.x, color.y, color.z);
    // }

    output.storeOnce(uint3(global_idx.xy, 0), color.x);
    output.storeOnce(uint3(global_idx.xy, 1), color.y);
    output.storeOnce(uint3(global_idx.xy, 2), color.z);
}

