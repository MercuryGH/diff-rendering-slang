// Call stack:
// main::SoftRasterizer()
// SoftRasterizer::forward()
// srf::soft_rasterize()
// SoftRasFunction::apply()
// SoftRasFunction::forward() & backward()
// (CUDA) forward_soft_rasterize_cuda() and backward_soft_rasterize_cuda()

// #define DEBUG
#include <debug.slangh>

import camera;
import math;
import transform;

struct RenderParams
{
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
    Transform transform,
    DiffTensorView<float> output,
    RenderParams params) // control soft rasterization
{
    uint3 global_idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint2 image_size = uint2(output.size(1), output.size(0));

    if (global_idx.x >= image_size.x || global_idx.y >= image_size.y)
        return;

    uint i = global_idx.y * image_size.x + global_idx.x;

    float2 pixel_coord = global_idx.xy + 0.5;

    const float3 fg_color = params.fg_color; // TODO: replace it with proper texture mapping
    const float3 bg_color = params.bg_color;

    const float epsilon = params.epsilon;
    const float gamma = params.gamma;

    let vp = VP(camera.getViewMatrix(), camera.getProjMatrix(), float2(image_size));
    let model = transform.getMatrix();

    float bg_weight = exp(epsilon / gamma);
    if_pixel(0, 0)
    {
        pdebug(bg_weight);
        pdebug4x4(model);
        pdebug4x4(vp.view);
        pdebug4x4(vp.proj);
    }

    float softmax_sum = bg_weight;
    float softmax_max = params.epsilon;

    float3 color = bg_weight * bg_color;

    [MaxIters(3)]
    for (uint bs = 0; bs < face_vertices.size(0); bs++) // for each batch
    {
        [MaxIters(600)]                                                 // TODO: remove this by using separate kernel
        for (uint i_face = 0; i_face < face_vertices.size(1); i_face++) // for each face
        {
            float3 A_local = loadFloat3FromTensor(face_vertices, int3(bs, i_face, 0));
            float3 B_local = loadFloat3FromTensor(face_vertices, int3(bs, i_face, 1));
            float3 C_local = loadFloat3FromTensor(face_vertices, int3(bs, i_face, 2));

            float3 A_world = mul(model, float4(A_local, 1)).xyz;
            float3 B_world = mul(model, float4(B_local, 1)).xyz;
            float3 C_world = mul(model, float4(C_local, 1)).xyz;

            float3 A_view = vp.worldToView(A_world);
            float3 B_view = vp.worldToView(B_world);
            float3 C_view = vp.worldToView(C_world);

            float3 A_screen = vp.viewToScreen(A_view);
            float3 B_screen = vp.viewToScreen(B_view);
            float3 C_screen = vp.viewToScreen(C_view);

            float3 barycentric = barycentric(float3(A_screen.xy, 0), float3(B_screen.xy, 0), float3(C_screen.xy, 0), float3(pixel_coord, 0));

            if_pixel(256, 256)
            {
                pdebug3(A);
                pdebug3(B);
                pdebug3(C);

                pdebug3(A_view);
                pdebug3(B_view);
                pdebug3(C_view);

                pdebug3(A_screen);
                pdebug3(B_screen);
                pdebug3(C_screen);

                pdebug3(barycentric);

                pdebug2(pixel_coord);
                pdebug2(vp.screen_resolution);
            }

            /// do barycentric interpolation for inverse z under perspective camera
            const float hit_z_view = -1.0 / (barycentric.x * (1.0 / A_view.z) + barycentric.y * (1.0 / B_view.z) + barycentric.z * (1.0 / C_view.z));

            if_pixel(256, 256)
            {
                pdebug(hit_z_view);
            }

            if (hit_z_view < camera.near || hit_z_view > camera.far)
            {
                continue; // hit point is outside the view frustum
            }

            // non-linear interpolation & reverse z
            const float hit_z_norm = 1.0 - (hit_z_view - camera.near) * camera.far / (camera.far - camera.near) / hit_z_view;

            float d = distanceToTriangle(pixel_coord, A_screen.xy, B_screen.xy, C_screen.xy);

            if (d > 0 && d * d > params.distance_epsilon * params.sigma)
                continue; // skip faces far away from the pixel to avoid NaN

            const float hit = sigmoid(-_sign(d) * d * d / params.sigma);

            float exp_delta_inv_z = 1.0;
            if (hit_z_norm > softmax_max)
            {
                exp_delta_inv_z = exp((hit_z_norm - softmax_max) / params.gamma);
                softmax_max = hit_z_norm;
            }
            const float exp_z = exp((hit_z_norm - softmax_max) / gamma);
            softmax_sum = softmax_sum / exp_delta_inv_z + exp_z * hit;
            color = color / exp_delta_inv_z + exp_z * hit * fg_color;

            if_pixel(256, 256)
            {
                pdebug(hit_z_norm);
                pdebug(d);
                pdebug(hit);
                pdebug(hit_z_norm);
                pdebug(softmax_max);
                pdebug(exp_delta_inv_z);
                pdebug(exp_z);
                pdebug(softmax_sum);
                pdebug3(color);
            }
        }
    }

    color /= softmax_sum;

    output.storeOnce(uint3(global_idx.yx, 0), color.x);
    output.storeOnce(uint3(global_idx.yx, 1), color.y);
    output.storeOnce(uint3(global_idx.yx, 2), color.z);
}

