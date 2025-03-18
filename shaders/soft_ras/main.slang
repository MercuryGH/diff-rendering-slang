// Call stack:
// main::SoftRasterizer()
// SoftRasterizer::forward()
// srf::soft_rasterize()
// SoftRasFunction::apply()
// SoftRasFunction::forward() & backward()
// (CUDA) forward_soft_rasterize_cuda() and backward_soft_rasterize_cuda()

#define DEBUG
#include <debug.slangh>

import camera;
import math;
import transform;
import mesh;
import texture;
import lighting;
import bxdf;

import memory;

struct RenderParams
{
    float sigma;
    float epsilon;
    float gamma;

    float distance_epsilon;

    float3 ambient_light;
    float3 bg_color;
    bool gamma_correction;
};

struct CookTorranceMaterial
{
    float roughness;
    float metallic;
};

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void Main(
    PerspectiveCamera camera,
    Mesh mesh,
    Transform transform,
    PointLight light,
    CookTorranceMaterial material,
    Texture2D texture0,
    DiffTensorView<float> output,
    DiffTensorView<float> bg_color,
    DiffTensorView<float> metallic,
    RenderParams params) // control soft rasterization
{
    uint3 global_idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint2 image_size = uint2(output.size(1), output.size(0));

    if (global_idx.x >= image_size.x || global_idx.y >= image_size.y)
        return;

    uint i = global_idx.y * image_size.x + global_idx.x;

    float2 pixel_coord = global_idx.xy + 0.5;

    // optimizable params must declare as DiffTensorView<float>
    let bg_color = float3(bg_color[0], bg_color[1], bg_color[2]);
    let metallic = float(metallic[0]);

    const float epsilon = params.epsilon;
    const float gamma = params.gamma;

    let vp = VP(camera.getViewMatrix(), camera.getProjMatrix(), float2(image_size));
    let model = transform.getMatrix();

    float bg_weight = exp(0.0 / gamma);
    float softmax_sum = bg_weight;
    float softmax_max = params.epsilon;

    float3 color = bg_weight * bg_color;

    let num_batch = mesh.faces.size(0);
    let num_face = mesh.faces.size(1);

    [MaxIters(116)]
    // [MaxIters(600)] if set MaxIters to a high value, may cause RuntimeError: CUDA error: invalid argument
    for (uint i_face = 0; i_face < num_face; i_face++) 
    {
        // uint i_face = 0;
        uint bs = 0;

        float3 vi = loadFloat3FromTensor(mesh.faces, int2(bs, i_face));
        int3 vertex_indices = asuint(vi);

        // if (i == 0) {
        // }

        // uint3 vertex_indices = loadUint3FromTensor(mesh.faces, int2(bs, i_face));
        // int3 vertex_indices = int3(1, 2, 3);

        // if (global_idx.x == 0) {
        //     pdebug3i(vertex_indices);
        // }
        // int3 vertex_indices = i_face == 0 ? int3(1, 2, 3) :
        //                     i_face == 1 ? int3(5, 3, 7) :
        //                     i_face == 2 ? int3(5, 4, 0) :
        //                     i_face == 3 ? int3(7, 4, 0) :
        //                     i_face == 4 ? int3(7, 6, 0) :
        //                     int3(1, 2, 3);

        // fetch triangle vertex positions
        float3 A_local = loadFloat3FromTensor(mesh.vertices, int2(bs, vertex_indices.x));
        float3 B_local = loadFloat3FromTensor(mesh.vertices, int2(bs, vertex_indices.y));
        float3 C_local = loadFloat3FromTensor(mesh.vertices, int2(bs, vertex_indices.z));

        // MVP
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

        /// do barycentric interpolation for inverse z under perspective camera
        let hit_z_view = -perspective_correct_interpolate(A_view.z, B_view.z, C_view.z, barycentric, float3(A_screen.z, B_screen.z, C_screen.z));
        const float hit_z_norm = 1.0 - (hit_z_view - camera.near) * camera.far / (camera.far - camera.near) / hit_z_view;
        float d = distanceToTriangle(pixel_coord, A_screen.xy, B_screen.xy, C_screen.xy);

        if (d < 0 || d * d <= params.distance_epsilon * params.sigma) {
            // float3 _base_color = float3(0.8, 0.7, 0.6);
            float3 _base_color = float3(1.0, 1.0, 1.0);

            const float hit = sigmoid(-sign(d) * d * d / params.sigma);
            // const float hit = 1.0;

            float exp_delta_zp = 1.0;
            if (hit_z_norm > softmax_max)
            {
                exp_delta_zp = exp((softmax_max - hit_z_norm) / params.gamma);
                softmax_max = hit_z_norm;
            }

            const float exp_z = exp((hit_z_norm - softmax_max) / params.gamma);

            softmax_sum = softmax_sum * exp_delta_zp + exp_z * hit;
            // color = _base_color / exp_delta_inv_z + exp_z * hit * params.ambient_light;

                /// texture mapping
                let uv_index = reinterpret<int3>(loadFloat3FromTensor(mesh.uv_indices, int2(bs, i_face)));

                // OK continue;

                let uv_A = loadFloat2FromTensor(mesh.tex_coords, int2(bs, uv_index.x));
                let uv_B = loadFloat2FromTensor(mesh.tex_coords, int2(bs, uv_index.y));
                let uv_C = loadFloat2FromTensor(mesh.tex_coords, int2(bs, uv_index.z));

                let u = perspective_correct_interpolate(uv_A.x, uv_B.x, uv_C.x, barycentric, float3(A_screen.z, B_screen.z, C_screen.z));
                let v = perspective_correct_interpolate(uv_A.y, uv_B.y, uv_C.y, barycentric, float3(A_screen.z, B_screen.z, C_screen.z));
                let uv = float2(u, v);

                let albedo : float3 = texture0.sample(uv);

                /// normal interpolation
                let normal_index = reinterpret<int3>(loadFloat3FromTensor(mesh.normal_indices, int2(bs, i_face)));

                let normal_A = loadFloat3FromTensor(mesh.normals, int2(bs, normal_index.x));
                let normal_B = loadFloat3FromTensor(mesh.normals, int2(bs, normal_index.y));
                let normal_C = loadFloat3FromTensor(mesh.normals, int2(bs, normal_index.z));
                let normal = normalize(perspective_correct_interpolate(normal_A, normal_B, normal_C, barycentric, float3(A_screen.z, B_screen.z, C_screen.z)));

                /// Lighting

                let normal_matrix = transpose(inverse(model));
                let normal_world = normalize(mul(normal_matrix, float4(normal, 0)).xyz);

                let Lo = light.illuminate(
                    CookTorrance(albedo, metallic, material.roughness),
                    SurfaceGeometry(A_world, normal_world, float2(0)),
                    normalize(light.position - A_world));

                let ambient = params.ambient_light * albedo;
                let lighted_color = ambient + Lo * 10;

                // TODO: use the constant base_color?
                // color = _base_color * exp_delta_zp + exp_z * hit * lighted_color;
                color = color * exp_delta_zp + exp_z * hit * lighted_color;
        }
    }

    color /= softmax_sum;

    /// Gamma correction
    if (params.gamma_correction)
    {
        color = color / (color + 1.0);
        color = pow(color, 1.0 / 2.2);
    }

    output.storeOnce(uint3(global_idx.yx, 0), color.x);
    output.storeOnce(uint3(global_idx.yx, 1), color.y);
    output.storeOnce(uint3(global_idx.yx, 2), color.z);
}

