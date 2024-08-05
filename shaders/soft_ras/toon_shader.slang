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

struct ToonRenderParam
{
    PerspectiveCamera camera;
    Mesh mesh;
    Transform transform;
    PointLight light;
    CookTorranceMaterial material;
    Texture2D texture0;
    RenderParams params;
};

[Differentiable]
float3 ToonBRDF(float3 L, float3 V, float3 N, float3 baseColor, float metallic)
{
    float3 H = normalize(L + V);
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    float NdotH = dot(N, H);
    float VdotH = dot(V, H);

    float diffuse = max(0.0, NdotL);
    diffuse = smoothstep(0.4, 0.6, diffuse);

    float specular = pow(max(0.0, NdotH), 20.0);
    specular = smoothstep(0.4, 0.6, specular);

    float3 color = baseColor * diffuse + float3(1.0) * specular * metallic;

    return color;
}

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void Main(
    ToonRenderParam renderParam,
    DiffTensorView<float> output)
{
    uint3 global_idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    uint2 image_size = uint2(output.size(1), output.size(0));

    if (global_idx.x >= image_size.x || global_idx.y >= image_size.y)
        return;

    uint i = global_idx.y * image_size.x + global_idx.x;
    float2 pixel_coord = global_idx.xy + 0.5;

    const float3 bg_color = renderParam.params.bg_color;
    const float epsilon = renderParam.params.epsilon;
    const float gamma = renderParam.params.gamma;

    let vp = VP(renderParam.camera.getViewMatrix(), renderParam.camera.getProjMatrix(), float2(image_size));
    let model = renderParam.transform.getMatrix();

    float bg_weight = exp(0.0 / gamma);
    if_pixel(0, 0)
    {
        pdebug(bg_weight);
        pdebug4x4(model);
        pdebug4x4(vp.view);
        pdebug4x4(vp.proj);
    }

    float softmax_sum = bg_weight;
    float softmax_max = renderParam.params.epsilon;

    float3 color = bg_weight * bg_color;

    let num_batch = renderParam.mesh.faces.size(0);
    let num_face = renderParam.mesh.faces.size(1);

    [MaxIters(3)]
    for (uint bs = 0; bs < num_batch; bs++) // for each batch
    {
        [MaxIters(600)]
        for (uint i_face = 0; i_face < num_face; i_face++) // for each face
        {
            int3 vertex_indices = reinterpret<int3>(loadFloat3FromTensor(renderParam.mesh.faces, int2(bs, i_face)));

            float3 A_local = loadFloat3FromTensor(renderParam.mesh.vertices, int2(bs, vertex_indices.x));
            float3 B_local = loadFloat3FromTensor(renderParam.mesh.vertices, int2(bs, vertex_indices.y));
            float3 C_local = loadFloat3FromTensor(renderParam.mesh.vertices, int2(bs, vertex_indices.z));

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

            let hit_z_view = -perspective_correct_interpolate(A_view.z, B_view.z, C_view.z, barycentric, float3(A_screen.z, B_screen.z, C_screen.z));

            if (hit_z_view < renderParam.camera.near || hit_z_view > renderParam.camera.far)
            {
                continue;
            }

            const float hit_z_norm = 1.0 - (hit_z_view - renderParam.camera.near) * renderParam.camera.far / (renderParam.camera.far - renderParam.camera.near) / hit_z_view;

            float d = distanceToTriangle(pixel_coord, A_screen.xy, B_screen.xy, C_screen.xy);

            if (d > 0 && d * d > renderParam.params.distance_epsilon * renderParam.params.sigma)
                continue;

            const float hit = sigmoid(-_sign(d) * d * d / renderParam.params.sigma);

            float exp_delta_inv_z = 1.0;
            if (hit_z_norm > softmax_max)
            {
                exp_delta_inv_z = exp((hit_z_norm - softmax_max) / renderParam.params.gamma);
                softmax_max = hit_z_norm;
            }
            const float exp_z = exp((hit_z_norm - softmax_max) / gamma);
            softmax_sum = softmax_sum / exp_delta_inv_z + exp_z * hit;

            let uv_index = reinterpret<int3>(loadFloat3FromTensor(renderParam.mesh.uv_indices, int2(bs, i_face)));

            let uv_A = loadFloat2FromTensor(renderParam.mesh.tex_coords, int2(bs, uv_index.x));
            let uv_B = loadFloat2FromTensor(renderParam.mesh.tex_coords, int2(bs, uv_index.y));
            let uv_C = loadFloat2FromTensor(renderParam.mesh.tex_coords, int2(bs, uv_index.z));

            let u = perspective_correct_interpolate(uv_A.x, uv_B.x, uv_C.x, barycentric, float3(A_screen.z, B_screen.z, C_screen.z));
            let v = perspective_correct_interpolate(uv_A.y, uv_B.y, uv_C.y, barycentric, float3(A_screen.z, B_screen.z, C_screen.z));

            let uv = float2(u, v);

            let albedo : float3 = renderParam.texture0.sample(uv);

            if_pixel(256, 256) pdebug3(albedo);

            let normal_index = reinterpret<int3>(loadFloat3FromTensor(renderParam.mesh.normal_indices, int2(bs, i_face)));

            let normal_A = loadFloat3FromTensor(renderParam.mesh.normals, int2(bs, normal_index.x));
            let normal_B = loadFloat3FromTensor(renderParam.mesh.normals, int2(bs, normal_index.y));
            let normal_C = loadFloat3FromTensor(renderParam.mesh.normals, int2(bs, normal_index.z));

            let normal = normalize(perspective_correct_interpolate(normal_A, normal_B, normal_C, barycentric, float3(A_screen.z, B_screen.z, C_screen.z)));

            let normal_matrix = transpose(inverse(model));
            let normal_world = normalize(mul(normal_matrix, float4(normal, 0)).xyz);

            let viewDir = normalize(renderParam.camera.eye - A_world);
            let lightDir = normalize(renderParam.light.position - A_world);
            let lightColor = renderParam.light.color;
            let colorToon = ToonBRDF(lightDir, viewDir, normal_world, albedo, renderParam.material.metallic);

            let ambient = renderParam.params.ambient_light * albedo;
            let lighted_color = ambient + colorToon;

            color = color / exp_delta_inv_z + exp_z * hit * lighted_color;
        }
    }

    color /= softmax_sum;

    if (renderParam.params.gamma_correction) {
        color = color / (color + 1.0);
        color = pow(color, 1.0 / 2.2);
    }

    output.storeOnce(uint3(global_idx.yx, 0), color.x);
    output.storeOnce(uint3(global_idx.yx, 1), color.y);
    output.storeOnce(uint3(global_idx.yx, 2), color.z);
}
