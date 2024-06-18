import memory;

struct Texture2D
{
    DiffTensorView<float> image;

    [Differentiable]
    float3 sample(float2 uv)
    {
        // bilinear interpolation

        uv = clamp(uv, 0.0, 1.0);

        let image_coord = uv * float2(image.size(1), image.size(0));

        int2 uv0 = int2(no_diff(floor(image_coord)));
        int2 uv1 = (uv0 + 1) % int2(image.size(1), image.size(0));

        float2 f = image_coord - uv0;

        float3 i00 = loadFloat3FromTensor(image, uv0.yx);
        float3 i01 = loadFloat3FromTensor(image, int2(uv0.y, uv1.x));
        float3 i10 = loadFloat3FromTensor(image, int2(uv1.y, uv0.x));
        float3 i11 = loadFloat3FromTensor(image, uv1.yx);

        return i00 * (1 - f.x) * (1 - f.y) +
               i01 * (1 - f.x) * f.y +
               i10 * f.x * (1 - f.y) +
               i11 * f.x * f.y;
    }
};

/// Test for texture

#define DEBUG
#include <debug.slangh>

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void TestTexture(Texture2D texture, DiffTensorView<float> output)
{
    // sample texture to output, equivalent to guassian blur

    let image_size = uint2(output.size(1), output.size(0));

    // pdebug2i(image_size);

    let global_idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    // pdebug2i(global_idx.xy);

    if (global_idx.x >= image_size.x || global_idx.y >= image_size.y)
        return;

    let pixel_coord = global_idx.xy + 0.5;

    let uv = pixel_coord / float2(image_size);

    if (any(uv < 0.0) || any(uv > 1.0))
        printf("BUG: uv = %f %f\n", uv.x, uv.y);

    let color = texture.sample(uv);

    output[global_idx.y, global_idx.x, 0] = color.x;
    output[global_idx.y, global_idx.x, 1] = color.y;
    output[global_idx.y, global_idx.x, 2] = color.z;

}