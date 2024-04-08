// Call stack:
// main::SoftRasterizer()
// SoftRasterizer::forward()
// srf::soft_rasterize()
// SoftRasFunction::apply()
// SoftRasFunction::forward() & backward()
// (CUDA) forward_soft_rasterize_cuda() and backward_soft_rasterize_cuda()
[Differentiable]
float getInputElement(DiffTensorView<float> input,
                      no_diff int3 loc)
{
    loc = min(loc, int3(input.size(0) - 1, input.size(1) - 1, input.size(2) - 1));
    loc = max(loc, int3(0, 0, 0));
    return input.load(loc);
}

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void forward_stub(
    DiffTensorView<float> face_vertices,
    DiffTensorView<float> output)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= output.size(0) || globalIdx.y >= output.size(1))
        return;

    const uint image_size = 20; // temp
    uint i = globalIdx.y * image_size + globalIdx.x;

    // float3 result = float3(face_vertices[3 * i], face_vertices[3 * i + 1], face_vertices[3 * i + 2]);
    float3 result = float3(
        getInputElement(face_vertices, int3(0, 3 * i, 0)),
        getInputElement(face_vertices, int3(0, 3 * i + 1, 0)),
        getInputElement(face_vertices, int3(0, 3 * i + 2, 0)));

    // printf("%d, %d, %d, %d, %d\n", face_vertices.size(0), face_vertices.size(1), face_vertices.size(2), face_vertices.size(3), face_vertices.size(4)); // 1, 5856, 3, 3, 0

    output.storeOnce(uint3(globalIdx.xy, 0), result.x);
    output.storeOnce(uint3(globalIdx.xy, 1), result.y);
    output.storeOnce(uint3(globalIdx.xy, 2), result.z);
}

