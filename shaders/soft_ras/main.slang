// Call stack:
// main::SoftRasterizer()
// SoftRasterizer::forward()
// srf::soft_rasterize()
// SoftRasFunction::apply()
// SoftRasFunction::forward() & backward()
// (CUDA) forward_soft_rasterize_cuda() and backward_soft_rasterize_cuda()

struct RenderParams {
    uint width;
    uint height;
    float sigma;
};

[Differentiable]
float getInputElement(DiffTensorView<float> input,
                       int3 loc)
{
    loc = min(loc, int3(input.size(0) - 1, input.size(1) - 1, input.size(2) - 1));
    loc = max(loc, int3(0, 0, 0));
    return input.load(loc);
}

[Differentiable]
float3 computeBarycentric2D(float3 v0, float3 v1, float3 v2, float2 p) {
    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 p0 = float3(p.x - v0.x, p.y - v0.y, 0.0f);
    float area = cross(v0v1, v0v2).z;
    float alpha = cross(v1 - v2, p0).z / area;
    float beta = cross(v2 - v0, float3(p.x - v1.x, p.y - v1.y, 0.0f)).z / area;
    float gamma = 1.0f - alpha - beta;
    return float3(alpha, beta, gamma);
}

[Differentiable]
float calculateDepth(float3 barycentric, float z0, float z1, float z2) {
    return barycentric.x * z0 + barycentric.y * z1 + barycentric.z * z2;
}

[Differentiable]
bool isInsideTriangle(float3 barycentric) {
    return barycentric.x >= 0.0f && barycentric.y >= 0.0f && barycentric.z >= 0.0f;
}

[Differentiable]
float computeSoftEdgeWeight(float3 barycentric, float sigma) {
    float weight = exp(-barycentric.x * barycentric.x / (2.0f * sigma * sigma));
    weight += exp(-barycentric.y * barycentric.y / (2.0f * sigma * sigma));
    weight += exp(-barycentric.z * barycentric.z / (2.0f * sigma * sigma));
    return weight;
}


[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void forward_stub(
    DiffTensorView<float> face_vertices,
    DiffTensorView<float> output,
    RenderParams params) //control soft rasterization
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (globalIdx.x >= params.width || globalIdx.y >= params.height)
        return;

    // position in normalized device coordinates
    float pixelCenterX = (globalIdx.x + 0.5f) / params.width * 2.0f - 1.0f;
    float pixelCenterY = (globalIdx.y + 0.5f) / params.height * 2.0f - 1.0f;
    float4 color = { 0.0f, 0.0f, 0.0f, 0.0f }; // RGBA
    float depth = 1e6;
    int numVertices = face_vertices.size(1) / 3;

    //uint i = globalIdx.y * image_size + globalIdx.x;

    // float3 result = float3(face_vertices[3 * i], face_vertices[3 * i + 1], face_vertices[3 * i + 2]);
    [MaxIters(100)]
    for (int i = 0; i < numVertices / 3; i++) {
        float3 v0 = getInputElement(face_vertices, int3(0, 3 * i, 0));
        float3 v1 = getInputElement(face_vertices, int3(0, 3 * i + 1, 0));
        float3 v2 = getInputElement(face_vertices, int3(0, 3 * i + 2, 0));

        float3 barycentric = computeBarycentric2D(v0, v1, v2, float2(pixelCenterX, pixelCenterY));

        float pixelDepth = calculateDepth(barycentric, v0.z, v1.z, v2.z);

        if (isInsideTriangle(barycentric) && pixelDepth < depth) {
            float weight = computeSoftEdgeWeight(barycentric, params.sigma);
            color = float4(weight * v0.x, weight * v1.y, weight * v2.z, 1.0f); // Example of color calculation
            depth = pixelDepth;
        }
    }
    // Storing the result once per thread
    output.storeOnce(uint3(globalIdx.x, globalIdx.y, 0), color.x);
    output.storeOnce(uint3(globalIdx.x, globalIdx.y, 1), color.y);
    output.storeOnce(uint3(globalIdx.x, globalIdx.y, 2), color.z);
    output.storeOnce(uint3(globalIdx.x, globalIdx.y, 3), color.w);
    //calculate barycentric coordinates and depth
     //printf("%d, %d, %d, %d, %d\n", face_vertices.size(0), face_vertices.size(1), face_vertices.size(2), face_vertices.size(3), face_vertices.size(4)); // 1, 5856, 3, 3, 0
}

