// util functions for memory access

[Differentiable]
float getInputElement(
    DiffTensorView<float> input,
    no_diff int3 loc // (bs, face, vertex)
)
{
    loc = min(loc, int3(input.size(0) - 1, input.size(1) - 1, input.size(2) - 1));
    loc = max(loc, int3(0, 0, 0));
    return input.load(loc);
}

[Differentiable]
float3 loadFloat3FromTensor(
    DiffTensorView<float> input,
    no_diff int3 loc // (bs, face, vertex)
)
{
    loc = min(loc, int3(input.size(0) - 1, input.size(1) - 1, input.size(2) - 1));
    loc = max(loc, int3(0, 0, 0));
    return float3(input[loc.x, loc.y, loc.z, 0], input[loc.x, loc.y, loc.z, 1], input[loc.x, loc.y, loc.z, 2]);
}

[Differentiable]
float3 loadFloat3FromTensor(
    DiffTensorView<float> input,
    no_diff int2 loc // (bs, xxx)
)
{
    loc = min(loc, int2(input.size(0) - 1, input.size(1) - 1));
    loc = max(loc, int2(0, 0));
    return float3(input[loc.x, loc.y, 0], input[loc.x, loc.y, 1], input[loc.x, loc.y, 2]);
}

[Differentiable]
float3 loadFloat3FromTensor(
    DiffTensorView<float> input,
    no_diff int loc)
{
    loc = min(loc, input.size(0) - 1);
    loc = max(loc, 0);
    return float3(input[loc, 0], input[loc, 1], input[loc, 2]);
}

[Differentiable]
uint3 loadUint3FromTensor(
    TensorView<uint> input,
    no_diff int2 loc // (bs, xxx)
)
{
    loc = min(loc, int2(input.size(0) - 1, input.size(1) - 1));
    loc = max(loc, int2(0, 0));
    return uint3(input[loc.x, loc.y, 0], input[loc.x, loc.y, 1], input[loc.x, loc.y, 2]);
}

[Differentiable]
float2 loadFloat2FromTensor(
    DiffTensorView<float> input,
    no_diff int2 loc // (bs, xxx)
)
{
    loc = min(loc, int2(input.size(0) - 1, input.size(1) - 1));
    loc = max(loc, int2(0, 0));
    return float2(input[loc.x, loc.y, 0], input[loc.x, loc.y, 1]);
}

[Differentiable]
float2 loadFloat2FromTensor(
    DiffTensorView<float> input,
    no_diff int loc)
{
    loc = min(loc, input.size(0) - 1);
    loc = max(loc, 0);
    return float2(input[loc, 0], input[loc, 1]);
}
