

struct Camera
{
    float4x4 view;
    float4x4 proj;

    float2 screen_resolution;

    [Differentiable]
    float2 worldToScreen(float3 world)
    {
        const float4 world_pos = float4(world, 1.0f);
        const float4 view_pos = mul(view, world_pos);
        const float4 proj_pos = mul(proj, view_pos);
        const float3 ndc_pos = proj_pos.xyz / proj_pos.w;
        const float2 screen_pos = 0.5f * (ndc_pos.xy + 1.0f) * screen_resolution;
        return screen_pos;
    }
};
