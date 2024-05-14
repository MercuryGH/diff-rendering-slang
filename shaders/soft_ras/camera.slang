[Differentiable]
float4x4 lookAt(float3 eye, float3 center, float3 up)
{
    float3 f = normalize(center - eye);
    float3 s = normalize(cross(f, up));
    float3 u = cross(s, f);

    return float4x4(
        float4(s, -dot(s, eye)),
        float4(u, -dot(u, eye)),
        float4(-f, dot(f, eye)),
        float4(0.0f, 0.0f, 0.0f, 1.0f));
}

[Differentiable]
float4x4 perspective(float fovy, float aspect, float near, float far)
{
    float f = 1.0f / tan(fovy * 0.5f);
    float nf = 1.0f / (near - far);

    return float4x4(
        float4(f / aspect, 0.0f, 0.0f, 0.0f),
        float4(0.0f, f, 0.0f, 0.0f),
        float4(0.0f, 0.0f, (far + near) * nf, -1.0f),
        float4(0.0f, 0.0f, 2.0f * far * near * nf, 0.0f));
}

struct PerspectiveCamera
{
    float3 eye;
    float3 dir;
    float3 up;

    float fov;
    float aspect;
    float near;
    float far;

    [Differentiable]
    float4x4 getViewMatrix()
    {
        return lookAt(eye, eye + dir, up);
    }

    [Differentiable]
    float4x4 getProjMatrix()
    {
        return perspective(fov, aspect, near, far);
    }
}

struct MVP : IDifferentiable
{
    float4x4 view;
    float4x4 proj;

    float2 screen_resolution;

    [Differentiable]
    __init(float4x4 view, float4x4 proj, float2 screen_resolution)
    {
        this.view = view;
        this.proj = proj;
        this.screen_resolution = screen_resolution;
    }

    // convert world space position to screen space position
    // return: (screen x in pixel, screen y in pixel, ndc z)
    [Differentiable]
    float3 worldToScreen(float3 world)
    {
        const float4 world_pos = float4(world, 1.0f);
        const float4 view_pos = mul(view, world_pos);
        const float4 clip_pos = mul(proj, view_pos);
        const float3 ndc_pos = clip_pos.xyz / clip_pos.w;
        const float2 screen_pos = 0.5f * (ndc_pos.xy + 1.0f) * screen_resolution;
        return float3(screen_pos, ndc_pos.z);
    }
};
