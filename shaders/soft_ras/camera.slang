[Differentiable]
float4x4 lookAt(float3 eye, float3 center, float3 up)
{
    const float3 f = normalize(center - eye);
    const float3 s = normalize(cross(f, up));
    const float3 u = cross(s, f);

    return float4x4(
        float4(s.x, s.y, s.z, -dot(s, eye)),
        float4(u.x, u.y, u.z, -dot(u, eye)),
        float4(-f.x, -f.y, -f.z, dot(f, eye)),
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
        float4(0.0f, 0.0f, (far + near) * nf, 2.0f * far * near * nf),
        float4(0.0f, 0.0f, -1.0f, 0.0f));
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

    [Differentiable]
    float3 worldToView(float3 world)
    {
        return mul(view, float4(world, 1.0f)).xyz;
    }

    // convert view space position to screen space position
    // return: (screen x in pixel, screen y in pixel, ndc z)
    [Differentiable]
    float3 viewToScreen(float3 view)
    {
        const float4 view_pos = float4(view, 1.0f);
        const float4 clip_pos = mul(proj, view_pos);
        const float3 ndc_pos = clip_pos.xyz / clip_pos.w;
        const float2 screen_pos = 0.5f * (ndc_pos.xy + 1.0f) * screen_resolution;
        return float3(screen_pos, ndc_pos.z);
    }
};

/// Test for camera

#define DEBUG
#include <debug.slangh>

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void TestPerspectiveCamera(PerspectiveCamera camera)
{
    // let cam = PerspectiveCamera(float3(0, 0, 3), float3(0, 0, -1), float3(0, 1, 0), 60 / 180 * 3.14159265358, 1.0, 0.1, 100);
    let p_n = float4(1, 0, -0.1, 1);
    let p_f = float4(1, 0, -100, 1);
    // let view = cam.getViewMatrix();
    let fovy = camera.fov;
    let proj = perspective(fovy, 1.0, 0.1, 100);

    pdebug(fovy);
    pdebug(tan(fovy * 0.5f));
    pdebug(1.0f / tan(fovy * 0.5f));

    pdebug4x4(proj);

    float4 p_pn = mul(proj, p_n);
    float4 p_pf = mul(proj, p_f);

    p_pn /= p_pn.w;
    p_pf /= p_pf.w;

    pdebug4(p_pn);
    pdebug4(p_pf);

    let eye = camera.eye;
    let center = camera.dir + eye;
    let up = camera.up;

    let view = lookAt(eye, center, up);

    pdebug4x4(view);

    let A = float3(0, 0, 0);
    let B = float3(1, 0, 3);
    let C = float3(2, 1, -3);

    let A_view = mul(view, float4(A, 1));
    let B_view = mul(view, float4(B, 1));
    let C_view = mul(view, float4(C, 1));

    pdebug4(A_view);
    pdebug4(B_view);
    pdebug4(C_view);
}
