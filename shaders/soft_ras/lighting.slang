import math;
import bxdf;
import mesh;

interface ILight
{
    float3 illuminate<B : IBxDF>(B bxdf, SurfaceGeometry geom, float3 wo);
};

struct AmbientLight : ILight
{
    float3 color;

    __init(float3 color)
    {
        this.color = color;
    }

    [Differentiable]
    float3 illuminate<B : IBxDF>(B bxdf, SurfaceGeometry geom, float3 wo)
    {
        return color;
    }
};

struct DirectionalLight : ILight
{
    float3 direction;
    float3 color;

    __init(float3 direction, float3 color)
    {
        this.direction = direction;
        this.color = color;
    }

    [Differentiable]
    float3 illuminate<B : IBxDF>(B bxdf, SurfaceGeometry geom, float3 wo)
    {
        let wi = -direction;
        let Li = color;

        return integrateSingleRay(bxdf, geom, wo, wi, Li);
    }
};

struct PointLight : ILight
{
    float3 position;
    float3 color;
    float3 attenuation;

    __init(float3 position, float3 color, float3 attenuation = float3(1, 0, 0))
    {
        this.position = position;
        this.color = color;
        this.attenuation = attenuation;
    }

    [Differentiable]
    float3 illuminate<B : IBxDF>(B bxdf, SurfaceGeometry geom, float3 wo)
    {
        let wi = normalize(position - geom.position);
        let dis = length(position - geom.position);
        let attenu = 1.0f / (attenuation.x + attenuation.y * dis + attenuation.z * dis * dis);
        let Li = color * attenu;

        return integrateSingleRay(bxdf, geom, wo, wi, Li);
    }
};

/// Test for Lighting

#define DEBUG
#include <debug.slangh>

struct TestLightParams
{
    float3 normal;
    float3 viewDir;
    float3 fragPos;
    float shininess;
};

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void TestLight(PointLight light, TestLightParams params)
{
    float3 normal = params.normal;
    float3 viewDir = params.viewDir;
    float3 fragPos = params.fragPos;
    float shininess = params.shininess;
    float3 kd = float3(0.7f, 0.6f, 0.5f);
    float3 ks = float3(1.0f, 1.0f, 1.0f);

    pdebug3(normal);
    pdebug3(viewDir);
    pdebug3(fragPos);
    pdebug(shininess);

    pdebug3(light.position);
    pdebug3(light.color);
    pdebug3(light.attenuation);

    let phong = Phong(kd, ks, shininess);

    let vv = phong.eval(viewDir, light.position, normal);

    pdebug3(vv);

    float3 lighting = light.illuminate(Phong(kd, ks, shininess), SurfaceGeometry(fragPos, normal, float2(0)), viewDir);

    pdebug3(lighting);
}
