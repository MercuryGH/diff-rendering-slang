import math;
import mesh;

interface IBxDF: IDifferentiable
{
    [Differentiable]
    float3 eval(float3 wo, float3 wi, float3 normal);
};

struct Lambertian : IBxDF
{
    float3 albedo;

    [Differentiable]
    __init(float3 albedo)
    {
        this.albedo = albedo;
    }

    [Differentiable]
    float3 eval(float3 wo, float3 wi, float3 normal)
    {
        return albedo / math.PI;
    }
};

/// Phong BRDF, w/ Blinn's half vector
/// Ref: https://link.springer.com/article/10.1007/s00371-020-01954-x
struct Phong : IBxDF
{
    float3 kd; // diffuse albedo
    float3 ks; // specular albedo
    float shininess;

    [Differentiable]
    __init(float3 kd, float3 ks, float shininess)
    {
        this.kd = kd;
        this.ks = ks;
        this.shininess = shininess;
    }

    [Differentiable]
    float3 eval(float3 wo, float3 wi, float3 normal)
    {
        float3 h = normalize(wo + wi);
        float diffuse = max(0, dot(normal, wi)) + 1e-3;
        float specular = pow(max(0, dot(normal, h)), shininess);

        return kd / math.PI + ks * (shininess + 1) / (2 * math.PI) / diffuse * specular;
    }
};

/// Modified Blinn-Phong BRDF, w/ extra geometric term on specular
struct ModifiedPhong : IBxDF
{
    float3 kd;
    float3 ks;
    float shininess;

    [Differentiable]
    __init(float3 kd, float3 ks, float shininess)
    {
        this.kd = kd;
        this.ks = ks;
        this.shininess = shininess;
    }

    [Differentiable]
    float3 eval(float3 wo, float3 wi, float3 normal)
    {
        float3 h = normalize(wo + wi);
        float specular = pow(max(0, dot(normal, h)), shininess);

        return kd / math.PI + ks * (shininess + 2) / (2 * math.PI) * specular;
    }
};

[Differentiable]
float D_GGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float d = NdotH2 * (a2 - 1) + 1;
    return a2 / (math.PI * d * d);
}

[Differentiable]
float G_Smith(float NdotV, float NdotL, float roughness)
{
    float a = roughness + 1.0;
    float k = (a * a) / 8.0;
    float G1V = NdotV / (NdotV * (1 - k) + k);
    float G1L = NdotL / (NdotL * (1 - k) + k);
    return G1V * G1L;
}

[Differentiable]
float3 F_Schlick(float VdotH, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1 - VdotH, 5);
}

struct CookTorrance : IBxDF
{
    float3 albedo;
    float metallic;
    float roughness;

    [Differentiable]
    __init(float3 albedo, float metallic, float roughness)
    {
        this.albedo = albedo;
        this.metallic = metallic;
        this.roughness = roughness;
    }

    [Differentiable]
    float3 eval(float3 wo, float3 wi, float3 normal)
    {
        float3 h = normalize(wo + wi);
        float NdotH = max(0, dot(normal, h));
        float NdotV = max(0, dot(normal, wo));
        float NdotL = max(0, dot(normal, wi));
        float VdotH = max(0, dot(wo, h));

        float D = D_GGX(NdotH, roughness);
        float G = G_Smith(NdotV, NdotL, roughness);

        float3 F0 = float3(0.04); // non-metallic
        F0 = lerp(F0, albedo, metallic);
        float3 F = F_Schlick(VdotH, F0);
        float3 ks = F;
        float3 kd = (1 - ks) * (1.0 - metallic);

        return kd * albedo / math.PI + ks * D * G * F / (4 * NdotV * NdotL + 1e-3);
    }
};

[Differentiable]
float3 integrateSingleRay<B : IBxDF>(B bxdf, SurfaceGeometry geom, float3 wo, float3 wi, float3 Li)
{
    return bxdf.eval(wo, wi, geom.normal) * Li * max(0, dot(wi, geom.normal));
}
