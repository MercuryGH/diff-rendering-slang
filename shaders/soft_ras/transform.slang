

struct Transform
{
    float4 r; // unit quaternion for rotation
    float3 t;
    float3 s;

    [Differentiable]
    float4x4 getMatrix()
    {
        let rot_trans = float4x4(
            float4(1.0f - 2.0f * r.y * r.y - 2.0f * r.z * r.z, 2.0f * r.x * r.y + 2.0f * r.z * r.w, 2.0f * r.x * r.z - 2.0f * r.y * r.w, t.x),
            float4(2.0f * r.x * r.y - 2.0f * r.z * r.w, 1.0f - 2.0f * r.x * r.x - 2.0f * r.z * r.z, 2.0f * r.y * r.z + 2.0f * r.x * r.w, t.y),
            float4(2.0f * r.x * r.z + 2.0f * r.y * r.w, 2.0f * r.y * r.z - 2.0f * r.x * r.w, 1.0f - 2.0f * r.x * r.x - 2.0f * r.y * r.y, t.z),
            float4(0.0f, 0.0f, 0.0f, 1.0f));

        let scale = float4x4(
            float4(s.x, 0.0f, 0.0f, 0.0f),
            float4(0.0f, s.y, 0.0f, 0.0f),
            float4(0.0f, 0.0f, s.z, 0.0f),
            float4(0.0f, 0.0f, 0.0f, 1.0f));

        return mul(rot_trans, scale);
    }
};

/// Test for transform

#define DEBUG
#include <debug.slangh>

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void TestTransform(Transform transform)
{
    let transform_mat = transform.getMatrix();

    pdebug4x4(transform_mat);

    let p = float4(1, 0, 0, 1);
    let q = float4(3, -2, 1, 1);
    let p_t = mul(transform_mat, p);
    let q_t = mul(transform_mat, q);

    pdebug3(p_t);
    pdebug3(q_t);
}