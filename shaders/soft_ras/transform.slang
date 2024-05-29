

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

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void TestTransform(Transform transform)
{
    let transform_mat = transform.getMatrix();

    printf("transform_mat row0: %f %f %f %f\n", transform_mat[0][0], transform_mat[0][1], transform_mat[0][2], transform_mat[0][3]);
    printf("transform_mat row1: %f %f %f %f\n", transform_mat[1][0], transform_mat[1][1], transform_mat[1][2], transform_mat[1][3]);
    printf("transform_mat row2: %f %f %f %f\n", transform_mat[2][0], transform_mat[2][1], transform_mat[2][2], transform_mat[2][3]);
    printf("transform_mat row3: %f %f %f %f\n", transform_mat[3][0], transform_mat[3][1], transform_mat[3][2], transform_mat[3][3]);

    let p = float4(1, 0, 0, 1);
    let q = float4(3, -2, 1, 1);
    let p_t = mul(transform_mat, p);
    let q_t = mul(transform_mat, q);
    printf("p_t: %f %f %f\n", p_t.x, p_t.y, p_t.z);
    printf("q_t: %f %f %f\n", q_t.x, q_t.y, q_t.z);
}