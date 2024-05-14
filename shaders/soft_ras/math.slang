[PreferRecompute]
int _sign(float x)
{
    return sign(x);
}

[Differentiable]
[PreferRecompute]
float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

[Differentiable]
[PreferRecompute]
float distanceToVertex(float2 xy, float2 v)
{
    // Compute the distance from a point to a vertex
    return length(xy - v);
}

[Differentiable]
[PreferRecompute]
float distanceToEdge(float2 u, float2 v, float2 pt)
{
    float2 e = v - u;
    float2 n = float2(-e.y, e.x);
    float2 d = pt - u;
    float n_dot_d = dot(n, d);

    // Compute the signed distance from a point to an edge
    if (dot(pt - u, v - u) < 0.f)
        return distanceToVertex(pt, u) * (_sign(n_dot_d)); // u is the closest vertex
    else if (dot(pt - v, u - v) < 0.f)
        return distanceToVertex(pt, v) * (_sign(n_dot_d)); // v is the closest vertex
    else
    {
        return n_dot_d / length(n); // closest edge
    }
}

/// distance from a point to a triangle
/// if the point is inside the triangle, the distance is negative
[Differentiable]
[PreferRecompute]
float distanceToTriangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // Minimum distance to the edge of the triangle
    float d1 = distanceToEdge(v2, v1, xy);
    float d2 = distanceToEdge(v3, v2, xy);
    float d3 = distanceToEdge(v1, v3, xy);

    int sd1 = _sign(d1);
    int sd2 = _sign(d2);
    int sd3 = _sign(d3);

    if (sd1 > 0 && sd2 > 0)
        return distanceToVertex(xy, v2); // v2 is the closest vertex
    else if (sd1 > 0 && sd3 > 0)
        return distanceToVertex(xy, v1); // v1 is the closest vertex
    else if (sd2 > 0 && sd3 > 0)
        return distanceToVertex(xy, v3); // v3 is the closest vertex
    else
        return max(max(d1, d2), d3);
}

[Differentiable]
[PreferRecompute]
float soft_triangle(float2 p, float2 v1, float2 v2, float2 v3, float sigma)
{
    float d = distanceToTriangle(p, v1, v2, v3);
    return sigmoid(-_sign(d) * d * d / sigma);
}

[Differentiable]
float3 computeBarycentric2D(float3 v0, float3 v1, float3 v2, float2 p)
{
    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 p0 = float3(p.x - v0.x, p.y - v0.y, 0.0f);
    float area = cross(v0v1, v0v2).z;
    float alpha = cross(v1 - v2, p0).z / area;
    float beta = cross(v2 - v0, float3(p.x - v1.x, p.y - v1.y, 0.0f)).z / area;
    float gamma = 1.0f - alpha - beta;
    return float3(alpha, beta, gamma);
}


