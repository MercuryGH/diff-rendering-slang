struct Mesh
{
    // ^ only float supported in DiffTensorView, use reinterpret<int> to unpack it
    // [bs, num of face, 3]
    DiffTensorView<float> faces;
    DiffTensorView<float> normal_indices;
    DiffTensorView<float> uv_indices;

    DiffTensorView<float> vertices;   // [bs, num of vertex, 3]
    DiffTensorView<float> normals;    // [bs, num of vertex, 3]
    DiffTensorView<float> tex_coords; // [bs, num of vertex, 2]
};

