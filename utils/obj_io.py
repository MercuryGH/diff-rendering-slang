import os

import torch
import numpy as np


def load_textures(filename_obj: str, filename_mtl: str, texture_res: int):
    return NotImplemented


def load_obj(
    filename_obj: str,
    normalization=False,
    load_texture=False,
    texture_res=4,
    texture_type="surface",
):
    """
    Load Wavefront .obj file.
    This function supports:
        * vertices (v x x x)
        * faces (f x x x)
        * normals (vn x x x)
        * texture coordinates (vt x x)
    """

    assert texture_type in ["surface", "vertex"]

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "v":
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    normals = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "vn":
            normals.append([float(v) for v in line.split()[1:4]])
    if len(normals) != 0:
        normals = torch.from_numpy(np.vstack(normals).astype(np.float32)).cuda()
    else:
        normals = torch.zeros_like(vertices)

    tex_coords = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "vt":
            tex_coords.append([float(v) for v in line.split()[1:3]])
    if len(tex_coords) != 0:
        tex_coords = torch.from_numpy(np.vstack(tex_coords).astype(np.float32)).cuda()
    else:
        tex_coords = torch.zeros((vertices.shape[0], 2), dtype=torch.float32).cuda()

    # load faces
    faces = []
    uv_indices = []
    normal_indices = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "f":
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split("/")[0])
            if len(vs[0].split("/")) > 1:
                uv0 = int(vs[0].split("/")[1])
                if len(vs[0].split("/")) > 2:
                    n0 = int(vs[0].split("/")[2])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split("/")[0])
                v2 = int(vs[i + 2].split("/")[0])
                faces.append((v0, v1, v2))
                if len(vs[0].split("/")) > 1:
                    uv1 = int(vs[i + 1].split("/")[1])
                    uv2 = int(vs[i + 2].split("/")[1])
                    uv_indices.append((uv0, uv1, uv2))
                    if len(vs[0].split("/")) > 2:
                        n1 = int(vs[i + 1].split("/")[2])
                        n2 = int(vs[i + 2].split("/")[2])
                        normal_indices.append((n0, n1, n2))

    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    if len(uv_indices) == 0:
        uv_indices = torch.zeros_like(faces)
    else:
        uv_indices = torch.from_numpy(np.vstack(uv_indices).astype(np.int32)).cuda() - 1

    if len(normal_indices) == 0:
        normal_indices = torch.zeros_like(faces)
    else:
        normal_indices = (
            torch.from_numpy(np.vstack(normal_indices).astype(np.int32)).cuda() - 1
        )

    # load textures
    textures = None
    if load_texture and texture_type == "surface":
        textures = None
        for line in lines:
            if line.startswith("mtllib"):
                filename_mtl = os.path.join(
                    os.path.dirname(filename_obj), line.split()[1]
                )
                textures = load_textures(filename_obj, filename_mtl, texture_res)
        if textures is None:
            raise Exception("Failed to load textures.")
    elif load_texture and texture_type == "vertex":
        textures = []
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == "v":
                textures.append([float(v) for v in line.split()[4:7]])
        textures = torch.from_numpy(np.vstack(textures).astype(np.float32)).cuda()

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    return vertices, normals, tex_coords, faces, textures, uv_indices, normal_indices


def save_obj(
    filename, vertices, faces, textures=None, texture_res=16, texture_type="surface"
):
    # stub!
    pass
