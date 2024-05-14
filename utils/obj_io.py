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
    This function only supports vertices (v x x x) and faces (f x x x).
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

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "f":
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split("/")[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split("/")[0])
                v2 = int(vs[i + 2].split("/")[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    # load textures
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

    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces


def save_obj(
    filename, vertices, faces, textures=None, texture_res=16, texture_type="surface"
):
    # stub!
    pass
