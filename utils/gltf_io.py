from pygltflib import GLTF2
import trimesh
import torch
import numpy as np


def load_gltf(
    filename: str,
    normalization=False,
):
    """
    Load GLTF file.
    This function supports:
        * vertices
        * faces
        * normals
        * texture coordinates
    """

    scene = trimesh.load(filename)
    if len(scene.geometry) > 0:  # type: ignore
        print("Loading meshes:", list(scene.geometry.keys()))  # type: ignore

    vertices = torch.zeros(1)
    vertex_normals = torch.zeros(1)
    uv = torch.zeros(1)
    faces = torch.zeros(1)

    for mesh_name, mesh in scene.geometry.items():  # type: ignore
        # TODO: add support for multiple meshes
        # print(mesh.faces)
        vertices = torch.from_numpy(mesh.vertices.copy()).float().cuda()
        faces = torch.from_numpy(mesh.faces.copy()).int().cuda()
        vertex_normals = torch.from_numpy(mesh.vertex_normals.copy()).float().cuda()
        uv = torch.from_numpy(mesh.visual.uv.copy()).float().cuda()
        # print(mesh.visual.uv)

        print("Loaded mesh:", mesh_name)
        print("Vertices:", vertices)
        print("Faces:", faces)
        print("Normals:", vertex_normals)
        print("UVs:", uv)
        break

    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    return vertices, vertex_normals, uv, faces


if __name__ == "__main__":
    load_gltf("resources/Duck.glb")
