from typing import Optional, Literal
from attr import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.obj_io import load_obj, save_obj
from utils.vertex_normals import vertex_normals
from utils.face_vertices import face_vertices


class Mesh(object):
    """
    A simple class for creating and manipulating trimesh objects
    """

    faces: torch.Tensor
    vertices: torch.Tensor
    normals: Optional[torch.Tensor]
    normal_indices: Optional[torch.Tensor]
    tex_coords: Optional[torch.Tensor]
    uv_indices: Optional[torch.Tensor]
    textures: torch.Tensor
    texture_res: int = 1
    texture_type: str = "surface"

    @property
    def device(self):
        return self.vertices.device

    @property
    def batch_size(self):
        return self.vertices.shape[0]

    @property
    def num_vertices(self):
        return self.vertices.shape[1]

    @property
    def num_faces(self):
        return self.faces.shape[1]

    def __init__(
        self,
        vertices: torch.Tensor | np.ndarray,
        faces: torch.Tensor | np.ndarray,
        normals: torch.Tensor | np.ndarray | None = None,
        normal_indices: torch.Tensor | np.ndarray | None = None,
        tex_coords: torch.Tensor | np.ndarray | None = None,
        uv_indices: torch.Tensor | np.ndarray | None = None,
        textures: torch.Tensor | np.ndarray | None = None,
        texture_res=1,
        texture_type: Literal["surface", "vertex"] = "surface",
    ):
        """
        vertices, faces and textures(if not None) are expected to be Tensor objects
        """
        if isinstance(vertices, np.ndarray):
            self.vertices = torch.from_numpy(vertices).float().cuda()
        else:
            self.vertices = vertices

        if isinstance(faces, np.ndarray):
            self.faces = torch.from_numpy(faces).int().cuda()
        else:
            self.faces = faces

        if normals is not None:
            if isinstance(normals, np.ndarray):
                self.normals = torch.from_numpy(normals).float().cuda()
            else:
                self.normals = normals

        if normal_indices is not None:
            if isinstance(normal_indices, np.ndarray):
                self.normal_indices = torch.from_numpy(normal_indices).int().cuda()
            else:
                self.normal_indices = normal_indices

        if tex_coords is not None:
            if isinstance(tex_coords, np.ndarray):
                self.tex_coords = torch.from_numpy(tex_coords).float().cuda()
            else:
                self.tex_coords = tex_coords

        if uv_indices is not None:
            if isinstance(uv_indices, np.ndarray):
                self.uv_indices = torch.from_numpy(uv_indices).int().cuda()
            else:
                self.uv_indices = uv_indices

        if self.vertices.ndimension() == 2:
            self.vertices = self.vertices[None, :, :]
        if self.faces.ndimension() == 2:
            self.faces = self.faces[None, :, :]
        if self.normals is not None and self.normals.ndimension() == 2:
            self.normals = self.normals[None, :, :]
        if self.normal_indices is not None and self.normal_indices.ndimension() == 2:
            self.normal_indices = self.normal_indices[None, :, :]
        if self.tex_coords is not None and self.tex_coords.ndimension() == 2:
            self.tex_coords = self.tex_coords[None, :, :]
        if self.uv_indices is not None and self.uv_indices.ndimension() == 2:
            self.uv_indices = self.uv_indices[None, :, :]

        self.texture_type = texture_type

        # create textures
        if textures is None:
            if texture_type == "surface":
                self.textures = torch.ones(
                    self.batch_size,
                    self.num_faces,
                    texture_res**2,
                    3,
                    dtype=torch.float32,
                ).to(self.device)
                self.texture_res = texture_res
            elif texture_type == "vertex":
                self.textures = torch.ones(
                    self.batch_size, self.num_vertices, 3, dtype=torch.float32
                ).to(self.device)
                self.texture_res = 1  # vertex color doesn't need resolution
        else:
            if isinstance(textures, np.ndarray):
                textures = torch.from_numpy(textures).float().cuda()
            if textures.ndimension() == 3 and texture_type == "surface":
                textures = textures[None, :, :, :]
            if textures.ndimension() == 2 and texture_type == "vertex":
                textures = textures[None, :, :]
            self.textures = textures
            self.texture_res = int(np.sqrt(self.textures.shape[2]))

    @classmethod
    def from_obj(
        cls,
        filename_obj,
        normalization=False,
        load_texture=False,
        texture_res=1,
        texture_type: Literal["surface", "vertex"] = "surface",
    ):
        """
        Create a Mesh object from a .obj file
        """

        vertices, normals, tex_coords, faces, textures, normal_indices, uv_indices = (
            load_obj(
                filename_obj,
                normalization=normalization,
                texture_res=texture_res,
                load_texture=False,
            )
        )
        if not load_texture:
            textures = None

        return cls(
            vertices=vertices,
            faces=faces,
            normals=normals,
            normal_indices=normal_indices,
            tex_coords=tex_coords,
            uv_indices=uv_indices,
            textures=textures,
            texture_res=texture_res,
            texture_type=texture_type,
        )

    def save_obj(self, filename_obj, save_texture=False, texture_res_out=16):
        if self.batch_size != 1:
            raise ValueError("Could not save when batch size > 1")
        if save_texture:
            save_obj(
                filename_obj,
                self.vertices[0],
                self.faces[0],
                textures=self.textures[0],
                texture_res=texture_res_out,
                texture_type=self.texture_type,
            )
        else:
            save_obj(filename_obj, self.vertices[0], self.faces[0], textures=None)

    @property
    def face_vertices(self):
        return face_vertices(self.vertices, self.faces)

    @property
    def surface_normals(self):
        v10 = self.face_vertices[:, :, 0] - self.face_vertices[:, :, 1]
        v12 = self.face_vertices[:, :, 2] - self.face_vertices[:, :, 1]
        return F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)

    @property
    def vertex_normals(self):
        return vertex_normals(self.vertices, self.faces)

    @property
    def face_textures(self):
        if self.texture_type in ["surface"]:
            return self.textures
        elif self.texture_type in ["vertex"]:
            return face_vertices(self.textures, self.faces)
        else:
            raise ValueError("texture type not applicable")

    def serialize(self):
        return {
            "faces": self.faces.view(torch.float32),
            "vertices": self.vertices,
            "normals": self.normals,
            "normal_indices": (
                self.normal_indices.view(torch.float32)
                if self.normal_indices is not None
                else None
            ),
            "tex_coords": self.tex_coords,
            "uv_indices": (
                self.uv_indices.view(torch.float32)
                if self.uv_indices is not None
                else None
            ),
        }

    def voxelize(self, voxel_size=32):
        pass
        # stub!
        # face_vertices_norm = self.face_vertices * voxel_size / (voxel_size - 1) + 0.5
        # return srf.voxelization(face_vertices_norm, voxel_size, False)
