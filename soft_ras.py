import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from renderers.soft_ras import SoftRas
from resources.resource import spot_obj
from utils.util import wrap_float_tensor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def apply_transform(vertices, matrix):
    """ Apply transformation defined by a matrix to vertices. """
    batch_size, num_faces, num_vertices_per_face, _ = vertices.shape
    ones = torch.ones((batch_size, num_faces, num_vertices_per_face, 1), device=vertices.device)
    vertices_hom = torch.cat([vertices, ones], dim=-1)  # Add homogeneous coordinate
    transformed_vertices = torch.einsum('bfvc,cd->bfvd', vertices_hom, matrix)
    return transformed_vertices[:, :, :, :3]  # Remove homogeneous coordinate after transformation

def get_view_matrix(angle_degrees_x, angle_degrees_y, translate_z):
    """Generate a view matrix for given angles and translation along the z-axis."""
    angle_radians_x = np.radians(angle_degrees_x)
    angle_radians_y = np.radians(angle_degrees_y)

    cos_x, sin_x = np.cos(angle_radians_x), np.sin(angle_radians_x)
    cos_y, sin_y = np.cos(angle_radians_y), np.sin(angle_radians_y)

    # Rotation matrix around X-axis
    rotate_x = torch.tensor([
        [1, 0, 0, 0],
        [0, cos_x, -sin_x, 0],
        [0, sin_x, cos_x, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # Rotation matrix around Y-axis
    rotate_y = torch.tensor([
        [cos_y, 0, sin_y, 0],
        [0, 1, 0, 0],
        [-sin_y, 0, cos_y, translate_z],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # Combine rotations and translate
    return torch.matmul(rotate_y, rotate_x)


def apply_directional_light(vertices, surface_normals, light_dir, light_intensity):
    """apply directional light to the vertices based on their surface normals and light direction."""
    light_dir = light_dir / torch.norm(light_dir)  # Normalize light direction
    dot_product = torch.sum(surface_normals * light_dir, dim=2, keepdim=True)  # Calculate dot product and keep dimension for broadcasting
    lighting = light_intensity * torch.clamp(dot_product, 0, 1)  # Clamp between 0 and 1 and apply intensity
    # Expand the lighting from each face to each vertex and coordinate
    lighting = lighting.repeat(1, 1, 3)  # Expand from [1, 5856, 1] to [1, 5856, 3] for each vertex
    lighting = lighting.unsqueeze(-1)  # Add an extra dimension for coordinates [1, 5856, 3, 1]
    lighting = lighting.repeat(1, 1, 1, 3)  # Now [1, 5856, 3, 3] for each coordinate
    return lighting

def apply_texture(vertices, uv_coords, texture):
    """ apply texture to vertices based on uv coordinates."""
    u = np.clip((uv_coords[:, 0] * texture.shape[1]).astype(int), 0, texture.shape[1] - 1)
    v = np.clip((uv_coords[:, 1] * texture.shape[0]).astype(int), 0, texture.shape[0] - 1)
    vertex_colors = texture[v, u]
    return vertex_colors

def single_forward(renderer: SoftRas, width, height, face_vertices):
    #render the given three-dimensional surface vertex data and return the rendering result
    output = renderer.forward(ctx=None, width=width, height=height, face_vertices=face_vertices)
    return output



def single_forward(renderer: SoftRas, width, height, face_vertices):
    output = renderer.forward(ctx=None, width=width, height=height, face_vertices=face_vertices)
    return output

def main():
    renderer = SoftRas() 
    print(dir(SoftRas))
    width = height = 512  # Slightly larger render size for better visualization
    angle = 30  # Camera angle
    translate_z = -10  # Camera translation along the z-axis

    # Assuming spot_obj.face_vertices and spot_obj.vertex_normals are already tensors on the correct device
    face_vertices = spot_obj.face_vertices
    #print("Shape of face_vertices:", face_vertices.shape) # Ensure shape is correct
    #Shape of face_vertices: torch.Size([1, 5856, 3, 3])
    normals = spot_obj.vertex_normals
    #print("Shape of normals:", normals.shape) # Ensure shape is correct
    surface_normals = spot_obj.surface_normals
    #print("Shape of surface_normals:", surface_normals.shape) # Ensure shape is correct
    angle_x = 45
    angle_y = 45
    view_matrix = get_view_matrix(angle_x, angle_y, translate_z).cuda()  # Adjust to the new function
    # Ensure matrix is on GPU if using CUDA
    transformed_vertices = apply_transform(face_vertices, view_matrix)

    # Assume a light direction and intensity
    light_direction = torch.tensor([0, 0, 1], dtype=torch.float32, device=face_vertices.device)
    light_intensity = 1.0
    lighting = apply_directional_light(transformed_vertices, surface_normals, light_direction, light_intensity)

    # Combine vertices with lighting
    colored_vertices = transformed_vertices * lighting # Updated broadcasting to include lighting
    # Forward rendering with transformed and colored vertices
    output = single_forward(renderer, width, height, colored_vertices)
    plt.figure(figsize=(10, 10))
    plt.imshow(output.cpu().numpy())  # Ensure to move output to CPU before converting to numpy for visualization
    plt.show()

if __name__ == '__main__':
    main()
