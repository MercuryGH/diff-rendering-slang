import cv2
import numpy as np

img = np.divide(cv2.imread(R"resources/diffuse.jpg")[64:64+128,64:64+128,[2,1,0]], 255)
normal = np.divide(cv2.imread(R"resources/normal.jpg")[64:64+128,64:64+128,[2,1,0]], 255)
roughness = np.divide(cv2.imread(R"resources/roughness.jpg")[64:64+128,64:64+128,[2,1,0]], 255)
roughness *= roughness

from utils.mesh import Mesh

spot_obj = Mesh.from_obj("resources/spot_simplified.obj", load_texture=False)
tex_image = np.divide(cv2.imread(R"resources/Grid.png")[:,:,[2,1,0]], 255)
# spot_obj = Mesh.from_obj("resources/single_triangle.obj", load_texture=False)
