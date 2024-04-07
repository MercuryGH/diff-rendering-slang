import cv2

img = cv2.imread(R"resources/diffuse.jpg")[64:64+128,64:64+128,[2,1,0]]/255
normal = cv2.imread(R"resources/normal.jpg")[64:64+128,64:64+128,[2,1,0]]/255
roughness = cv2.imread(R"resources/roughness.jpg")[64:64+128,64:64+128,[2,1,0]]/255
roughness *= roughness

from utils.mesh import Mesh

spot_obj = Mesh.from_obj("resources/spot_triangulated.obj", load_texture=False)
