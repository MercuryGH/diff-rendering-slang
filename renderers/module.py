import slangpy

rasterizer2d = slangpy.loadModule("shaders/rasterizer/main.slang", verbose=True)

camera = rasterizer2d.Camera(o=(0.0, 0.0), scale=(1.0, 1.0), frameDim=(1024, 1024))

# ---

brdf2d = slangpy.loadModule("shaders/renderer/BRDF.slang", verbose=True)

light_kernel = slangpy.loadModule("shaders/renderer/lightkernel.slang", verbose=True)

