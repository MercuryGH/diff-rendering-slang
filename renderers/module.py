import slangtorch

rasterizer2d = slangtorch.loadModule("shaders/rasterizer/rasterizer.slang", verbose=True)

camera = rasterizer2d.Camera(o=(0.0, 0.0), scale=(1.0, 1.0), frameDim=(1024, 1024))

# ---

brdf2d = slangtorch.loadModule("shaders/renderer/BRDF.slang", verbose=True)

light_kernel = slangtorch.loadModule("shaders/renderer/lightkernel.slang", verbose=True)

# softras

soft_ras = slangtorch.loadModule("shaders/soft_ras/main.slang", verbose=True)
# 加载 Toon Shader 模块
toon_shader = slangtorch.loadModule("shaders/soft_ras/toon_shader.slang", verbose=True)
