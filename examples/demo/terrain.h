layout(constant_id = 0) const uint QUAD_COUNT = 2;
layout(constant_id = 1) const uint HEIGHTMAP_ARRAY_SIZE = 1;
layout(constant_id = 2) const uint HEIGHTMAP_ARRAY_COUNT = 1;
layout(constant_id = 3) const float RADIUS = 6371e3;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 projection;
    mat4 view;
};
layout(set = 0, binding = 1) uniform sampler2DArray heightmap[HEIGHTMAP_ARRAY_COUNT];
layout(set = 0, binding = 2) uniform sampler2DArray normals[HEIGHTMAP_ARRAY_COUNT];
