#include "../../chunk.h"

layout(constant_id = 0) const uint QUAD_COUNT = 2;
layout(constant_id = 1) const uint CACHE_ARRAY_SIZE = 1;
layout(constant_id = 2) const uint CACHE_ARRAY_COUNT = 1;
layout(constant_id = 3) const float RADIUS = 6371e3;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 projection;
    mat4 view;
};
layout(set = 0, binding = 1) uniform sampler2DArray heightmap[CACHE_ARRAY_COUNT];
layout(set = 0, binding = 2) uniform sampler2DArray normals[CACHE_ARRAY_COUNT];
layout(set = 0, binding = 3) uniform sampler2DArray colors[CACHE_ARRAY_COUNT];

struct Instance {
    mat4 worldview;
    Chunk chunk;
};

layout(set = 0, binding = 4) readonly buffer Instances {
    Instance instances[];
};
