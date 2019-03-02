#version 450

#include "../../heightmap.glsl"

layout(constant_id = 0) const uint QUAD_COUNT = 2;
layout(constant_id = 1) const uint HEIGHTMAP_ARRAY_SIZE = 1;
layout(constant_id = 2) const uint HEIGHTMAP_ARRAY_COUNT = 1;
layout(constant_id = 3) const float RADIUS = 6371e3;

layout(set = 0, binding = 0) uniform Globals {
    mat4 projection;
};

layout(set = 0, binding = 1) uniform sampler2DArray heightmap[HEIGHTMAP_ARRAY_COUNT];

layout(location = 0) in mat4 worldview;
layout(location = 4) in uvec2 chunk_coords;
layout(location = 4, component = 2) in uint depth;
layout(location = 4, component = 3) in uint slot;
layout(location = 5) in vec3 origin;
layout(location = 6) in uint neighborhood;

void main() {
    Chunk chunk;
    chunk.coords = chunk_coords;
    chunk.depth = depth;
    chunk.slot = slot % HEIGHTMAP_ARRAY_SIZE;
    chunk.origin = origin;
    chunk.neighborhood = neighborhood;
    Vertex vert = chunk_vertex(QUAD_COUNT, RADIUS, heightmap[slot / HEIGHTMAP_ARRAY_SIZE], chunk);
    gl_Position = projection * worldview * vec4(vert.position, 1);
}
