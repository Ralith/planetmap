#version 450

#include "../../heightmap.glsl"

#include "terrain.h"

layout(location = 0) in mat4 worldview;
layout(location = 4) in uvec2 chunk_coords;
layout(location = 4, component = 2) in uint depth;
layout(location = 4, component = 3) in uint slot;
layout(location = 5) in vec3 origin;
layout(location = 6) in uint neighborhood;

layout(location = 0) out vec2 texcoords;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec3 tangent;
layout(location = 3) out uint slot_out;

void main() {
    Chunk chunk;
    chunk.coords = chunk_coords;
    chunk.depth = depth;
    chunk.slot = slot % HEIGHTMAP_ARRAY_SIZE;
    chunk.origin = origin;
    chunk.neighborhood = neighborhood;
    Vertex vert = chunk_vertex(QUAD_COUNT, RADIUS, heightmap[slot / HEIGHTMAP_ARRAY_SIZE], chunk);
    gl_Position = projection * worldview * vec4(vert.position, 1);
    texcoords = vert.texcoords;
    normal = mat3(worldview) * vert.normal;
    tangent = mat3(worldview) * vert.tangent;
    slot_out = slot;
}
