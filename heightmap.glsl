#ifndef PLANETMAP_HEIGHTMAP_GLSL_
#define PLANETMAP_HEIGHTMAP_GLSL_

#include "chunk.h"

struct Vertex {
    // Position of the vertex relative to Chunk.origin
    vec3 position;
    // Texture coordinates of the vertex, in [0, 1]. For seamless sampling, this must be adjusted
    // for texel centers according to texture resolution.
    vec2 texcoords;
    // Unit vector away from the center of the sphere
    vec3 normal;
    // Unit vector perpendicular to normal
    vec3 tangent;
};

// Convert a point on a unit quad mapped to the current chunk into a point on the +Z surface of the
// unit sphere
vec3 unit_to_sphere(Chunk chunk, vec2 unit) {
    float side_length = 2.0 / pow(2, chunk.depth);
    return normalize(vec3((chunk.coords + unit) * side_length - 1.0, 1.0));
}

// Computes the vertices of a chunk, based on the gl_VertexIndex global. No vertex buffers
// necessary. The resulting vertices need to be assembled into primitives using an index buffer. We
// could emit triangles directly by repeating vertices, but this way we can benefit from the
// post-transform cache.
//
// quads: number of quads along one edge of a chunk. Must be a power of two for stitching to work.
// radius: radius of the planet
// heightmaps: array of heightmap cache slots
// chunk: chunk to generate geometry for
Vertex chunk_vertex(uint quads, float radius, sampler2DArray heightmaps, Chunk chunk) {
    // Number of vertices along an edge
    uint verts = quads + 1;
    // Integer vertex coordinates within the chunk
    uvec2 quad_coord = uvec2(gl_VertexIndex % verts, gl_VertexIndex / verts);
    uvec4 neighborhood = uvec4(chunk.neighborhood >> 24, (chunk.neighborhood >> 16) & 0xFF,
                               (chunk.neighborhood >> 8) & 0xFF, chunk.neighborhood & 0xFF);

    // Make some edges degenerate to line up with lower neighboring LoDs
    quad_coord.x &= (-1 << uint(quad_coord.y == 0) * neighborhood.y)
        & (-1 << uint(quad_coord.y == quads) * neighborhood.w);
    quad_coord.y &= (-1 << uint(quad_coord.x == 0) * neighborhood.x)
        & (-1 << uint(quad_coord.x == quads) * neighborhood.z);

    vec2 unit_coords = vec2(quad_coord) / quads; // 0..1
    float offset = 0.5 / (quads+1);              // Center of the first texel
    float range = 1.0 - (2.0 * offset);          // Distance between centers of first and last texels
    vec2 geom_coords = unit_coords * range + vec2(offset);

    float height = texture(heightmaps, vec3(geom_coords, chunk.slot)).x;

    vec3 local_normal = unit_to_sphere(chunk, unit_coords);
    
    Vertex result;
    result.position = (radius * local_normal - chunk.origin) + height * local_normal;
    result.texcoords = unit_coords;
    result.normal = local_normal;
    result.tangent = cross(local_normal, vec3(0, 1, 0));
    
    return result;
}

#endif
