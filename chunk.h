#ifndef PLANETMAP_CHUNK_H_
#define PLANETMAP_CHUNK_H_

struct Chunk {
    // Position within the chunk's depth in the quadtree
    uvec2 coords;
    // Depth in the quadtree
    uint depth;
    // Index of heightmap in the array texture.
    uint slot;
    // Position of the chunk relative to the center of the planet, as if on the +Z face of the
    // cubemap
    vec3 origin;
    // Level of detail of neighboring chunks
    uint neighborhood;
};

#endif
