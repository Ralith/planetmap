# Planetmap

Planetmap is a library for processing lazily evaluated radial
functions with dynamic level-of-detail.

## Screenshots

![high orbit](https://user-images.githubusercontent.com/3484507/65808207-c6e2ce80-e18a-11e9-9625-6ad85e5d54d5.png)
![low orbit](https://user-images.githubusercontent.com/3484507/65808209-c77b6500-e18a-11e9-84d3-9efa44f2f902.png)
![close](https://user-images.githubusercontent.com/3484507/65808208-c6e2ce80-e18a-11e9-83e1-100f0863473d.png)

## Terrain Rendering

Planetmap is motivated by the desire to model to-scale planetary
terrain in real time on commodity hardware. To accomplish this, it
addresses multiple challenges:

- Dynamic level of detail
- Streaming larger-than-memory or procedural heightmap data
- Addressing the surface of the sphere without singularities
- Maintaining numerical stability over massive distances

This is accomplished by organizing data into a 6 virtual quadtrees,
each corresponding to one face of a cubemap. Each node in a quadtree,
referred to as a `Chunk`, can be associated with a square grid of
height samples, each of which represents the altitude relative to sea
level along a ray originating at the center of a planet. Streaming and
level of detail can then be implemented with a `CacheManager`
associated with GPU-resident buffers.

When generating geometry to be displaced by the heightmap, care must
be taken to ensure numerical stability and avoid hairline cracks
between neighboring chunks. One effective tactic is to upload the
corners of each chunk from the CPU, then find interior points using
interpolation in a vertex shader, accounting for curvature using the
law of sines.

## Features

- `ncollide` exposes helpers for nphysics-compatible collision
  detection against the surface of a radial heightmap
- `simd` exposes `simdeez`-based iterators for data-parallel
  computation of sample coordinates within a `Chunk` or
  `chunk::Coord`, useful for fast noise sampling
