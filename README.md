# Planetmap

Planetmap is a library for processing lazily evaluated radial
functions with dynamic level-of-detail.

## Terrain Rendering

Planetmap is motivated by the desire for visualizing to-scale
planetary terrain in real time on commodity hardware. To accomplish
this, it addresses multiple challenges:

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

The transform of a `Chunk` relative to a camera can be computed on the
CPU using double precision and then rounded to single precision. This
allows for efficient processing on a commodity GPU while providing
high precision for positions close to the camera, because while a
`Chunk` may be distant enough from the center of the planet to produce
jitter when naively rendered from a nearby camera, its distance to
such a camera is small, and can hence be represented much more
accurately.

Because every `Chunk` has a slightly different shape when mapped onto
the surface of a sphere, to avoid discontinuities at LoD boundaries,
and to reduce GPU memory and memory bandwith requirements, it is
convenient to generate vertexes on-demand on the GPU based on
heightmap data and `Chunk` coordinates. `heightmap.glsl` provides an
example of this suitable for use in a vertex shader.

## Example

Planetmap includes a simple, unoptimized Vulkan demo that can be
launched with `cargo run --release --features=ash --example demo`. Use
WASDRF and click-and-drag to move; velocity is proportional to
altitude. For simplicity, it skips important optimizations such as
generating and transferring chunks asynchronously in background
threads, memoizing noise function samplings across height/color/normal
map computation, and precomputing coarse octaves of the noise function
used. The jittery performance which results can be avoided in a real
application by not blocking frame rendering on chunk generation.
