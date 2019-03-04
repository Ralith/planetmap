#version 450

#include "terrain.h"

layout(location = 0) in vec2 raw_texcoords;
layout(location = 1) in vec3 base_normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in flat uint slot;

layout(location = 0) out vec4 color;

vec2 texcoords_for(sampler2DArray texture) {
    float res = textureSize(texture, 0).x;
    float offset = 0.5 / res;           // Center of the first texel
    float range = 1.0 - (2.0 * offset); // Distance between centers of first and last texels
    return raw_texcoords * range + offset;
}

void main() {
    vec3 base_normal_ = normalize(base_normal);
    vec3 tangent_ = normalize(tangent);
    vec3 bitangent = cross(base_normal_, tangent_);
    mat3 tangent_basis = mat3(tangent_, bitangent, base_normal_);

    uint array = slot / CACHE_ARRAY_SIZE;
    uint layer = slot % CACHE_ARRAY_SIZE;
    vec3 tangent_normal = normalize(texture(normals[array], vec3(texcoords_for(normals[array]), layer)).xyz);
    vec3 normal = tangent_basis * tangent_normal;

    vec4 base_color = texture(colors[array], vec3(texcoords_for(colors[array]), layer));

    vec3 sun = mat3(view) * vec3(0, 1, 0);
    color = base_color * dot(normal, sun);
}
