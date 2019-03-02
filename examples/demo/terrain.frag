#version 450

#include "terrain.h"

layout(location = 0) in vec2 texcoords;
layout(location = 1) in vec3 base_normal;
layout(location = 2) in vec3 tangent;
layout(location = 0) out vec4 color;

void main() {
    vec3 base_normal_ = normalize(base_normal);

    vec3 sun = mat3(view) * vec3(0, 1, 0);
    color = vec4(vec3(1, 1, 1) * dot(base_normal_, sun), 1);
}
