#version 460
#extension GL_EXT_ray_tracing : require

#include "ray.glsl"

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

void main() {
  shadowRay.shadowed = false;
}
