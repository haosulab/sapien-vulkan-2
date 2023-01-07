#version 460
#extension GL_EXT_ray_tracing : require

#include "ray.glsl"
#include "push_constant.glsl"

layout(location=0) rayPayloadInEXT Ray ray;

layout(set = 1, binding = 10) uniform samplerCube samplerEnvironment;  // TODO: check this

void main() {
  vec3 dir = ray.direction;
  dir = vec3(-dir.y, dir.z, -dir.x);
  ray.radiance = texture(samplerEnvironment, dir).xyz;
  ray.depth = maxDepth + 1;
}
