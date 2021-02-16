#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable


layout(set = 0, binding = 0) uniform LightSpaceBuffer {
  mat4 lightViewMatrix;
  mat4 lightProjectionMatrix;
} lightSpaceBuffer;

layout(set = 1, binding = 0) uniform ObjectBuffer {
  mat4 modelMatrix;
  uvec4 segmentation;
} objectBuffer;

layout(location = 0) in vec3 position;

void main() {
}
