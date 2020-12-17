#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable


layout(set = 1, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
} cameraBuffer;

layout(set = 2, binding = 0) uniform ObjectBuffer {
  mat4 modelMatrix;
  uvec4 segmentation;
} objectBuffer;

layout(location = 0) in vec3 position;

layout(location = 0) out vec4 outPosition;

void main() {
}
