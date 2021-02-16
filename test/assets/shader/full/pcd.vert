#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout(set = 1, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
} cameraBuffer;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in uvec4 segmentation;

const float size = 0.01;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outColor;
layout(location = 2) out flat uvec4 outSegmentation;

void main() {
  vec3 N = normalize(normal);
  vec3 d1 = cross(N, vec3(0,1,0));
  if (dot(d1, d1) < 1e-6) {
    d1 = cross(N, vec3(1,0,0));
  }
  d1 = normalize(d1);
  d2 = cross(N, d1);

  if (gl_VertexID % 4  == 0) {
    position = position - d1 * size - d2 * size;
  } else if (gl_VertexID % 4  == 1) {
    position = position + d1 * size - d2 * size;
  } else if (gl_VertexID % 4  == 2) {
    position = position + d1 * size + d2 * size;
  } else if (gl_VertexID % 4  == 3) {
    position = position - d1 * size + d2 * size;
  }

  outSegmentation = segmentation;
  outColor = color;
  
  mat4 view = cameraBuffer.viewMatrix;
  mat3 normalMatrix = mat3(transpose(inverse(view)));

  gl_Position = cameraBuffer.projectionMatrix * view * vec4(position, 1);
  outNormal = normalMatrix *  normal;
}
