#version 450

layout(set = 3, binding = 0) uniform MaterialBuffer {
  vec4 baseColor;
  float fresnel;
  float roughness;
  float metallic;
  float transparency;
  int textureMask;
} materialBuffer;

layout(set = 3, binding = 1) uniform sampler2D colorTexture;
layout(set = 3, binding = 2) uniform sampler2D roughnessTexture;
layout(set = 3, binding = 3) uniform sampler2D normalTexture;
layout(set = 3, binding = 4) uniform sampler2D metallicTexture;

layout(location = 0) in vec4 inPosition;

// required output textures
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outPosition;
layout(location = 2) out vec4 outSpecular;
layout(location = 3) out vec4 outNormal;
layout(location = 4) out uvec4 outSegmentation;

void main() {
}
