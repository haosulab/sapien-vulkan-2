#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inColor;
layout(location = 2) in flat uvec4 inSegmentation;

layout(location = 0) out vec4 outPointCloud;
layout(location = 1) out uvec4 outSegmentation;

void main() {
  outPointCloud = inColor;
  outSegmentation = inSegmentation;
}
