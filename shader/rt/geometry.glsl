struct GeometryInstance {
  mat4 transform;
  uint geometryIndex;
  uint materialIndex;
  int padding0;
  int padding1;
};

struct Material {
  vec4 diffuse;
  vec4 emission;

  float alpha;
  float metallic;
  float specular;
  float roughness;
  float ior;
  float transmission;

  int diffuseTextureIndex;
  int metallicTextureIndex;
  int roughnessTextureIndex;
  int emissionTextureIndex;
  int normalTextureIndex;
  int padding0;
};
