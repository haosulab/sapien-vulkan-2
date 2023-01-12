layout(push_constant) uniform Constants {
  vec3 ambientLight;
  int frameCount;
  uint spp;
  uint maxDepth;

  uint russianRoulette;
  uint russianRouletteMinBounces;

  uint pointLightCount;
  uint directionalLightCount;
  uint spotLightCount;
};
