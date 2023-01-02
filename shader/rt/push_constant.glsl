layout(push_constant) uniform Constants {
  int frameCount;
  uint spp;
  uint maxDepth;
  vec3 environmentLight;

  bool russianRoulette;
  uint russianRouletteMinBounces;

  uint pointLightCount;
  uint directionalLightCount;
  uint spotLightCount;
};
