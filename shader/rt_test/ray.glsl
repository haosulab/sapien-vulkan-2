struct Ray {
  vec3 origin;
  vec3 direction;
  vec3 albedo;
  vec3 normal;
  vec3 radiance;
  vec3 attenuation;
  uint depth;
  uint done;
  uint seed;
  vec3 debug;
};

struct ShadowRay {
  bool shadowed;
};
