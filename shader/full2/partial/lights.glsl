struct PointLight {
  vec4 position;
  vec4 emission;
};

struct DirectionalLight {
  vec4 direction;
  vec4 emission;
};

struct SpotLight {
  vec4 position;
  vec4 direction;
  vec4 emission;
};
