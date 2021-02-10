#version 450 

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 3;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 10;

struct PointLight {
  vec4 position;
  vec4 emission;
};
struct DirectionalLight {
  vec4 direction;
  vec4 emission;
};
layout(set = 0, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
  DirectionalLight directionalLights[NUM_DIRECTIONAL_LIGHTS];
  PointLight pointLights[NUM_POINT_LIGHTS];
} sceneBuffer;

layout(set = 1, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
} cameraBuffer;

layout(set = 2, binding = 0) uniform sampler2D samplerAlbedo;
layout(set = 2, binding = 1) uniform sampler2D samplerPosition;
layout(set = 2, binding = 2) uniform sampler2D samplerSpecular;
layout(set = 2, binding = 3) uniform sampler2D samplerNormal;
layout(set = 2, binding = 4) uniform sampler2D samplerDepth;
layout(set = 2, binding = 5) uniform sampler2D samplerCustom;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outLighting;

vec4 world2camera(vec4 pos) {
  return cameraBuffer.viewMatrix * pos;
}

vec3 getBackgroundColor(vec3 texcoord) {
  return vec3(1,1,1);
}

float diffuse(vec3 L, vec3 V, vec3 N) {
  float NoL = dot(N, L);
  return max(NoL, 0.f) / 3.141592653589793f;
}

float ggx(vec3 L, vec3 V, vec3 N, float roughness, float fresnel) {
  float NoV = dot(N, V);
  float NoL = dot(N, L);
  if (NoV <= 0 || NoL <= 0) {
    return 0;
  }

  vec3 H = normalize((L+V) / 2);
  NoL = clamp(NoL, 1e-6, 1);
  NoV = clamp(NoV, 1e-6, 1);
  float NoH = clamp(dot(N, H), 1e-6, 1);
  float VoH = clamp(dot(V, H), 1e-6, 1);

  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;
  float k = (alpha + 2 * roughness + 1.0) / 8.0;
  float FMi = ((-5.55473) * VoH - 5.98316) * VoH;
  float frac0 = fresnel + (1 - fresnel) * pow(2.0, FMi);
  float frac = frac0 * alpha2;
  float nom0 = NoH * NoH * (alpha2 - 1) + 1;
  float nom1 = NoV * (1 - k) + k;
  float nom2 = NoL * (1 - k) + k;
  float nom = clamp((4 * 4 * 3.141592653589793f * nom0 * nom0 * nom1 * nom2), 1e-6, 4 * 3.141592653589793f);
  float spec = frac / nom;

  return spec * NoL;
}

void main() {
  vec3 albedo = texture(samplerAlbedo, inUV).xyz;
  vec3 frm = texture(samplerSpecular, inUV).xyz;
  float F0 = frm.x;
  float roughness = frm.y;
  float metallic = frm.z;

  vec3 normal = texture(samplerNormal, inUV).xyz;
  vec4 csPosition = texture(samplerPosition, inUV);
  vec3 camDir = -normalize(csPosition.xyz);

  vec3 color = vec3(0.f);
  for (int i = 0; i < NUM_POINT_LIGHTS; i++) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    float d = max(length(l), 0.0001);

    if (length(l) == 0) {
      continue;
    }

    vec3 lightDir = normalize(l);

    // diffuse
    color += (1 - metallic) * albedo * sceneBuffer.pointLights[i].emission.rgb *
             diffuse(lightDir, camDir, normal) / d / d;

    // metallic
    color += metallic * albedo * sceneBuffer.pointLights[i].emission.rgb *
             ggx(lightDir, camDir, normal, roughness, 1.f) / d / d;

    // specular
    color += sceneBuffer.pointLights[i].emission.rgb * ggx(lightDir, camDir, normal, roughness, F0) / d / d;
  }

  for (int i = 0; i < NUM_DIRECTIONAL_LIGHTS; ++i) {
    if (length(sceneBuffer.directionalLights[i].direction.xyz) == 0) {
      continue;
    }

    vec3 lightDir = -normalize((cameraBuffer.viewMatrix *
                                vec4(sceneBuffer.directionalLights[i].direction.xyz, 0)).xyz);

    // diffuse
    color += (1 - metallic) * albedo * sceneBuffer.directionalLights[i].emission.rgb *
             diffuse(lightDir, camDir, normal);

    // metallic
    color += metallic * albedo * sceneBuffer.directionalLights[i].emission.rgb *
             ggx(lightDir, camDir, normal, roughness, 1.f);

    // specular
    color += sceneBuffer.directionalLights[i].emission.rgb * ggx(lightDir, camDir, normal, roughness, F0);
  }

  color += sceneBuffer.ambientLight.rgb * albedo;

  float depth = texture(samplerDepth, inUV).x;
  if (depth == 1) {
    outLighting = vec4(getBackgroundColor((cameraBuffer.viewMatrixInverse * csPosition).xyz), 1.f);
  } else {
    outLighting = vec4(color, 1);
  }
}


