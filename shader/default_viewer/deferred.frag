#version 450

layout (constant_id = 0) const int NUM_DIRECTIONAL_LIGHTS = 3;
layout (constant_id = 1) const int NUM_POINT_LIGHTS = 10;
layout (constant_id = 2) const int NUM_DIRECTIONAL_LIGHT_SHADOWS = 1;
layout (constant_id = 3) const int NUM_POINT_LIGHT_SHADOWS = 3;
layout (constant_id = 4) const int NUM_CUSTOM_LIGHT_SHADOWS = 1;
layout (constant_id = 6) const int NUM_SPOT_LIGHTS = 10;
layout (constant_id = 5) const int NUM_SPOT_LIGHT_SHADOWS = 10;

#include "../common/lights.glsl"

layout(set = 0, binding = 0) uniform SceneBuffer {
  vec4 ambientLight;
  DirectionalLight directionalLights[3];
  SpotLight spotLights[10];
  PointLight pointLights[10];
} sceneBuffer;

struct LightBuffer {
  mat4 viewMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrix;
  mat4 projectionMatrixInverse;
};

layout(set = 0, binding = 1) uniform ShadowBuffer {
  LightBuffer directionalLightBuffers[3];
  LightBuffer spotLightBuffers[10];
  LightBuffer pointLightBuffers[60];
  LightBuffer customLightBuffers[1];
} shadowBuffer;

layout(set = 0, binding = 2) uniform samplerCubeArray samplerPointLightDepths;
layout(set = 0, binding = 3) uniform sampler2DArray samplerDirectionalLightDepths;
layout(set = 0, binding = 4) uniform sampler2DArray samplerCustomLightDepths;
layout(set = 0, binding = 5) uniform sampler2DArray samplerSpotLightDepths;

layout(set = 1, binding = 0) uniform CameraBuffer {
  mat4 viewMatrix;
  mat4 projectionMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrixInverse;
  mat4 prevViewMatrix;
  mat4 prevViewMatrixInverse;
  float width;
  float height;
} cameraBuffer;

layout(set = 2, binding = 0) uniform sampler2D samplerAlbedo;
layout(set = 2, binding = 1) uniform sampler2D samplerPosition;
layout(set = 2, binding = 2) uniform sampler2D samplerSpecular;
layout(set = 2, binding = 3) uniform sampler2D samplerNormal;
layout(set = 2, binding = 4) uniform sampler2D samplerGbufferDepth;
layout(set = 2, binding = 5) uniform sampler2D samplerCustom;
layout(set = 2, binding = 6) uniform samplerCube samplerEnvironment;
layout(set = 2, binding = 7) uniform sampler2D samplerBRDFLUT;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outLighting;

vec4 world2camera(vec4 pos) {
  return cameraBuffer.viewMatrix * pos;
}

vec3 getBackgroundColor(vec3 texcoord) {
  return pow(textureLod(samplerEnvironment, texcoord, 0).rgb, vec3(2.2));
}

vec3 diffuseIBL(vec3 albedo, vec3 N) {
  vec3 color = pow(textureLod(samplerEnvironment, N, 5).rgb, vec3(2.2));
  return color * albedo;
}

vec3 specularIBL(vec3 fresnel, float roughness, vec3 N, vec3 V) {
  float dotNV = max(dot(N, V), 0);
  vec3 R = 2 * dot(N, V) * N - V;
  vec3 color = pow(textureLod(samplerEnvironment, R, roughness * 5).rgb, vec3(2.2));
  vec2 envBRDF = texture(samplerBRDFLUT, vec2(roughness, dotNV)).xy;
  return color * (fresnel * envBRDF.x + envBRDF.y);
}

vec3 project(mat4 proj, vec3 point) {
  vec4 v = proj * vec4(point, 1);
  return v.xyz / v.w;
}

const int PCF_SampleCount = 25;
vec2 PCF_Samples[PCF_SampleCount] = {
  {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
  {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
  {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0},
  {-2, 1}, {-1, 1}, {0, 1}, {1, 1}, {2, 1},
  {-2, 2}, {-1, 2}, {0, 2}, {1, 2}, {2, 2}
};

// vec2 PCSS_Rotate(vec2 offset, vec2 rotationTrig) {
//   return vec2(rotationTrig.x * offset.x - rotationTrig.y * offset.y,
//               rotationTrig.y * offset.x + rotationTrig.x * offset.y);
// }

// float PCSS_BlockerDistance(
//     sampler2DArray shadowTex, int shadowIndex, mat4 shadowProjInv,
//     vec3 projCoord, float searchUV, vec2 rotationTrig)
// {
// 	// Perform N samples with pre-defined offset and random rotation, scale by input search size
// 	int blockers = 0;
// 	float avgBlocker = 0.0f;
// 	for (int i = 0; i < PCSS_SampleCount; i++)
// 	{
// 		vec2 offset = PCSS_Samples[i] * searchUV;
// 		offset = PCSS_Rotate(offset, rotationTrig);

// 		// Compare given sample depth with receiver depth, if it puts receiver into shadow, this sample is a blocker
//     float z = texture(shadowTex, vec3(projCoord.xy + offset, shadowIndex)).x;

// 		if (z < projCoord.z)
// 		{
// 			blockers++;
// 			avgBlocker += -project(shadowProjInv, vec3(0,0,z)).z;
// 		}
// 	}

// 	// Calculate average blocker depth
// 	avgBlocker /= blockers;

// 	// To solve cases where there are no blockers - we output 2 values - average blocker depth and no. of blockers
//   if (blockers == 0) {
//     return -1;
//   }
// 	return avgBlocker;
// }

float ShadowMapPCF(
    sampler2DArray shadowTex, int shadowIndex,
    vec3 projCoord, float resolution, float searchUV, float filterSize)
{
	float shadow = 0.0f;
	vec2 grad = fract(projCoord.xy * resolution + 0.5f);

	for (int i = 0; i < PCF_SampleCount; i++)
	{
    vec4 tmp = textureGather(shadowTex, vec3(projCoord.xy +
                                             filterSize * PCF_Samples[i] * searchUV,
                                             shadowIndex));
    tmp.x = tmp.x < projCoord.z ? 0.0f : 1.0f;
    tmp.y = tmp.y < projCoord.z ? 0.0f : 1.0f;
    tmp.z = tmp.z < projCoord.z ? 0.0f : 1.0f;
    tmp.w = tmp.w < projCoord.z ? 0.0f : 1.0f;
    shadow += mix(mix(tmp.w, tmp.z, grad.x), mix(tmp.x, tmp.y, grad.x), grad.y);
  }
	return shadow / PCF_SampleCount;
}

float interleavedGradientNoise(vec2 position_screen)
{
  const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
  return fract(magic.z * fract(dot(position_screen, magic.xy)));
}

const float eps = 1e-2;


void main() {
  vec3 albedo = texture(samplerAlbedo, inUV).xyz;
  vec3 frm = texture(samplerSpecular, inUV).xyz;
  float specular = frm.x;
  float roughness = frm.y;
  float metallic = frm.z;

  vec3 normal = normalize(texture(samplerNormal, inUV).xyz);
  float depth = texture(samplerGbufferDepth, inUV).x;
  vec4 csPosition = cameraBuffer.projectionMatrixInverse * (vec4(inUV * 2 - 1, depth, 1));
  csPosition /= csPosition.w;

  vec3 camDir = -normalize(csPosition.xyz);

  vec3 diffuseAlbedo = albedo * (1 - metallic);
  vec3 fresnel = specular * (1 - metallic) + albedo * metallic;

  vec3 color = vec3(0.f);

  // point light
  for (int i = 0; i < NUM_POINT_LIGHT_SHADOWS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;

    vec3 wsl = vec3(cameraBuffer.viewMatrixInverse * vec4(l - normal * eps, 0));

    mat4 shadowProj = shadowBuffer.pointLightBuffers[6 * i].projectionMatrix;
    vec3 v = abs(wsl);
    vec4 p = shadowProj * vec4(0, 0, -max(max(v.x, v.y), v.z), 1);
    float pixelDepth = p.z / p.w;
    float shadowDepth = texture(samplerPointLightDepths, vec4(-wsl, i)).x;

    float visibility = step(pixelDepth - shadowDepth, 0);
    color += visibility * computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_POINT_LIGHT_SHADOWS; i < NUM_POINT_LIGHTS; i++) {
    vec3 pos = world2camera(vec4(sceneBuffer.pointLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    color += computePointLight(
        sceneBuffer.pointLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // directional light
  for (int i = 0; i < NUM_DIRECTIONAL_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.directionalLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.directionalLightBuffers[i].projectionMatrix;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz + normal * eps), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float resolution = textureSize(samplerDirectionalLightDepths, 0).x;

    float visibility = ShadowMapPCF(
        samplerDirectionalLightDepths, i, shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    // float visibility = step(shadowMapCoord.z - texture(samplerDirectionalLightDepths, vec3(shadowMapCoord.xy, i)).x, 0);

    color += visibility * computeDirectionalLight(
        mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_DIRECTIONAL_LIGHT_SHADOWS; i < NUM_DIRECTIONAL_LIGHTS; ++i) {
    color += computeDirectionalLight(
        mat3(cameraBuffer.viewMatrix) * sceneBuffer.directionalLights[i].direction.xyz,
        sceneBuffer.directionalLights[i].emission.rgb,
        normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  // spot light
  for (int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.spotLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.spotLightBuffers[i].projectionMatrix;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz + normal * eps), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    // float visibility = step(shadowMapCoord.z - texture(samplerSpotLightDepths, vec3(shadowMapCoord.xy, i)).x, 0);

    // float r = 6.28318531 * interleavedGradientNoise(inUV * vec2(cameraBuffer.width, cameraBuffer.height));

    float resolution = textureSize(samplerSpotLightDepths, 0).x;
    float visibility = ShadowMapPCF(
        samplerSpotLightDepths, i, shadowMapCoord.xyz, resolution, 1 / resolution, 1);

    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    vec3 l = pos - csPosition.xyz;
    color += visibility * computeSpotLight(
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  for (int i = NUM_SPOT_LIGHT_SHADOWS; i < NUM_SPOT_LIGHTS; ++i) {
    vec3 pos = world2camera(vec4(sceneBuffer.spotLights[i].position.xyz, 1.f)).xyz;
    vec3 l = pos - csPosition.xyz;
    vec3 centerDir = mat3(cameraBuffer.viewMatrix) * sceneBuffer.spotLights[i].direction.xyz;
    color += computeSpotLight(
        sceneBuffer.spotLights[i].direction.a,
        centerDir,
        sceneBuffer.spotLights[i].emission.rgb,
        l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }


  for (int i = 0; i < NUM_CUSTOM_LIGHT_SHADOWS; ++i) {
    mat4 shadowView = shadowBuffer.customLightBuffers[i].viewMatrix;
    mat4 shadowProj = shadowBuffer.customLightBuffers[i].projectionMatrix;

    vec4 ssPosition = shadowView * cameraBuffer.viewMatrixInverse * vec4((csPosition.xyz + normal * eps), 1);
    vec4 shadowMapCoord = shadowProj * ssPosition;
    shadowMapCoord /= shadowMapCoord.w;
    shadowMapCoord.xy = shadowMapCoord.xy * 0.5 + 0.5;

    float visibility = step(shadowMapCoord.z - texture(samplerCustomLightDepths, vec3(shadowMapCoord.xy, i)).x, 0);
    // visibility *= pow(texture(samplerLightMap, shadowMapCoord.xy).x, 2.2);  // un-gamma
    visibility *= step(shadowMapCoord.x, 1) * step(0, shadowMapCoord.x) * step(shadowMapCoord.y, 1) * step(0, shadowMapCoord.y);

    vec4 lightWPos = shadowBuffer.customLightBuffers[i].viewMatrixInverse * vec4(0,0,0,1);
    vec4 lightCPos = cameraBuffer.viewMatrix * lightWPos;
    vec3 l = (lightCPos.xyz - csPosition.xyz);

    color += visibility * computePointLight(vec3(1.f), l, normal, camDir, diffuseAlbedo, roughness, fresnel);
  }

  vec3 wnormal = mat3(cameraBuffer.viewMatrixInverse) * normal;

  // diffuse IBL
  color += diffuseIBL(diffuseAlbedo, wnormal);

  // specular IBL
  color += specularIBL(fresnel, roughness,
                       wnormal,
                       mat3(cameraBuffer.viewMatrixInverse) * camDir);

  color += sceneBuffer.ambientLight.rgb * diffuseAlbedo;

  if (depth == 1) {
    outLighting = vec4(getBackgroundColor((mat3(cameraBuffer.viewMatrixInverse) * csPosition.xyz)), 1.f);
  } else {
    outLighting = vec4(color, 1);
  }
}