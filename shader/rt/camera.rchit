#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable

hitAttributeEXT vec3 attribs;

#include "random.glsl"
#include "ray.glsl"
#include "geometry.glsl"
#include "light.glsl"
#include "push_constant.glsl"

layout(location = 0) rayPayloadInEXT Ray ray;
layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

layout(set = 1, binding = 0) uniform accelerationStructureEXT tlas;

layout(set = 1, binding = 1) readonly buffer GeometryInstances
{
  GeometryInstance i[];
} geometryInstances;

layout(set = 1, binding = 2) readonly buffer Materials
{
  Material m;
} materials[];

layout(set = 1, binding = 3) readonly buffer TextureIndices
{
  TextureIndex t[];
} textureIndices;

layout(set = 1, binding = 4) uniform sampler2D textures[];

layout(set = 1, binding = 5) readonly buffer PointLights
{
  PointLight l[];
} pointLights;

layout(set = 1, binding = 6) readonly buffer DirectionalLights
{
  DirectionalLight l[];
} directionalLights;

layout(set = 1, binding = 7) readonly buffer SpotLights
{
  SpotLight l[];
} spotLights;

struct Vertex {
  float x, y, z;
  float nx, ny, nz;
  float u, v;

  float tx, ty, tz;
  float bx, by, bz;
};

layout(std430, set = 1, binding = 8) readonly buffer Vertices {
  Vertex v[];
} vertices[];

layout(set = 1, binding = 9) readonly buffer Indices {
  uint i[];
} indices[];


void unpackVertex(uint index, uint geometryIndex, out vec3 position, out vec3 normal, out vec2 uv, out vec3 tangent) {
  Vertex v = vertices[nonuniformEXT(geometryIndex)].v[index];
  position = vec3(v.x, v.y, v.z);
  normal = vec3(v.nx, v.ny, v.nz);
  uv = vec2(v.u, v.v);
  tangent = vec3(v.tx, v.ty, v.tz);
}

float maxVec3(vec3 v) {
  return max(v.x, max(v.y, v.z));
}

void sampleDiffuse(inout uint seed, in mat3 tbn, vec3 albedo, out vec3 L, out vec3 attenuation) {
  L = tbn * cosineSampleHemisphere(seed);
  attenuation = albedo;
}

void sampleGGX(inout uint seed, in mat3 tbn, vec3 F0, vec3 N, vec3 V, float roughness, out vec3 L, out vec3 attenuation) {
  float u0 = rnd(seed);
  float u1 = rnd(seed);

  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;
  float k = (alpha + 2 * roughness + 1) / 8.0;

  float z1 = (1.0 - u1) / (1.0 + (alpha2 - 1.0) * u1);
  float cosTheta = sqrt(z1);
  float sinTheta = sqrt(1.0 - z1);
  float phi = 2 * M_PI * u0;

  vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
  H = tbn * H;

  L = 2 * dot(V, H) * H - V;

  float dotNH = clamp(dot(N, H), 1e-6, 1);
  float dotVH = clamp(dot(V, H), 1e-6, 1);
  float dotNL = clamp(dot(N, L), 0, 1);
  float dotNV = clamp(dot(N, V), 1e-6, 1);

  if (dotNL >= 0) {
    vec3 F = F0 * (1 - F0) * pow(2.0, (-5.55473 * dotVH - 6.98316) * dotVH);
    float G = (dotNV * dotNL) / ((dotNV * (1 - k) + k) * (dotNL * (1 - k) + k));
    attenuation = F * G * dotVH / (dotNH * dotNV);
  } else {
    attenuation = vec3(0.0);
  }
}

vec3 evalDiffuse(float dotNL, vec3 albedo) {
  return albedo * dotNL / M_PI;
}

vec3 evalGGX(
  float dotNL,
  float dotNV,
  float dotNH,
  float dotVH,
  float roughness,
  vec3 F0
) {
  // https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;
  float k = (alpha + 2 * roughness + 1) / 8.0;

  vec3 F = F0 * (1 - F0) * pow(2.0, (-5.55473 * dotVH - 6.98316) * dotVH);

  // denominator for D
  float d0 = dotNH * dotNH * (alpha2 - 1) + 1;
  d0 = M_PI * d0 * d0;

  // denominator for G
  float d1 = (dotNV * (1 - k) + k) * (dotNL * (1 - k) + k);

  vec3 spec = (F * alpha2) / (4 * d0 * d1);

  return spec * dotNL;
}


vec3 traceDirectionalLights(vec3 pos, vec3 normal, vec3 diffuseColor, vec3 specularColor, float roughness) {
  vec3 result = vec3(0.0);
  for (uint i = 0; i < directionalLightCount; ++i) {
    DirectionalLight light = directionalLights.l[i];
    vec3 emission = light.rgb;
    vec3 L = normalize(-light.direction);
    vec3 V = normalize(-ray.direction);
    if (light.softness != 0) {
      L = normalize(L + vec3(rnd(ray.seed), rnd(ray.seed), rnd(ray.seed)) * light.softness);
    }

    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;

    shadowRay.shadowed = true;
    traceRayEXT(tlas,
                flags,
                0xff,
                0,
                0,
                1,  // miss index
                pos,
                0.001,
                L,
                1.0e6,
                1  // payload location
    );

    vec3 H = normalize(V + L);

    float dotNH = clamp(dot(normal, H), 1e-6, 1);
    float dotVH = clamp(dot(V, H), 1e-6, 1);
    float dotNL = clamp(dot(normal, L), 0, 1);
    float dotNV = clamp(dot(normal, V), 1e-6, 1);

    result += evalDiffuse(dotNL, diffuseColor) * emission;  // diffuse
    result += evalGGX(dotNL, dotNV, dotNH, dotVH, roughness, specularColor) * emission;  // specular
  }
  return result;
}

vec3 tracePointLights(vec3 pos, vec3 normal, vec3 diffuseColor, vec3 specularColor, float roughness) {
  vec3 result = vec3(0.0);
  for (uint i = 0; i < pointLightCount; ++i) {
    PointLight light = pointLights.l[i];
    vec3 emission = light.rgb;
    vec3 d = light.position - pos;
    float d2 = dot(d, d);
    vec3 L = normalize(d);
    vec3 V = normalize(-ray.direction);

    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;

    shadowRay.shadowed = true;
    traceRayEXT(tlas,
                flags,
                0xff,
                0,
                0,
                1,  // miss index
                pos,
                0.001,
                L,
                1.0e6,
                1  // payload location
    );

    vec3 H = normalize(V + L);

    float dotNH = clamp(dot(normal, H), 1e-6, 1);
    float dotVH = clamp(dot(V, H), 1e-6, 1);
    float dotNL = clamp(dot(normal, L), 0, 1);
    float dotNV = clamp(dot(normal, V), 1e-6, 1);

    result += evalDiffuse(dotNL, diffuseColor) * emission / d2;  // diffuse
    result += evalGGX(dotNL, dotNV, dotNH, dotVH, roughness, specularColor) * emission / d2;  // specular
  }
  return result;
}

void main() {
  uint geometryIndex = geometryInstances.i[gl_InstanceID].geometryIndex;
  uint materialIndex = geometryInstances.i[gl_InstanceID].materialIndex;

  uint v0 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 0];
  uint v1 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 1];
  uint v2 = indices[nonuniformEXT(geometryIndex)].i[3 * gl_PrimitiveID + 2];

  vec3 p0, p1, p2;
  vec3 n0, n1, n2;
  vec2 uv0, uv1, uv2;
  vec3 t0, t1, t2;

  unpackVertex(v0, geometryIndex, p0, n0, uv0, t0);
  unpackVertex(v1, geometryIndex, p1, n1, uv1, t1);
  unpackVertex(v2, geometryIndex, p2, n2, uv2, t2);

  float b0 = 1.0 - attribs.x - attribs.y;
  float b1 = attribs.x;
  float b2 = attribs.y;

  vec3 geometricNormal = cross(p1 - p0, p2 - p0);
  vec3 shadingNormal = n0 * b0 + n1 * b1 + n2 * b2;
  vec2 uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
  vec3 tangent = t0 * b0 + t1 * b1 + t2 * b2;

  vec3 worldGeometricNormal = normalize(vec3(geometricNormal * gl_WorldToObjectEXT));
  vec3 worldShadingNormal = normalize(vec3(shadingNormal * gl_WorldToObjectEXT));
  vec3 worldPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  vec3 worldTangent = normalize(vec3(tangent * gl_WorldToObjectEXT));

  Material mat = materials[nonuniformEXT(materialIndex)].m;
  TextureIndex ti = textureIndices.t[materialIndex];

  vec3 baseColor = mat.baseColor.rgb;
  if (ti.diffuse >= 0) {
    baseColor = texture(textures[nonuniformEXT(ti.diffuse)], uv).rgb;
  }
  baseColor /= M_PI;

  float metallic = mat.metallic;
  if (ti.metallic >= 0) {
    metallic = texture(textures[nonuniformEXT(ti.metallic)], uv).x;
  }
  metallic = clamp(metallic, 0.0, 1.0);

  float roughness = mat.roughness;
  if (ti.roughness >= 0) {
    roughness = texture(textures[nonuniformEXT(ti.roughness)], uv).x;
  }
  float a2 = roughness * roughness;

  vec3 texNorm = vec3(0.0, 0.0, 1.0);  // TODO use
  if (ti.normal >= 0) {
    texNorm = texture(textures[nonuniformEXT(ti.normal)], uv).xyz;
  }

  vec3 emission = mat.emission.rgb;
  if (ti.emission >= 0) {
    emission = texture(textures[nonuniformEXT(ti.emission)], uv).rgb;
  }

  float transmission = clamp(mat.transmission, 0.0, 1.0);

  float specular = clamp(mat.fresnel * 0.08, 0.0, 1.0);

  float nonMetallic = 1.0 - metallic;

  vec3 diffuseColor = (1.0 - transmission) * baseColor * nonMetallic;
  vec3 specularColor = baseColor * metallic + vec3(specular) * nonMetallic;
  vec3 transmissionColor = transmission * baseColor * nonMetallic;

  float diffuseProb = maxVec3(diffuseColor);
  float specularProb = maxVec3(specularColor);
  float transmissionProb = maxVec3(transmissionColor);
  float total = diffuseProb / (diffuseProb + specularProb + transmissionProb);

  diffuseProb = diffuseProb / total;
  specularProb = specularProb / total;
  transmissionProb = 1.0 - diffuseProb - specularProb;

  float rand = rnd(ray.seed);

  vec3 V = normalize(-ray.direction);

  bool isInside = gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT;
  // vec3 N = isInside ? -worldShadingNormal : worldShadingNormal;

  mat3 tbn = mat3(worldTangent, cross(worldShadingNormal, worldTangent), worldShadingNormal);

  if (isInside) {
    tbn *= -1.0;
  }

  vec3 attenuation = vec3(0.0);
  vec3 L = vec3(0.0);
  if (rand <= diffuseProb) {
    sampleDiffuse(ray.seed, tbn, diffuseColor, L, attenuation);
    M_PI * diffuseColor / diffuseProb;
  } else if (rand <= diffuseProb + specularProb) {  // specular
    sampleGGX(ray.seed, tbn, specularColor, worldShadingNormal, V, roughness, L, attenuation);
  } else {  // transmission
    // TODO
  }

  ray.radiance = emission
    + traceDirectionalLights(worldPosition, worldShadingNormal, diffuseColor, specularColor, roughness)
    + tracePointLights(worldPosition, worldShadingNormal, diffuseColor, specularColor, roughness);

  ray.origin = worldPosition;
  ray.direction = L;
  ray.attenuation = attenuation;
  ray.normal = shadingNormal;  // TODO transform normal to camera space
  ray.albedo = baseColor;
}
