#include "../common/logger.h"
#include "svulkan2/common/image.h"
#include "svulkan2/core/context.h"
#include "svulkan2/scene/scene.h"

#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace svulkan2 {
namespace scene {

std::shared_ptr<Scene> LoadScene(std::string const &path) {
  auto scene = std::make_shared<Scene>();
  auto model = resource::SVModel::FromFile(path);
  scene->addObject(model);

  Assimp::Importer importer;
  uint32_t flags = aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_GenNormals |
                   aiProcess_FlipUVs | aiProcess_PreTransformVertices;
  importer.SetPropertyBool(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, true);
  const aiScene *s = importer.ReadFile(path, flags);

  if (!s) {
    throw std::runtime_error("failed to parse scene " + path);
  }

  for (uint32_t i = 0; i < s->mNumLights; ++i) {
    switch (s->mLights[i]->mType) {
    case aiLightSource_POINT: {
      auto &l = scene->addPointLight();
      l.setColor(glm::vec3{s->mLights[i]->mColorDiffuse.r, s->mLights[i]->mColorDiffuse.g,
                           s->mLights[i]->mColorDiffuse.b} /
                 5.f);
      l.setPosition(
          {s->mLights[i]->mPosition.x, s->mLights[i]->mPosition.y, s->mLights[i]->mPosition.z});
      break;
    }
    case aiLightSource_DIRECTIONAL: {
      auto &l = scene->addDirectionalLight();
      l.setColor(glm::vec3{s->mLights[i]->mColorDiffuse.r, s->mLights[i]->mColorDiffuse.g,
                           s->mLights[i]->mColorDiffuse.b} /
                 5.f);
      l.setPosition(
          {s->mLights[i]->mPosition.x, s->mLights[i]->mPosition.y, s->mLights[i]->mPosition.z});
      l.setDirection(
          {s->mLights[i]->mDirection.x, s->mLights[i]->mDirection.y, s->mLights[i]->mDirection.z});
      break;
    }
    case aiLightSource_SPOT: {
      auto &l = scene->addSpotLight();
      l.setColor(glm::vec3{s->mLights[i]->mColorDiffuse.r, s->mLights[i]->mColorDiffuse.g,
                           s->mLights[i]->mColorDiffuse.b} /
                 5.f);
      l.setPosition(
          {s->mLights[i]->mPosition.x, s->mLights[i]->mPosition.y, s->mLights[i]->mPosition.z});
      l.setDirection(
          {s->mLights[i]->mDirection.x, s->mLights[i]->mDirection.y, s->mLights[i]->mDirection.z});
      l.setFovSmall(s->mLights[i]->mAngleInnerCone);
      l.setFov(s->mLights[i]->mAngleOuterCone);
      break;
    }
    case aiLightSource_AREA: {
      auto &l = scene->addParallelogramLight();
      l.setColor(glm::vec3{s->mLights[i]->mColorDiffuse.r, s->mLights[i]->mColorDiffuse.g,
                           s->mLights[i]->mColorDiffuse.b} /
                 5.f);
      l.setShape({s->mLights[i]->mSize.x, s->mLights[i]->mSize.y});

      l.setRotation(glm::quatLookAt(
          glm::vec3{s->mLights[i]->mDirection.x, s->mLights[i]->mDirection.y,
                    s->mLights[i]->mDirection.z},
          glm::vec3{s->mLights[i]->mUp.x, s->mLights[i]->mUp.y, s->mLights[i]->mUp.z}));
      l.setPosition(
          {s->mLights[i]->mPosition.x, s->mLights[i]->mPosition.y, s->mLights[i]->mPosition.z});
      break;
    }
    case aiLightSource_AMBIENT: {
      scene->setAmbientLight({s->mLights[i]->mColorAmbient.r, s->mLights[i]->mColorAmbient.g,
                              s->mLights[i]->mColorAmbient.b, 1.f});
      break;
    }
    default:
      logger::warn("failed to load light: cannot recognized light type");
      break;
    }
  }
  return scene;
};

} // namespace scene
} // namespace svulkan2
