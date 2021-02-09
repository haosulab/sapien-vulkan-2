#include "svulkan2/resource/model.h"
#include "svulkan2/common/log.h"
#include "svulkan2/resource/manager.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace resource {

static float shininessToRoughness(float ns) {
  if (ns <= 5.f) {
    return 1.f;
  }
  if (ns >= 1605.f) {
    return 0.f;
  }
  return 1.f - (std::sqrt(ns - 5.f) * 0.025f);
}

std::shared_ptr<SVModel> SVModel::FromFile(std::string const &filename) {
  auto model = std::shared_ptr<SVModel>(new SVModel);
  model->mDescription = {.source = ModelDescription::SourceType::eFILE,
                         .filename = filename};
  return model;
}

std::shared_ptr<SVModel>
SVModel::FromData(std::vector<std::shared_ptr<SVShape>> shapes) {
  auto model = std::shared_ptr<SVModel>(new SVModel);
  model->mDescription = {.source = ModelDescription::SourceType::eCUSTOM,
                         .filename = {}};
  model->mShapes = shapes;
  model->mLoaded = true;
  return model;
}

void SVModel::load() {
  if (isLoaded()) {
    log::warn("skip loading: model already loaded");
  }
  if (mDescription.source != ModelDescription::SourceType::eFILE) {
    throw std::runtime_error(
        "loading failed: only mesh created from files can be loaded");
  }
  if (!mManager) {
    throw std::runtime_error(
        "loading failed: resource manager is required for model loading");
  }

  std::string path = mDescription.filename;

  Assimp::Importer importer;
  uint32_t flags = aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                   aiProcess_GenNormals | aiProcess_FlipUVs |
                   aiProcess_PreTransformVertices;
  importer.SetPropertyInteger(AI_CONFIG_PP_PTV_ADD_ROOT_TRANSFORMATION, 1);
  const aiScene *scene = importer.ReadFile(path, flags);

  if (scene->mRootNode->mMetaData) {
    throw std::runtime_error(
        "Failed to load mesh file: file contains unsupported metadata, " +
        path);
  }
  if (!scene) {
    throw std::runtime_error(
        "Failed to load scene: " + std::string(importer.GetErrorString()) +
        ", " + path);
  }

  fs::path parentDir = fs::path(path).parent_path();

  std::vector<std::shared_ptr<SVMaterial>> materials;
  if (mManager->getMaterialPipelineType() ==
      ShaderConfig::MaterialPipeline::eMETALLIC) {
    for (uint32_t mat_idx = 0; mat_idx < scene->mNumMaterials; ++mat_idx) {
      auto *m = scene->mMaterials[mat_idx];
      aiColor3D diffuse{0, 0, 0};
      aiColor3D specular{0, 0, 0};
      float alpha = 1.f;
      float shininess = 0.f;
      m->Get(AI_MATKEY_OPACITY, alpha);
      m->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
      m->Get(AI_MATKEY_COLOR_SPECULAR, specular);
      m->Get(AI_MATKEY_SHININESS, shininess);

      std::shared_ptr<SVTexture> baseColorTexture{};
      std::shared_ptr<SVTexture> normalTexture{};
      std::shared_ptr<SVTexture> roughnessTexture{};
      std::shared_ptr<SVTexture> metallicTexture{};

      aiString path;
      if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0 &&
          m->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
        std::string p = std::string(path.C_Str());
        std::string fullPath = (parentDir / p);
        baseColorTexture = mManager->CreateTextureFromFile(
            fullPath, 1); // TODO configurable mip levels
        baseColorTexture->load();
      }
      if (m->GetTextureCount(aiTextureType_METALNESS) > 0 &&
          m->GetTexture(aiTextureType_METALNESS, 0, &path) == AI_SUCCESS) {
        std::string p = std::string(path.C_Str());
        std::string fullPath = (parentDir / p);
        metallicTexture = mManager->CreateTextureFromFile(fullPath, 1);
        metallicTexture->load();
      }
      if (m->GetTextureCount(aiTextureType_NORMALS) > 0 &&
          m->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS) {
        std::string p = std::string(path.C_Str());
        std::string fullPath = (parentDir / p);
        normalTexture = mManager->CreateTextureFromFile(fullPath, 1);
        normalTexture->load();
      }
      if (m->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0 &&
          m->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS) {
        std::string p = std::string(path.C_Str());
        std::string fullPath = (parentDir / p);
        roughnessTexture = mManager->CreateTextureFromFile(fullPath, 1);
        roughnessTexture->load();
      }

      auto material = std::make_shared<SVMetallicMaterial>(
          glm::vec4{diffuse.r, diffuse.g, diffuse.b, alpha},
          (specular.r + specular.g + specular.b) / 3,
          shininessToRoughness(shininess), 0, 0);
      material->setTextures(baseColorTexture, roughnessTexture, normalTexture,
                            metallicTexture);
      materials.push_back(material);
    }
  } else {
    for (uint32_t mat_idx = 0; mat_idx < scene->mNumMaterials; ++mat_idx) {
      auto *m = scene->mMaterials[mat_idx];
      aiColor3D diffuse{0, 0, 0};
      aiColor3D specular{0, 0, 0};
      float alpha = 1.f;
      float shininess = 0.f;
      m->Get(AI_MATKEY_OPACITY, alpha);
      m->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
      m->Get(AI_MATKEY_COLOR_SPECULAR, specular);
      m->Get(AI_MATKEY_SHININESS, shininess);

      std::shared_ptr<SVTexture> diffuseTexture{};
      std::shared_ptr<SVTexture> specularTexture{};
      std::shared_ptr<SVTexture> normalTexture{};

      aiString path;
      if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0 &&
          m->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
        std::string p = std::string(path.C_Str());
        std::string fullPath = (parentDir / p);
        diffuseTexture = mManager->CreateTextureFromFile(
            fullPath, 1); // TODO configurable mip levels
        diffuseTexture->load();
      }
      if (m->GetTextureCount(aiTextureType_SPECULAR) > 0 &&
          m->GetTexture(aiTextureType_SPECULAR, 0, &path) == AI_SUCCESS) {
        std::string p = std::string(path.C_Str());
        std::string fullPath = (parentDir / p);
        specularTexture = mManager->CreateTextureFromFile(fullPath, 1);
        specularTexture->load();
      }
      if (m->GetTextureCount(aiTextureType_NORMALS) > 0 &&
          m->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS) {
        std::string p = std::string(path.C_Str());
        std::string fullPath = (parentDir / p);
        normalTexture = mManager->CreateTextureFromFile(fullPath, 1);
        normalTexture->load();
      }

      auto material = std::make_shared<SVSpecularMaterial>(
          glm::vec4{diffuse.r, diffuse.g, diffuse.b, alpha},
          glm::vec4{specular.r, specular.g, specular.b, shininess}, 0);
      material->setTextures(diffuseTexture, specularTexture, normalTexture);
      materials.push_back(material);
    }
  }

  for (uint32_t mesh_idx = 0; mesh_idx < scene->mNumMeshes; ++mesh_idx) {
    auto mesh = scene->mMeshes[mesh_idx];
    if (!mesh->HasFaces()) {
      continue;
    }

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texcoords;
    std::vector<float> tangents;
    std::vector<float> bitangents;
    std::vector<float> colors;

    for (uint32_t v = 0; v < mesh->mNumVertices; v++) {
      glm::vec4 color = glm::vec4(1, 1, 1, 1);
      glm::vec3 normal = glm::vec3(0);
      glm::vec2 texcoord = glm::vec2(0);
      glm::vec3 position = glm::vec3(mesh->mVertices[v].x, mesh->mVertices[v].y,
                                     mesh->mVertices[v].z);
      glm::vec3 tangent = glm::vec3(0);
      glm::vec3 bitangent = glm::vec3(0);
      if (mesh->HasNormals()) {
        normal = glm::vec3{mesh->mNormals[v].x, mesh->mNormals[v].y,
                           mesh->mNormals[v].z};
      }
      if (mesh->HasTextureCoords(0)) {
        texcoord = glm::vec2{mesh->mTextureCoords[0][v].x,
                             mesh->mTextureCoords[0][v].y};
      }
      if (mesh->HasTangentsAndBitangents()) {
        tangent = glm::vec3{mesh->mTangents[v].x, mesh->mTangents[v].y,
                            mesh->mTangents[v].z};
        bitangent = glm::vec3{mesh->mBitangents[v].x, mesh->mBitangents[v].y,
                              mesh->mBitangents[v].z};
      }
      if (mesh->HasVertexColors(0)) {
        color = glm::vec4{mesh->mColors[0][v].r, mesh->mColors[0][v].g,
                          mesh->mColors[0][v].b, mesh->mColors[0][v].a};
      }
      positions.push_back(position.x);
      positions.push_back(position.y);
      positions.push_back(position.z);

      normals.push_back(normal.x);
      normals.push_back(normal.y);
      normals.push_back(normal.z);

      texcoords.push_back(texcoord.x);
      texcoords.push_back(texcoord.y);

      tangents.push_back(tangent.x);
      tangents.push_back(tangent.y);
      tangents.push_back(tangent.z);

      bitangents.push_back(bitangent.x);
      bitangents.push_back(bitangent.y);
      bitangents.push_back(bitangent.z);

      colors.push_back(color.r);
      colors.push_back(color.g);
      colors.push_back(color.b);
      colors.push_back(color.a);
    }

    std::vector<uint32_t> indices;
    for (uint32_t f = 0; f < mesh->mNumFaces; f++) {
      auto face = mesh->mFaces[f];
      if (face.mNumIndices != 3) {
        continue;
      }
      indices.push_back(face.mIndices[0]);
      indices.push_back(face.mIndices[1]);
      indices.push_back(face.mIndices[2]);
    }

    if (positions.size() == 0 || indices.size() == 0) {
      log::warn("A mesh in the file has no triangles: {}", path);
      continue;
    }
    auto svmesh = std::make_shared<SVMesh>(mManager->getVertexLayout());
    svmesh->setIndices(indices);
    svmesh->setVertexAttribute("position", positions);
    svmesh->setVertexAttribute("normal", normals);
    svmesh->setVertexAttribute("tangent", tangents);
    svmesh->setVertexAttribute("bitangent", bitangents);
    svmesh->setVertexAttribute("uv", texcoords);
    svmesh->setVertexAttribute("color", colors);

    auto shape = std::make_shared<SVShape>();
    shape->mesh = svmesh;
    shape->material = materials[mesh->mMaterialIndex];
    mShapes.push_back(shape);
  }
}

} // namespace resource
} // namespace svulkan2
