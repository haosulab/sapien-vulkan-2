#include "svulkan2/resource/model.h"
#include "svulkan2/common/image.h"
#include "svulkan2/common/log.h"
#include "svulkan2/resource/manager.h"
#include <assimp/Importer.hpp>
#include <assimp/pbrmaterial.h>
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

std::vector<std::shared_ptr<SVShape>> const &SVModel::getShapes() {
  loadAsync().get();
  return mShapes;
}

static std::vector<uint8_t> loadCompressedTexture(aiTexture const *texture,
                                                  int &width, int &height) {
  return loadImageFromMemory(reinterpret_cast<unsigned char *>(texture->pcData),
                             texture->mWidth, width, height);
}

static std::shared_ptr<SVTexture> loadEmbededTexture(aiTexture const *texture,
                                                     uint32_t mipLevels,
                                                     uint32_t channels = 4,
                                                     bool srgb = false) {
  if (channels != 1 && channels != 4) {
    throw std::runtime_error("Texture must contain 1 or 4 channels");
  }
  std::vector<uint8_t> data;
  if (texture->mHeight == 0) {
    int width, height;
    auto loaded = loadCompressedTexture(texture, width, height);
    if (channels == 1) {
      data.reserve(loaded.size() / 4);
      for (uint32_t i = 0; i < loaded.size() / 4; ++i) {
        data.push_back(loaded[4 * i]);
      }
    } else {
      data = loaded;
    }
    return SVTexture::FromData(width, height, channels, data, mipLevels,
                               vk::Filter::eLinear, vk::Filter::eLinear,
                               vk::SamplerAddressMode::eRepeat,
                               vk::SamplerAddressMode::eRepeat, srgb);
  }
  data = std::vector<uint8_t>(reinterpret_cast<uint8_t *>(texture->pcData),
                              reinterpret_cast<uint8_t *>(texture->pcData) +
                                  texture->mWidth * texture->mHeight * 4);
  return SVTexture::FromData(texture->mWidth, texture->mHeight, 4, data,
                             mipLevels);
}

static std::tuple<std::shared_ptr<SVTexture>, std::shared_ptr<SVTexture>>
loadEmbededRoughnessMetallicTexture(aiTexture const *texture,
                                    uint32_t mipLevels) {
  std::vector<uint8_t> roughness;
  std::vector<uint8_t> metallic;
  if (texture->mHeight != 0) {
    std::runtime_error("Invalid roughness metallic texture");
  }
  int width, height;
  auto loaded = loadCompressedTexture(texture, width, height);
  roughness.reserve(loaded.size() / 4);
  metallic.reserve(loaded.size() / 4);
  for (uint32_t i = 0; i < loaded.size() / 4; ++i) {
    roughness.push_back(loaded[4 * i + 1]);
    metallic.push_back(loaded[4 * i + 2]);
  }
  return {
      SVTexture::FromData(width, height, 1, roughness, mipLevels),
      SVTexture::FromData(width, height, 1, metallic, mipLevels),
  };
}

std::future<void> SVModel::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }
  if (mDescription.source != ModelDescription::SourceType::eFILE) {
    throw std::runtime_error(
        "loading failed: only mesh created from files can be loaded");
  }
  if (!mManager) {
    throw std::runtime_error(
        "loading failed: resource manager is required for model loading.");
  }

  return std::async(std::launch::async, [this]() {
    std::lock_guard<std::mutex> lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }
    log::info("Loading: {}", mDescription.filename);

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

    std::vector<std::future<void>> futures; // futures of sub tasks

    const uint32_t MIP_LEVEL = mManager->getDefaultMipLevels();
    std::vector<std::shared_ptr<SVMaterial>> materials;
    std::unordered_map<std::string, std::shared_ptr<SVTexture>> textureCache;
    std::unordered_map<std::string, std::tuple<std::shared_ptr<SVTexture>,
                                               std::shared_ptr<SVTexture>>>
        roughnessMetallicTextureCache;

    for (uint32_t mat_idx = 0; mat_idx < scene->mNumMaterials; ++mat_idx) {
      auto *m = scene->mMaterials[mat_idx];
      aiColor3D emission{0, 0, 0};
      aiColor3D diffuse{0, 0, 0};
      aiColor3D specular{0, 0, 0};
      float alpha = 1.f;
      float shininess = 0.f;
      m->Get(AI_MATKEY_OPACITY, alpha);
      if (alpha < 1e-5 && (mDescription.filename.ends_with(".dae") ||
                           mDescription.filename.ends_with(".DAE"))) {
        log::warn("The DAE file {} is fully transparent. This is probably "
                  "due to modeling error. Setting opacity to 1 instead...",
                  mDescription.filename);
        alpha = 1.f;
      }
      m->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
      m->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
      m->Get(AI_MATKEY_COLOR_SPECULAR, specular);
      m->Get(AI_MATKEY_SHININESS, shininess);

      std::shared_ptr<SVTexture> baseColorTexture{};
      std::shared_ptr<SVTexture> normalTexture{};
      std::shared_ptr<SVTexture> roughnessTexture{};
      std::shared_ptr<SVTexture> metallicTexture{};
      std::shared_ptr<SVTexture> emissionTexture{};

      aiString path;
      if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0 &&
          m->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
        log::info("Trying to load texture {}", path.C_Str());
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            baseColorTexture = textureCache[std::string(path.C_Str())];
          } else {
            log::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = baseColorTexture =
                loadEmbededTexture(texture, MIP_LEVEL, 4, true);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          baseColorTexture = mManager->CreateTextureFromFile(
              fullPath, MIP_LEVEL, vk::Filter::eLinear, vk::Filter::eLinear,
              vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
              true);
          futures.push_back(baseColorTexture->loadAsync());
        }
      }
      if (m->GetTextureCount(aiTextureType_METALNESS) > 0 &&
          m->GetTexture(aiTextureType_METALNESS, 0, &path) == AI_SUCCESS) {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            metallicTexture = textureCache[std::string(path.C_Str())];
          } else {
            log::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = metallicTexture =
                loadEmbededTexture(texture, MIP_LEVEL);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          metallicTexture =
              mManager->CreateTextureFromFile(fullPath, MIP_LEVEL);
          futures.push_back(metallicTexture->loadAsync());
        }
      }

      if (m->GetTextureCount(aiTextureType_NORMALS) > 0 &&
          m->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS) {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            normalTexture = textureCache[std::string(path.C_Str())];
          } else {
            log::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = normalTexture =
                loadEmbededTexture(texture, MIP_LEVEL);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          normalTexture = mManager->CreateTextureFromFile(fullPath, MIP_LEVEL);
          futures.push_back(normalTexture->loadAsync());
        }
      }

      if (m->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0 &&
          m->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &path) ==
              AI_SUCCESS) {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (textureCache.contains(std::string(path.C_Str()))) {
            roughnessTexture = textureCache[std::string(path.C_Str())];
          } else {
            log::info("Loading embeded texture {}", path.C_Str());
            textureCache[std::string(path.C_Str())] = roughnessTexture =
                loadEmbededTexture(texture, MIP_LEVEL);
          }
        } else {
          std::string p = std::string(path.C_Str());
          std::string fullPath = (parentDir / p).string();
          roughnessTexture =
              mManager->CreateTextureFromFile(fullPath, MIP_LEVEL);
          futures.push_back(roughnessTexture->loadAsync());
        }
      }

      if (m->GetTexture(
              AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
              &path) == AI_SUCCESS) {
        if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
          if (roughnessMetallicTextureCache.contains(
                  std::string(path.C_Str()))) {
            std::tie(roughnessTexture, metallicTexture) =
                roughnessMetallicTextureCache[std::string(path.C_Str())];
          } else {
            log::info("Loading embeded roughness metallic texture {}",
                      path.C_Str());
            std::tie(roughnessTexture, metallicTexture) =
                roughnessMetallicTextureCache[std::string(path.C_Str())] =
                    loadEmbededRoughnessMetallicTexture(texture, MIP_LEVEL);
          }
        } else {
          log::warn("Loading non-embeded roughness metallic texture is "
                    "currently not supported");
        }
      }

      auto material = std::make_shared<SVMetallicMaterial>(
          glm::vec4{emission.r, emission.g, emission.b, 1},
          glm::vec4{diffuse.r, diffuse.g, diffuse.b, alpha},
          (specular.r + specular.g + specular.b) / 3,
          shininessToRoughness(shininess), 0, 0);
      material->setTextures(baseColorTexture, roughnessTexture, normalTexture,
                            metallicTexture, emissionTexture);
      materials.push_back(material);
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
        glm::vec3 position = glm::vec3(
            mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z);
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
      auto layout = mManager->getVertexLayout();
      auto svmesh = std::make_shared<SVMesh>();
      svmesh->setIndices(indices);
      svmesh->setVertexAttribute("position", positions);
      if (layout->elements.find("normal") != layout->elements.end()) {
        svmesh->setVertexAttribute("normal", normals);
      }
      if (layout->elements.find("tangent") != layout->elements.end()) {
        svmesh->setVertexAttribute("tangent", tangents);
      }
      if (layout->elements.find("bitangent") != layout->elements.end()) {
        svmesh->setVertexAttribute("bitangent", bitangents);
      }
      if (layout->elements.find("uv") != layout->elements.end()) {
        svmesh->setVertexAttribute("uv", texcoords);
      }
      if (layout->elements.find("color") != layout->elements.end()) {
        svmesh->setVertexAttribute("color", colors);
      }

      auto shape = std::make_shared<SVShape>();
      shape->mesh = svmesh;
      shape->material = materials[mesh->mMaterialIndex];
      mShapes.push_back(shape);
    }
    for (auto &f : futures) {
      f.get();
    }
    mLoaded = true;
    log::info("Loaded: {}", mDescription.filename);
  });
}

} // namespace resource
} // namespace svulkan2
