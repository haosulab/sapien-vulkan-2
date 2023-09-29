#include "svulkan2/resource/model.h"
#include "../common/logger.h"
#include "svulkan2/common/image.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/core/context.h"
#include "svulkan2/resource/manager.h"
#include <assimp/GltfMaterial.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVModel> SVModel::FromPrototype(std::shared_ptr<SVModel> prototype) {
  auto model = std::shared_ptr<SVModel>(new SVModel);
  model->mPrototype = prototype;
  model->mDescription = prototype->mDescription;
  return model;
}

std::shared_ptr<SVModel> SVModel::FromFile(std::string const &filename) {
  auto model = std::shared_ptr<SVModel>(new SVModel);
  model->mDescription = {.source = ModelDescription::SourceType::eFILE, .filename = filename};
  return model;
}

std::shared_ptr<SVModel> SVModel::FromData(std::vector<std::shared_ptr<SVShape>> shapes) {
  auto model = std::shared_ptr<SVModel>(new SVModel);
  model->mDescription = {.source = ModelDescription::SourceType::eCUSTOM, .filename = {}};
  model->mShapes = shapes;
  model->mLoaded = true;
  return model;
}

std::vector<std::shared_ptr<SVShape>> const &SVModel::getShapes() {
  loadAsync().get();
  return mShapes;
}

static std::vector<uint8_t> loadCompressedTexture(aiTexture const *texture, int &width,
                                                  int &height, int &channels) {
  return loadImageFromMemory(reinterpret_cast<unsigned char *>(texture->pcData), texture->mWidth,
                             width, height, channels);
}

static std::shared_ptr<SVTexture> loadEmbededTexture(aiTexture const *texture, uint32_t mipLevels,
                                                     int desiredChannels = 1, bool srgb = false) {
  // TODO: check desired channels against channels

  if (texture->mHeight == 0) {
    int width, height, channels;
    std::vector<uint8_t> data = loadCompressedTexture(texture, width, height, channels);

    vk::Format format;
    switch (channels) {
    case 1:
      format = vk::Format::eR8Unorm;
      break;
    case 2:
      format = vk::Format::eR8G8Unorm;
      break;
    case 3:
      format = vk::Format::eR8G8B8Unorm;
      break;
    case 4:
      format = vk::Format::eR8G8B8A8Unorm;
      break;
    default:
      throw std::runtime_error("invalid image channels");
    }

    return SVTexture::FromRawData(width, height, 1, format, toRawBytes(data), 2, mipLevels,
                                  vk::Filter::eLinear, vk::Filter::eLinear,
                                  vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
                                  vk::SamplerAddressMode::eRepeat, srgb);
  }

  if (strcmp(texture->achFormatHint, "rgba8888") != 0) {
    throw std::runtime_error("unsupported texture: only rgba8888 format is supported");
  }
  std::vector<char> rawData(reinterpret_cast<char *>(texture->pcData),
                            reinterpret_cast<char *>(texture->pcData) +
                                texture->mWidth * texture->mHeight * 4);

  return SVTexture::FromRawData(texture->mWidth, texture->mHeight, 1, vk::Format::eR8G8B8A8Unorm,
                                rawData, 2, mipLevels, vk::Filter::eLinear, vk::Filter::eLinear,
                                vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
                                vk::SamplerAddressMode::eRepeat, srgb);
}

static std::tuple<std::shared_ptr<SVTexture>, std::shared_ptr<SVTexture>>
loadEmbededRoughnessMetallicTexture(aiTexture const *texture, uint32_t mipLevels) {
  std::vector<uint8_t> roughness;
  std::vector<uint8_t> metallic;
  if (texture->mHeight != 0) {
    std::runtime_error("Invalid roughness metallic texture");
  }
  int width, height, channels;
  auto loaded = loadCompressedTexture(texture, width, height, channels);
  // TODO: check loaded channels
  roughness.reserve(loaded.size() / channels);
  metallic.reserve(loaded.size() / channels);
  for (uint32_t i = 0; i < loaded.size() / channels; ++i) {
    roughness.push_back(loaded[channels * i + 1]);
    metallic.push_back(loaded[channels * i + 2]);
  }

  return {
      SVTexture::FromRawData(width, height, 1, vk::Format::eR8Unorm, toRawBytes(roughness), 2,
                             mipLevels),
      SVTexture::FromRawData(width, height, 1, vk::Format::eR8Unorm, toRawBytes(metallic), 2,
                             mipLevels),
  };
}

std::future<void> SVModel::loadAsync() {
  if (mLoaded) {
    return std::async(std::launch::deferred, []() {});
  }
  if (mDescription.source != ModelDescription::SourceType::eFILE) {
    throw std::runtime_error("loading failed: only mesh created from files can be loaded");
  }
  auto context = core::Context::Get();
  auto manager = context->getResourceManager();

  return std::async(LAUNCH_ASYNC, [this, manager]() {
    std::lock_guard<std::mutex> lock(mLoadingMutex);
    if (mLoaded) {
      return;
    }

    if (mPrototype) {
      auto shapes = mPrototype->getShapes();
      for (auto s : shapes) {
        auto newShape = std::make_shared<SVShape>();

        auto mat = std::dynamic_pointer_cast<SVMetallicMaterial>(s->material);
        auto newMat = std::make_shared<SVMetallicMaterial>(
            mat->getEmission(), mat->getBaseColor(), mat->getFresnel(), mat->getRoughness(),
            mat->getMetallic(), mat->getTransmission(), mat->getIor());
        newMat->setTextures(mat->getDiffuseTexture(), mat->getRoughnessTexture(),
                            mat->getNormalTexture(), mat->getMetallicTexture(),
                            mat->getEmissionTexture(), mat->getTransmissionTexture());

        newShape->mesh = s->mesh;
        newShape->material = newMat;
        mShapes.push_back(newShape);
      }
      mLoaded = true;
      return;
    }

    logger::info("Loading: {}", mDescription.filename);

    std::string path = mDescription.filename;
    Assimp::Importer importer;
    uint32_t flags = aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_GenNormals |
                     aiProcess_FlipUVs | aiProcess_PreTransformVertices;

    importer.SetPropertyBool(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, true);

    const aiScene *scene = importer.ReadFile(path, flags);

    if (scene->mRootNode->mMetaData) {
      throw std::runtime_error("Failed to load mesh file: file contains unsupported metadata, " +
                               path);
    }
    if (!scene) {
      throw std::runtime_error("Failed to load scene: " + std::string(importer.GetErrorString()) +
                               ", " + path);
    }

    fs::path parentDir = fs::path(path).parent_path();

    std::vector<std::future<void>> futures; // futures of sub tasks

    const uint32_t MIP_LEVEL = manager->getDefaultMipLevels();
    std::vector<std::shared_ptr<SVMaterial>> materials;
    std::unordered_map<std::string, std::shared_ptr<SVTexture>> textureCache;
    std::unordered_map<std::string,
                       std::tuple<std::shared_ptr<SVTexture>, std::shared_ptr<SVTexture>>>
        roughnessMetallicTextureCache;

    for (uint32_t mat_idx = 0; mat_idx < scene->mNumMaterials; ++mat_idx) {
      auto *m = scene->mMaterials[mat_idx];
      aiColor3D emission{0, 0, 0};
      aiColor3D diffuse{0, 0, 0};
      float glossiness = 0.5f;

      float alpha = 1.f;
      float metallic = 0.f;
      float transmission = 0.f;
      float ior = 1.01f;

      float roughness = 1.f;

      if (m->Get(AI_MATKEY_OPACITY, alpha) == AI_SUCCESS) {
        if (alpha < 1e-5) {
          logger::warn("The file {} has a fully transparent material. This is "
                       "probably due to modeling error. Setting opacity to 1. If "
                       "it is not an error, please remove the object entirely.",
                       mDescription.filename);
          alpha = 1.f;
        }
      } else {
        if (m->Get(AI_MATKEY_TRANSPARENCYFACTOR, alpha) == AI_SUCCESS) {
          alpha = 1 - alpha;
        }
      }

      m->Get(AI_MATKEY_COLOR_EMISSIVE, emission);
      m->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);

      {
        aiColor4D baseColor = {0, 0, 0, 1};
        if (m->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS) {
          diffuse = {baseColor.r, baseColor.g, baseColor.b};
          alpha = baseColor.a;
        }
      }

      // CHECK GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR

      // assimp code for reading roughness, metallic, and glossiness
      if (m->Get(AI_MATKEY_METALLIC_FACTOR, metallic) != AI_SUCCESS) {
        metallic = 0.f;
      }
      if (m->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) != AI_SUCCESS) {
        aiColor4D specularColor;
        ai_real shininess;
        if (m->Get(AI_MATKEY_COLOR_SPECULAR, specularColor) == AI_SUCCESS &&
            m->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
          float specularIntensity =
              specularColor[0] * 0.2125f + specularColor[1] * 0.7154f + specularColor[2] * 0.0721f;
          float normalizedShininess = std::sqrt(shininess / 1000);
          normalizedShininess = std::min(std::max(normalizedShininess, 0.0f), 1.0f);
          normalizedShininess = normalizedShininess * specularIntensity;
          roughness = 1 - normalizedShininess;
        }
      }

      if (m->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission) != AI_SUCCESS) {
        transmission = 0.f;
      }
      if (m->Get(AI_MATKEY_REFRACTI, ior) != AI_SUCCESS) {
        ior = 1.01f;
      }

      if (m->Get(AI_MATKEY_GLOSSINESS_FACTOR, glossiness) != AI_SUCCESS) {
        float shininess;
        if (m->Get(AI_MATKEY_SHININESS, shininess)) {
          glossiness = shininess / 1000;
        }
      }

      std::shared_ptr<SVTexture> baseColorTexture{};
      std::shared_ptr<SVTexture> normalTexture{};
      std::shared_ptr<SVTexture> roughnessTexture{};
      std::shared_ptr<SVTexture> metallicTexture{};
      std::shared_ptr<SVTexture> emissionTexture{};
      std::shared_ptr<SVTexture> transmissionTexture{};

      aiString path;
      if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0 &&
          m->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
        if (core::Context::Get()->shouldNotLoadTexture()) {
          logger::info("Texture ignored {}", path.C_Str());
        } else {
          logger::info("Trying to load texture {}", path.C_Str());
          if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
            if (textureCache.contains(std::string(path.C_Str()))) {
              baseColorTexture = textureCache[std::string(path.C_Str())];
            } else {
              logger::info("Loading embeded texture {}", path.C_Str());
              textureCache[std::string(path.C_Str())] = baseColorTexture =
                  loadEmbededTexture(texture, MIP_LEVEL, 4, true);
            }
          } else {
            std::string p = std::string(path.C_Str());
            std::string fullPath = (parentDir / p).string();
            baseColorTexture = manager->CreateTextureFromFile(
                fullPath, MIP_LEVEL, vk::Filter::eLinear, vk::Filter::eLinear,
                vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, true);
            futures.push_back(baseColorTexture->loadAsync());
          }
        }
      }

      if (m->GetTextureCount(aiTextureType_METALNESS) > 0 &&
          m->GetTexture(aiTextureType_METALNESS, 0, &path) == AI_SUCCESS) {
        if (core::Context::Get()->shouldNotLoadTexture()) {
          logger::info("Texture ignored {}", path.C_Str());
        } else {
          if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
            if (textureCache.contains(std::string(path.C_Str()))) {
              metallicTexture = textureCache[std::string(path.C_Str())];
            } else {
              logger::info("Loading embeded texture {}", path.C_Str());
              textureCache[std::string(path.C_Str())] = metallicTexture =
                  loadEmbededTexture(texture, MIP_LEVEL);
            }
          } else {
            std::string p = std::string(path.C_Str());
            std::string fullPath = (parentDir / p).string();
            metallicTexture = manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
            futures.push_back(metallicTexture->loadAsync());
          }
        }
      }

      if (m->GetTextureCount(aiTextureType_NORMALS) > 0 &&
          m->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS) {
        if (core::Context::Get()->shouldNotLoadTexture()) {
          logger::info("Texture ignored {}", path.C_Str());
        } else {
          if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
            if (textureCache.contains(std::string(path.C_Str()))) {
              normalTexture = textureCache[std::string(path.C_Str())];
            } else {
              logger::info("Loading embeded texture {}", path.C_Str());
              textureCache[std::string(path.C_Str())] = normalTexture =
                  loadEmbededTexture(texture, MIP_LEVEL);
            }
          } else {
            std::string p = std::string(path.C_Str());
            std::string fullPath = (parentDir / p).string();
            normalTexture = manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
            futures.push_back(normalTexture->loadAsync());
          }
        }
      }

      if (m->GetTextureCount(aiTextureType_EMISSIVE) > 0 &&
          m->GetTexture(aiTextureType_EMISSIVE, 0, &path) == AI_SUCCESS) {
        if (core::Context::Get()->shouldNotLoadTexture()) {
          logger::info("Texture ignored {}", path.C_Str());
        } else {
          if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
            if (textureCache.contains(std::string(path.C_Str()))) {
              emissionTexture = textureCache[std::string(path.C_Str())];
            } else {
              logger::info("Loading embeded texture {}", path.C_Str());
              textureCache[std::string(path.C_Str())] = emissionTexture =
                  loadEmbededTexture(texture, MIP_LEVEL, 4, true);
            }
          } else {
            std::string p = std::string(path.C_Str());
            std::string fullPath = (parentDir / p).string();
            emissionTexture = manager->CreateTextureFromFile(
                fullPath, MIP_LEVEL, vk::Filter::eLinear, vk::Filter::eLinear,
                vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, true);
            futures.push_back(emissionTexture->loadAsync());
          }
        }
      }

      if (m->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0 &&
          m->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &path) == AI_SUCCESS) {
        if (core::Context::Get()->shouldNotLoadTexture()) {
          logger::info("Texture ignored {}", path.C_Str());
        } else {
          if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
            if (textureCache.contains(std::string(path.C_Str()))) {
              roughnessTexture = textureCache[std::string(path.C_Str())];
            } else {
              logger::info("Loading embeded texture {}", path.C_Str());
              textureCache[std::string(path.C_Str())] = roughnessTexture =
                  loadEmbededTexture(texture, MIP_LEVEL);
            }
          } else {
            std::string p = std::string(path.C_Str());
            std::string fullPath = (parentDir / p).string();
            roughnessTexture = manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
            futures.push_back(roughnessTexture->loadAsync());
          }
        }
      }

      if (m->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &path) ==
          AI_SUCCESS) {
        if (core::Context::Get()->shouldNotLoadTexture()) {
          logger::info("Texture ignored {}", path.C_Str());
        } else {
          if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
            if (roughnessMetallicTextureCache.contains(std::string(path.C_Str()))) {
              std::tie(roughnessTexture, metallicTexture) =
                  roughnessMetallicTextureCache[std::string(path.C_Str())];
            } else {
              logger::info("Loading embeded roughness metallic texture {}", path.C_Str());
              std::tie(roughnessTexture, metallicTexture) =
                  roughnessMetallicTextureCache[std::string(path.C_Str())] =
                      loadEmbededRoughnessMetallicTexture(texture, MIP_LEVEL);
            }
          } else {
            logger::warn("Loading non-embeded roughness metallic texture is "
                         "currently not supported");
          }
        }
      }

      if (m->GetTexture(AI_MATKEY_TRANSMISSION_TEXTURE, &path) == AI_SUCCESS) {
        if (core::Context::Get()->shouldNotLoadTexture()) {
          logger::info("Texture ignored {}", path.C_Str());
        } else {
          if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
            if (textureCache.contains(std::string(path.C_Str()))) {
              transmissionTexture = textureCache[std::string(path.C_Str())];
            } else {
              logger::info("Loading embeded texture {}", path.C_Str());
              textureCache[std::string(path.C_Str())] = transmissionTexture =
                  loadEmbededTexture(texture, MIP_LEVEL);
            }
          } else {
            std::string p = std::string(path.C_Str());
            std::string fullPath = (parentDir / p).string();
            transmissionTexture = manager->CreateTextureFromFile(fullPath, MIP_LEVEL);
            futures.push_back(transmissionTexture->loadAsync());
          }
        }
      }

      auto material =
          std::make_shared<SVMetallicMaterial>(glm::vec4{emission.r, emission.g, emission.b, 1},
                                               glm::vec4{diffuse.r, diffuse.g, diffuse.b, alpha},
                                               glossiness, roughness, metallic, transmission, ior);
      material->setTextures(baseColorTexture, roughnessTexture, normalTexture, metallicTexture,
                            emissionTexture, transmissionTexture);
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
        glm::vec3 position =
            glm::vec3(mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z);
        glm::vec3 tangent = glm::vec3(0);
        glm::vec3 bitangent = glm::vec3(0);
        if (mesh->HasNormals()) {
          normal = glm::vec3{mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z};
        }
        if (mesh->HasTextureCoords(0)) {
          texcoord = glm::vec2{mesh->mTextureCoords[0][v].x, mesh->mTextureCoords[0][v].y};
        }
        if (mesh->HasTangentsAndBitangents()) {
          tangent = glm::vec3{mesh->mTangents[v].x, mesh->mTangents[v].y, mesh->mTangents[v].z};
          bitangent =
              glm::vec3{mesh->mBitangents[v].x, mesh->mBitangents[v].y, mesh->mBitangents[v].z};
        }
        if (mesh->HasVertexColors(0)) {
          color = glm::vec4{mesh->mColors[0][v].r, mesh->mColors[0][v].g, mesh->mColors[0][v].b,
                            mesh->mColors[0][v].a};
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
        logger::warn("A mesh in the file has no triangles: {}", path);
        continue;
      }
      auto svmesh = std::make_shared<SVMeshRigid>();
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
    for (auto &f : futures) {
      f.get();
    }
    mLoaded = true;
    logger::info("Loaded: {}", mDescription.filename);
  });
}

void SVModel::buildBLAS(bool update) {
  std::vector<vk::AccelerationStructureGeometryKHR> geometries;
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
  std::vector<uint32_t> maxPrimitiveCount;
  for (auto shape : getShapes()) {
    geometries.push_back(shape->mesh->getASGeometry());
    ranges.push_back({shape->mesh->getTriangleCount(), 0, 0, 0});
    maxPrimitiveCount.push_back(shape->mesh->getTriangleCount());
  }

  mBLAS = std::make_unique<core::BLAS>(geometries, ranges, maxPrimitiveCount, !update, update);
  mBLAS->build();
}

void SVModel::recordUpdateBLAS(vk::CommandBuffer commandBuffer) {
  std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
  for (auto shape : getShapes()) {
    ranges.push_back({shape->mesh->getTriangleCount(), 0, 0, 0});
  }
  mBLAS->recordUpdate(commandBuffer, ranges);
}

core::BLAS *SVModel::getBLAS() { return mBLAS.get(); }

SVModel::SVModel() {}

SVModel::~SVModel() {}

} // namespace resource
} // namespace svulkan2
