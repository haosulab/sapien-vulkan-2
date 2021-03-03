#include "svulkan2/resource/mesh.h"
#include "svulkan2/common/assimp.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include <memory>

namespace svulkan2 {
namespace resource {

static void strided_memcpy(void *target, void *source, size_t chunk_size,
                           size_t chunks, size_t stride) {
  char *target_ = reinterpret_cast<char *>(target);
  char *source_ = reinterpret_cast<char *>(source);

  for (size_t i = 0; i < chunks; ++i) {
    std::memcpy(target_, source_, chunk_size);
    target_ += stride;
    source_ += chunk_size;
  }
}

SVMesh::SVMesh() {}

void SVMesh::setIndices(std::vector<uint32_t> const &indices) {
  mDirty = true;
  mIndices = indices;
  mIndexCount = indices.size();
}

std::vector<uint32_t> const &SVMesh::getIndices() const { return mIndices; }

void SVMesh::setVertexAttribute(std::string const &name,
                                std::vector<float> const &attrib) {
  mDirty = true;
  mAttributes[name] = attrib;
}

std::vector<float> const &
SVMesh::getVertexAttribute(std::string const &name) const {
  if (!mAttributes.contains(name)) {
    throw std::runtime_error("attribute " + name + " does not exist on vertex");
  }
  return mAttributes.at(name);
}

bool SVMesh::hasVertexAttribute(std::string const &name) const {
  return mAttributes.contains(name);
}

void SVMesh::uploadToDevice(core::Context &context) {
  if (!mDirty) {
    return;
  }

  auto layout = context.getResourceManager().getVertexLayout();

  if (mAttributes.find("position") == mAttributes.end() ||
      mAttributes["position"].size() == 0) {
    throw std::runtime_error("mesh upload failed: no vertex positions");
  }
  if (!mIndices.size()) {
    throw std::runtime_error("mesh upload failed: empty vertex indices");
  }
  if (mAttributes["position"].size() / 3 * 3 !=
      mAttributes["position"].size()) {
    throw std::runtime_error(
        "mesh upload failed: size of vertex positions is not a multiple of 3");
  }
  if (mIndices.size() / 3 * 3 != mIndices.size()) {
    throw std::runtime_error(
        "mesh upload failed: size of vertex indices is not a multiple of 3");
  }

  size_t vertexCount = mAttributes["position"].size() / 3;
  size_t vertexSize = layout->getSize();
  size_t bufferSize = vertexSize * vertexCount;
  size_t indexBufferSize = sizeof(uint32_t) * mIndices.size();

  std::vector<char> buffer(bufferSize, 0);
  auto elements = layout->getElementsSorted();
  uint32_t offset = 0;
  for (auto &elem : elements) {
    if (mAttributes.find(elem.name) != mAttributes.end()) {
      if (mAttributes[elem.name].size() * sizeof(float) !=
          vertexCount * elem.getSize()) {
        throw std::runtime_error("vertex attribute " + elem.name +
                                 " has incorrect size");
      }
      // TODO: test this
      strided_memcpy(buffer.data() + offset, mAttributes[elem.name].data(),
                     elem.getSize(), vertexCount, vertexSize);
    }
    offset += elem.getSize();
  }

  if (!mVertexBuffer) {
    mVertexBuffer = std::make_unique<core::Buffer>(
        context, bufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    mIndexBuffer = std::make_unique<core::Buffer>(
        context, indexBufferSize,
        vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
  }
  mVertexBuffer->upload(buffer.data(), bufferSize);
  mIndexBuffer->upload<uint32_t>(mIndices);
  mOnDevice = true;
  mDirty = false;
}

void SVMesh::removeFromDevice() {
  mDirty = true;
  mOnDevice = false;
  mVertexBuffer.reset();
  mIndexBuffer.reset();
}

void SVMesh::exportToFile(std::string const &filename) const {
  exportTriangleMesh(
      filename,
      mAttributes.contains("position") ? mAttributes.at("position")
                                       : std::vector<float>{},
      mIndices,
      mAttributes.contains("normal") ? mAttributes.at("normal")
                                     : std::vector<float>{},
      mAttributes.contains("uv") ? mAttributes.at("uv") : std::vector<float>{});
}

static std::shared_ptr<SVMesh> makeMesh(std::vector<glm::vec3> const &vertices,
                                        std::vector<glm::ivec3> const &indices,
                                        std::vector<glm::vec3> const &normals,
                                        std::vector<glm::vec2> const &uvs) {
  auto mesh = std::make_shared<SVMesh>();
  std::vector<uint32_t> indices_;
  indices_.reserve(3 * indices.size());
  std::vector<float> vertices_;
  vertices_.reserve(3 * vertices.size());
  std::vector<float> normals_;
  normals_.reserve(3 * normals.size());
  std::vector<float> uvs_;
  uvs_.reserve(2 * uvs.size());
  for (auto &index : indices) {
    indices_.push_back(index.x);
    indices_.push_back(index.y);
    indices_.push_back(index.z);
    assert(static_cast<uint32_t>(index.x) < vertices.size() &&
           static_cast<uint32_t>(index.y) < vertices.size() &&
           static_cast<uint32_t>(index.z) < vertices.size());
  }
  for (auto &vertex : vertices) {
    vertices_.push_back(vertex.x);
    vertices_.push_back(vertex.y);
    vertices_.push_back(vertex.z);
  }
  for (auto &normal : normals) {
    normals_.push_back(normal.x);
    normals_.push_back(normal.y);
    normals_.push_back(normal.z);
  }
  for (auto &uv : uvs) {
    uvs_.push_back(uv.x);
    uvs_.push_back(uv.y);
  }
  mesh->setIndices(indices_);
  mesh->setVertexAttribute("position", vertices_);
  mesh->setVertexAttribute("normal", normals_);
  mesh->setVertexAttribute("uv", uvs_);

  return mesh;
}

std::shared_ptr<SVMesh> SVMesh::CreateUVSphere(int segments, int rings) {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec2> uvs;
  std::vector<glm::ivec3> indices;

  for (int s = 0; s < segments; ++s) {
    vertices.push_back({1.f, 0.f, 0.f});
    uvs.push_back({(0.5f + s) / segments, 1.f});
  }
  for (int r = 1; r < rings; ++r) {
    float theta = glm::pi<float>() * r / rings;
    float x = glm::cos(theta);
    float yz = glm::sin(theta);
    for (int s = 0; s < segments + 1; ++s) {
      float phi = glm::pi<float>() * s * 2 / segments;
      float y = yz * glm::cos(phi);
      float z = yz * glm::sin(phi);
      vertices.push_back({x, y, z});
      uvs.push_back({static_cast<float>(s) / segments,
                     1.f - static_cast<float>(r) / rings});
    }
  }
  for (int s = 0; s < segments; ++s) {
    vertices.push_back({-1.f, 0.f, 0.f});
    uvs.push_back({(0.5f + s) / segments, 0.f});
  }

  for (int s = 0; s < segments; ++s) {
    indices.push_back({s, s + segments, s + segments + 1});
  }

  for (int r = 0; r < rings - 2; ++r) {
    for (int s = 0; s < segments; ++s) {
      indices.push_back({
          segments + (segments + 1) * r + s,
          segments + (segments + 1) * (r + 1) + s,
          segments + (segments + 1) * (r + 1) + s + 1,
      });
      indices.push_back({segments + (segments + 1) * r + s,
                         segments + (segments + 1) * (r + 1) + s + 1,
                         segments + (segments + 1) * r + s + 1});
    }
  }
  for (int s = 0; s < segments; ++s) {
    indices.push_back({segments + (segments + 1) * (rings - 2) + s,
                       segments + (segments + 1) * (rings - 1) + s,
                       segments + (segments + 1) * (rings - 2) + s + 1});
  }

  return makeMesh(vertices, indices, vertices, uvs);
}

std::shared_ptr<SVMesh> SVMesh::CreateCapsule(float radius, float halfLength,
                                              int segments, int halfRings) {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;
  std::vector<glm::ivec3> indices;

  for (int s = 0; s < segments; ++s) {
    vertices.push_back({radius + halfLength, 0.f, 0.f});
    normals.push_back({1.f, 0.f, 0.f});
    uvs.push_back({(0.5f + s) / segments, 1.f});
  }
  int rings = 2 * halfRings;
  for (int r = 1; r <= halfRings; ++r) {
    float theta = glm::pi<float>() * r / rings;
    float x = glm::cos(theta);
    float yz = glm::sin(theta);
    for (int s = 0; s < segments + 1; ++s) {
      float phi = glm::pi<float>() * s * 2 / segments;
      float y = yz * glm::cos(phi);
      float z = yz * glm::sin(phi);
      vertices.push_back(glm::vec3{x, y, z} * radius +
                         glm::vec3{halfLength, 0, 0});
      normals.push_back({x, y, z});
      uvs.push_back({static_cast<float>(s) / segments,
                     1.f - 0.5f * static_cast<float>(r) / rings});
    }
  }
  for (int r = halfRings; r < rings; ++r) {
    float theta = glm::pi<float>() * r / rings;
    float x = glm::cos(theta);
    float yz = glm::sin(theta);
    for (int s = 0; s < segments + 1; ++s) {
      float phi = glm::pi<float>() * s * 2 / segments;
      float y = yz * glm::cos(phi);
      float z = yz * glm::sin(phi);
      vertices.push_back(glm::vec3{x, y, z} * radius -
                         glm::vec3{halfLength, 0, 0});
      normals.push_back({x, y, z});
      uvs.push_back({static_cast<float>(s) / segments,
                     0.5f - 0.5f * static_cast<float>(r) / rings});
    }
  }

  for (int s = 0; s < segments; ++s) {
    vertices.push_back({-radius - halfLength, 0.f, 0.f});
    normals.push_back({-1.f, 0.f, 0.f});
    uvs.push_back({(0.5f + s) / segments, 0.f});
  }

  for (int s = 0; s < segments; ++s) {
    indices.push_back({s, s + segments, s + segments + 1});
  }

  for (int r = 0; r < rings - 1; ++r) {
    for (int s = 0; s < segments; ++s) {
      indices.push_back({
          segments + (segments + 1) * r + s,
          segments + (segments + 1) * (r + 1) + s,
          segments + (segments + 1) * (r + 1) + s + 1,
      });
      indices.push_back({segments + (segments + 1) * r + s,
                         segments + (segments + 1) * (r + 1) + s + 1,
                         segments + (segments + 1) * r + s + 1});
    }
  }
  for (int s = 0; s < segments; ++s) {
    indices.push_back({segments + (segments + 1) * (rings - 1) + s,
                       segments + (segments + 1) * (rings) + s,
                       segments + (segments + 1) * (rings - 1) + s + 1});
  }

  return makeMesh(vertices, indices, normals, uvs);
}

std::shared_ptr<SVMesh> SVMesh::CreateCone(int segments) {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;
  std::vector<glm::ivec3> indices;

  float s22 = glm::sqrt(2.f) / 2.f;
  for (int s = 0; s < segments; ++s) {
    vertices.push_back({1, 0, 0});
    float theta = glm::pi<float>() * 2.f * (s + 0.5f) / segments;
    normals.push_back({s22 * glm::cos(theta), s22 * glm::sin(theta), s22});
    uvs.push_back({0.25f, 0.5f});
  }
  for (int s = 0; s < segments; ++s) {
    float theta = glm::pi<float>() * 2.f * s / segments;
    vertices.push_back({0.f, glm::cos(theta), glm::sin(theta)});
    normals.push_back({s22 * glm::cos(theta), s22 * glm::sin(theta), s22});
    uvs.push_back({glm::cos(theta) * 0.25 + 0.25, glm::sin(theta) * 0.5 + 0.5});
  }
  for (int s = 0; s < segments; ++s) {
    float theta = glm::pi<float>() * 2.f * s / segments;
    vertices.push_back({0.f, glm::cos(theta), glm::sin(theta)});
    normals.push_back({-1.f, 0.f, 0.f});
    uvs.push_back({glm::cos(theta) * 0.25 + 0.75, glm::sin(theta) * 0.5 + 0.5});
  }
  vertices.push_back({0.f, 0.f, 0.f});
  normals.push_back({-1.f, 0.f, 0.f});
  uvs.push_back({0.75f, 0.5f});

  for (int s = 0; s < segments; ++s) {
    indices.push_back({s, s + segments, segments + (s + 1) % segments});
  }
  for (int s = 0; s < segments; ++s) {
    indices.push_back(
        {segments * 2 + (s + 1) % segments, segments * 2 + s, segments * 3});
  }

  return makeMesh(vertices, indices, normals, uvs);
}

std::shared_ptr<SVMesh> SVMesh::CreateCube() {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;
  std::vector<glm::ivec3> indices;

  // +Z
  vertices.push_back({-1.f, 1.f, 1.f});
  vertices.push_back({-1.f, -1.f, 1.f});
  vertices.push_back({1.f, -1.f, 1.f});
  vertices.push_back({1.f, 1.f, 1.f});
  for (int i = 0; i < 4; ++i) {
    normals.push_back({0.f, 0.f, 1.f});
  }
  uvs.push_back({1.f / 4.f, 2.f / 3.f});
  uvs.push_back({1.f / 4.f, 1.f / 3.f});
  uvs.push_back({2.f / 4.f, 1.f / 3.f});
  uvs.push_back({2.f / 4.f, 2.f / 3.f});

  // -Z
  vertices.push_back({1.f, 1.f, -1.f});
  vertices.push_back({1.f, -1.f, -1.f});
  vertices.push_back({-1.f, -1.f, -1.f});
  vertices.push_back({-1.f, 1.f, -1.f});
  for (int i = 0; i < 4; ++i) {
    normals.push_back({0.f, 0.f, -1.f});
  }
  uvs.push_back({3.f / 4.f, 2.f / 3.f});
  uvs.push_back({3.f / 4.f, 1.f / 3.f});
  uvs.push_back({4.f / 4.f, 1.f / 3.f});
  uvs.push_back({4.f / 4.f, 2.f / 3.f});

  // +X
  vertices.push_back({1.f, 1.f, 1.f});
  vertices.push_back({1.f, -1.f, 1.f});
  vertices.push_back({1.f, -1.f, -1.f});
  vertices.push_back({1.f, 1.f, -1.f});
  for (int i = 0; i < 4; ++i) {
    normals.push_back({1.f, 0.f, 0.f});
  }
  uvs.push_back({2.f / 4.f, 2.f / 3.f});
  uvs.push_back({2.f / 4.f, 1.f / 3.f});
  uvs.push_back({3.f / 4.f, 1.f / 3.f});
  uvs.push_back({3.f / 4.f, 2.f / 3.f});

  // -X
  vertices.push_back({-1.f, 1.f, -1.f});
  vertices.push_back({-1.f, -1.f, -1.f});
  vertices.push_back({-1.f, -1.f, 1.f});
  vertices.push_back({-1.f, 1.f, 1.f});
  for (int i = 0; i < 4; ++i) {
    normals.push_back({-1.f, 0.f, 0.f});
  }
  uvs.push_back({0.f / 4.f, 2.f / 3.f});
  uvs.push_back({0.f / 4.f, 1.f / 3.f});
  uvs.push_back({1.f / 4.f, 1.f / 3.f});
  uvs.push_back({1.f / 4.f, 2.f / 3.f});

  // +Y
  vertices.push_back({-1.f, 1.f, -1.f});
  vertices.push_back({-1.f, 1.f, 1.f});
  vertices.push_back({1.f, 1.f, 1.f});
  vertices.push_back({1.f, 1.f, -1.f});
  for (int i = 0; i < 4; ++i) {
    normals.push_back({0.f, 1.f, 0.f});
  }
  uvs.push_back({1.f / 4.f, 3.f / 3.f});
  uvs.push_back({1.f / 4.f, 2.f / 3.f});
  uvs.push_back({2.f / 4.f, 2.f / 3.f});
  uvs.push_back({2.f / 4.f, 3.f / 3.f});

  // -Y
  vertices.push_back({-1.f, -1.f, 1.f});
  vertices.push_back({-1.f, -1.f, -1.f});
  vertices.push_back({1.f, -1.f, -1.f});
  vertices.push_back({1.f, -1.f, 1.f});
  for (int i = 0; i < 4; ++i) {
    normals.push_back({0.f, -1.f, 0.f});
  }
  uvs.push_back({1.f / 4.f, 1.f / 3.f});
  uvs.push_back({1.f / 4.f, 0.f / 3.f});
  uvs.push_back({2.f / 4.f, 0.f / 3.f});
  uvs.push_back({2.f / 4.f, 1.f / 3.f});

  indices.push_back({0, 1, 2});
  indices.push_back({0, 2, 3});
  indices.push_back({4, 5, 6});
  indices.push_back({4, 6, 7});
  indices.push_back({8, 9, 10});
  indices.push_back({8, 10, 11});
  indices.push_back({12, 13, 14});
  indices.push_back({12, 14, 15});
  indices.push_back({16, 17, 18});
  indices.push_back({16, 18, 19});
  indices.push_back({20, 21, 22});
  indices.push_back({20, 22, 23});

  return makeMesh(vertices, indices, normals, uvs);
}

std::shared_ptr<SVMesh> SVMesh::CreateYZPlane() {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> uvs;
  std::vector<glm::ivec3> indices;

  vertices.push_back({0.f, 1.f, 1.f});
  vertices.push_back({0.f, -1.f, 1.f});
  vertices.push_back({0.f, -1.f, -1.f});
  vertices.push_back({0.f, 1.f, -1.f});
  for (int i = 0; i < 4; ++i) {
    normals.push_back({1.f, 0.f, 0.f});
  }
  uvs.push_back({0.f, 1.f});
  uvs.push_back({0.f, 0.f});
  uvs.push_back({1.f, 0.f});
  uvs.push_back({1.f, 1.f});

  indices.push_back({0, 1, 2});
  indices.push_back({0, 2, 3});

  return makeMesh(vertices, indices, normals, uvs);
}

} // namespace resource
} // namespace svulkan2
