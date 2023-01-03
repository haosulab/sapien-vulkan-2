#pragma once
#include "svulkan2/common/layout.h"
#include "svulkan2/common/log.h"
#include "svulkan2/common/vk.h"
#include "svulkan2/core/buffer.h"
#include <unordered_map>

namespace svulkan2 {
namespace resource {

class SVMesh {
public:
  SVMesh(bool dynamic = false);

  void setIndices(std::vector<uint32_t> const &indices);
  std::vector<uint32_t> const &getIndices() const;

  void setVertexAttribute(std::string const &name,
                          std::vector<float> const &attrib,
                          bool upload = false);
  std::vector<float> const &getVertexAttribute(std::string const &name) const;
  bool hasVertexAttribute(std::string const &name) const;

  uint32_t getVertexSize() const;
  size_t getVertexCount();

  void uploadToDevice();
  void removeFromDevice();

  std::tuple<vk::AccelerationStructureGeometryKHR,
             vk::AccelerationStructureBuildRangeInfoKHR>
  getASGeometry(); // TODO: notify BLAS update after vertex update

  inline bool isOnDevice() const { return mOnDevice; }

  core::Buffer &getVertexBuffer();
  core::Buffer &getIndexBuffer();
  inline uint32_t getIndexCount() const { return mIndexCount; }

  void exportToFile(std::string const &filename) const;

  static std::shared_ptr<SVMesh> CreateUVSphere(int segments, int rings);
  static std::shared_ptr<SVMesh> CreateCapsule(float radius, float halfLength,
                                               int segments, int halfRings);
  static std::shared_ptr<SVMesh> CreateCone(int segments);
  static std::shared_ptr<SVMesh> CreateCube();
  static std::shared_ptr<SVMesh> CreateYZPlane();
  static std::shared_ptr<SVMesh> Create(std::vector<float> const &position,
                                        std::vector<uint32_t> const &index);

private:
  bool mDynamic;
  std::vector<uint32_t> mIndices;
  uint32_t mIndexCount{};
  std::unordered_map<std::string, std::vector<float>> mAttributes;

  bool mDirty{true};
  bool mOnDevice{false};
  size_t mVertexCount{0}; // set only when upload
  std::unique_ptr<core::Buffer> mVertexBuffer;
  std::unique_ptr<core::Buffer> mIndexBuffer;

  std::mutex mUploadingMutex;
};
} // namespace resource
} // namespace svulkan2
