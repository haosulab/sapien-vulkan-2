#pragma once
#include "render_config.h"
#include "svulkan2/common/glm.h"
#include "svulkan2/common/layout.h"
#include <memory>
#include <vector>

namespace svulkan2 {

class VulkanMesh;

class Mesh {
public:
  std::unique_ptr<VulkanMesh> mVulkanMesh;
  DataLayout mVertexLayout;

  std::vector<glm::vec3> mPositions;
  std::vector<glm::vec3> mNormals;
  std::vector<glm::vec2> mUVs;
  std::vector<glm::vec3> mTangents;
  std::vector<glm::vec3> mBitangents;
  std::vector<char> mCustomData;

public:
  Mesh(RenderConfig const &config);
};

} // namespace svulkan2
