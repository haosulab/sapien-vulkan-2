#pragma once
#include "material.h"
#include "mesh.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {

struct SVShape {
  std::shared_ptr<SVMesh> mesh;
  std::shared_ptr<SVMaterial> material;
  std::string name;

  static std::shared_ptr<SVShape> Create(std::shared_ptr<SVMesh> mesh,
                                         std::shared_ptr<SVMaterial> material);
};

} // namespace resource
} // namespace svulkan2
