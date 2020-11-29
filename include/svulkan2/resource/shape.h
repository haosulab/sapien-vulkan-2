#pragma once
#include "material.h"
#include "mesh.h"
#include "texture.h"
#include <memory>
#include <vector>

namespace svulkan2 {

class Shape {
  std::shared_ptr<Mesh> mMesh;
  std::shared_ptr<Material> mMaterial;
};

} // namespace svulkan2
