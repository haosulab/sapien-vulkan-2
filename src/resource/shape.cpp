#include "svulkan2/resource/shape.h"

namespace svulkan2 {
namespace resource {

std::shared_ptr<SVShape> SVShape::Create(std::shared_ptr<SVMesh> mesh,
                                         std::shared_ptr<SVMaterial> material) {
  auto shape = std::make_shared<SVShape>();
  shape->mesh = mesh;
  shape->material = material;
  return shape;
}

} // namespace resource
} // namespace svulkan2
