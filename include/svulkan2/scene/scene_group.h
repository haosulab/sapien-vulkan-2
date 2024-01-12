#include "./scene.h"

namespace svulkan2 {
namespace scene {

class SceneGroup : public Scene {
public:
  SceneGroup(std::vector<std::shared_ptr<Scene>> const &scenes);
  std::vector<Object *> getObjects() override;

private:
  std::vector<std::shared_ptr<Scene>> mScenes;
};

} // namespace scene
} // namespace svulkan2
