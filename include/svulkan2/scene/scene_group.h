#include "./scene.h"

namespace svulkan2 {
namespace scene {

class SceneGroup : public Scene {
public:
  SceneGroup(std::vector<std::shared_ptr<Scene>> const &scenes,
             std::vector<Transform> const &transforms);
  std::vector<Object *> getObjects() override;
  void uploadObjectTransforms() override;

private:
  std::vector<std::shared_ptr<Scene>> mScenes;
  std::vector<Transform> mTransforms;
};

} // namespace scene
} // namespace svulkan2
