#include "svulkan2/scene/scene_group.h"
#include "../common/logger.h"
#include "svulkan2/core/context.h"
#include <algorithm>

namespace svulkan2 {
namespace scene {

SceneGroup::SceneGroup(std::vector<std::shared_ptr<Scene>> const &scenes) : mScenes(scenes) {}

std::vector<Object *> SceneGroup::getObjects() {
  std::vector<Object *> allObjs;
  for (auto s : mScenes) {
    auto objs = s->getObjects();
    allObjs.insert(allObjs.end(), objs.begin(), objs.end());
  }

  auto objs = Scene::getObjects();
  allObjs.insert(allObjs.end(), objs.begin(), objs.end());

  return allObjs;
}

} // namespace scene
} // namespace svulkan2
