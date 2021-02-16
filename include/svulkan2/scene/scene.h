#pragma once
#include "camera.h"
#include "node.h"
#include "object.h"
#include "svulkan2/resource/scene.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace scene {

class Scene {
  std::vector<std::unique_ptr<Node>> mNodes{};
  Node *mRootNode{nullptr};
  std::shared_ptr<resource::SVScene> mScene;

  bool mRequireForceRemove{};

public:
  inline Node &getRootNode() { return *mRootNode; };

  Node &addNode();
  Node &addNode(Node &parent);

  Object &addObject(std::shared_ptr<resource::SVModel> model);
  Object &addObject(Node &parent, std::shared_ptr<resource::SVModel> model);

  Camera &addCamera();
  Camera &addCamera(Node &parent);

  // void addNode(std::unique_ptr<Node> node);
  // void addNode(std::unique_ptr<Node> node, Node &parent);
  void removeNode(Node &node);
  void clearNodes();
  void forceRemove();

  void prepareDeviceResources(core::Context &context);

  std::vector<Object *> getObjects();
  Scene();

  inline void setSVScene(std::shared_ptr<resource::SVScene> scene) {
    mScene = scene;
  }

  inline std::shared_ptr<resource::SVScene> getSVScene() const {
    if (!mScene) {
      throw std::runtime_error(
          "getSVScene failed: setSVScene must be called first");
    }
    return mScene;
  }

  /** call exactly once per frame to update the object matrices */
  void updateModelMatrices();
};

} // namespace scene
} // namespace svulkan2
