#pragma once
#include "node.h"
#include "svulkan2/resource/scene.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace scene {

class Scene {
  std::vector<std::unique_ptr<Node>> mNodes{};
  Node *mRootNode{nullptr};
  std::shared_ptr<resource::SVScene> mScene;

public:
  inline Node &getRootNode() { return *mRootNode; };

  Node &addNode();
  Node &addNode(Node &parent);

  // void addNode(std::unique_ptr<Node> node);
  // void addNode(std::unique_ptr<Node> node, Node &parent);
  void removedNode(Node &node);
  void clearNodes();
  void forceRemove();

  void prepareObjectCameraForRendering();

  void prepareDeviceResources(core::Context &context);

  std::vector<std::shared_ptr<resource::SVObject>> getObjects() const;
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
