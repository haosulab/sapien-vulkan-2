#pragma once
#include "node.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace scene {

class Scene {
  std::vector<std::unique_ptr<Node>> mNodes{};
  Node *mRootNode{nullptr};

public:
  inline Node &getRootNode() { return *mRootNode; };

  void addNode(std::unique_ptr<Node> node);
  void addNode(std::unique_ptr<Node> node, Node &parent);
  void removedNode(Node &node);
  void clearNodes();

  void forceRemove();

  Scene();
};

} // namespace scene
} // namespace svulkan2
