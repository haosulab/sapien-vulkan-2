#include "svulkan2/scene/node.h"
#include <algorithm>

namespace svulkan2 {

Node::Node(std::string const &name) : mName(name) {}

void Node::removeChild(Node &child) {
  auto it = std::find(mChildren.begin(), mChildren.end(), &child);
  if (it == mChildren.end()) {
    throw std::invalid_argument("child node does not exist");
  }
  mChildren.erase(it);
}

void Node::clearChild() { mChildren = {}; }

} // namespace svulkan2
