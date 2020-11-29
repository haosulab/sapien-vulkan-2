#pragma once

#include "transform.h"
#include <memory>
#include <string>
#include <vector>

namespace svulkan2 {

class Model;
class Scene;

class Node {
  std::string mName;
  Transform mTransform{};
  Node *mParent;
  std::vector<Node *> mChildren{};
  std::shared_ptr<Model> mModel{nullptr};

  Scene *mScene{nullptr};

  bool mRemoved{false};

public:
  Node(std::string const &name = "");

  inline void setModel(std::shared_ptr<Model> model) { mModel = model; }
  inline std::shared_ptr<Model> getModel() const { return mModel; }

  inline void setName(std::string const &name) { mName = name; }
  inline std::string getName() const { return mName; }

  inline void setScene(Scene *scene) { mScene = scene; }
  inline Scene *getScene() const { return mScene; }

  inline void setParent(Node &parent) { mParent = &parent; }
  inline Node &getParent() const { return *mParent; }

  inline std::vector<Node *> const &getChildren() const { return mChildren; }

  inline void addChild(Node &child) { mChildren.push_back(&child); }
  void removeChild(Node &child);
  void clearChild();

  inline void markRemoved() { mRemoved = true; }
  inline bool isMarkedRemoved() { return mRemoved; };
};

} // namespace svulkan2
