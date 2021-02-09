#pragma once

#include "svulkan2/resource/camera.h"
#include "svulkan2/resource/model.h"
#include "svulkan2/resource/object.h"
#include "transform.h"
#include <memory>
#include <string>
#include <vector>

namespace svulkan2 {
namespace scene {

class Scene;

class Node {
  std::string mName;
  Transform mTransform{};
  Node *mParent;
  std::vector<Node *> mChildren{};
  std::shared_ptr<resource::SVObject> mObject{nullptr};
  std::shared_ptr<resource::SVCamera> mCamera{nullptr};

  Scene *mScene{nullptr};
  bool mRemoved{false};

public:
  Node(std::string const &name = "");

  void setObject(std::shared_ptr<resource::SVObject> object);
  std::shared_ptr<resource::SVObject> removeObject();
  inline std::shared_ptr<resource::SVObject> const getObject() const {
    return mObject;
  }

  void setCamera(std::shared_ptr<resource::SVCamera> camera);
  std::shared_ptr<resource::SVCamera> removeCamera();
  inline std::shared_ptr<resource::SVCamera> const getCamrea() const {
    return mCamera;
  }

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

  void setTransform(Transform const &transform);
  inline Transform const &getTransform() const { return mTransform; }

  /** called before rendering to update the cached model matrix */
  void updateGlobalModelMatrixRecursive();
  /** called right after updateGlobalModelMatrixRecursive to update object/camera model matrices */
  void updateObjectCameraModelMatrixRecursive();

  std::vector<std::shared_ptr<resource::SVObject>> getObjectsRecursive() const;

};

} // namespace scene
} // namespace svulkan2
