#pragma once
#include "camera.h"
#include "light.h"
#include "node.h"
#include "object.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace scene {

class Scene {
  std::vector<std::unique_ptr<Node>> mNodes{};
  std::vector<std::unique_ptr<Object>> mObjects{};
  std::vector<std::unique_ptr<Camera>> mCameras{};
  std::vector<std::unique_ptr<PointLight>> mPointLights{};
  std::vector<std::unique_ptr<DirectionalLight>> mDirectionalLights{};

  Node *mRootNode{nullptr};
  // std::shared_ptr<resource::SVScene> mScene;

  bool mRequireForceRemove{};

  glm::vec4 mAmbientLight;

public:
  inline Node &getRootNode() { return *mRootNode; };

  Node &addNode();
  Node &addNode(Node &parent);

  Object &addObject(std::shared_ptr<resource::SVModel> model);
  Object &addObject(Node &parent, std::shared_ptr<resource::SVModel> model);

  Camera &addCamera();
  Camera &addCamera(Node &parent);

  PointLight &addPointLight();
  PointLight &addPointLight(Node &parent);

  DirectionalLight &addDirectionalLight();
  DirectionalLight &addDirectionalLight(Node &parent);

  void removeNode(Node &node);
  void clearNodes();
  void forceRemove();

  void prepareDeviceResources(core::Context &context);

  inline void setAmbientLight(glm::vec4 const &color) { mAmbientLight = color; }
  inline glm::vec4 getAmbeintLight() const { return mAmbientLight; };

  std::vector<Object *> getObjects();
  std::vector<PointLight *> getPointLights();
  std::vector<DirectionalLight *> getDirectionalLights();
  Scene();

  void uploadToDevice(core::Buffer &sceneBuffer,
                      StructDataLayout const &sceneLayout);

  /** call exactly once per frame to update the object matrices */
  void updateModelMatrices();
};

} // namespace scene
} // namespace svulkan2
