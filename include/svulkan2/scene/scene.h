#pragma once
#include "camera.h"
#include "light.h"
#include "node.h"
#include "object.h"
#include <memory>
#include <vector>

namespace svulkan2 {

namespace resource {
class SVCubemap;
}

namespace scene {

class Scene {
  std::vector<std::unique_ptr<Node>> mNodes{};
  std::vector<std::unique_ptr<Object>> mObjects{};
  std::vector<std::unique_ptr<LineObject>> mLineObjects{};
  std::vector<std::unique_ptr<PointObject>> mPointObjects{};
  std::vector<std::unique_ptr<Camera>> mCameras{};
  std::vector<std::unique_ptr<PointLight>> mPointLights{};
  std::vector<std::unique_ptr<DirectionalLight>> mDirectionalLights{};
  std::vector<std::unique_ptr<SpotLight>> mSpotLights{};
  std::vector<std::unique_ptr<TexturedLight>> mTexturedLights{};

  Node *mRootNode{nullptr};

  bool mRequireForceRemove{};

  glm::vec4 mAmbientLight{};

  std::shared_ptr<resource::SVCubemap> mEnvironmentMap{};

  /** when anything is added or removed, the version changes */
  uint64_t mVersion{1l};

public:
  inline Node &getRootNode() { return *mRootNode; };

  Node &addNode(Transform const &transform = {});
  Node &addNode(Node &parent, Transform const &transform = {});

  Object &addObject(std::shared_ptr<resource::SVModel> model,
                    Transform const &transform = {});
  Object &addObject(Node &parent, std::shared_ptr<resource::SVModel> model,
                    Transform const &transform = {});

  LineObject &addLineObject(std::shared_ptr<resource::SVLineSet> lineSet,
                            Transform const &transform = {});
  LineObject &addLineObject(Node &parent,
                            std::shared_ptr<resource::SVLineSet> lineSet,
                            Transform const &transform = {});
  PointObject &addPointObject(std::shared_ptr<resource::SVPointSet> pointSet,
                              Transform const &transform = {});
  PointObject &addPointObject(Node &parent,
                              std::shared_ptr<resource::SVPointSet> pointSet,
                              Transform const &transform = {});
  Camera &addCamera(Transform const &transform = {});
  Camera &addCamera(Node &parent, Transform const &transform = {});

  PointLight &addPointLight();
  PointLight &addPointLight(Node &parent);

  DirectionalLight &addDirectionalLight();
  DirectionalLight &addDirectionalLight(Node &parent);

  SpotLight &addSpotLight();
  SpotLight &addSpotLight(Node &parent);

  TexturedLight &addTexturedLight();
  TexturedLight &addTexturedLight(Node &parent);

  void removeNode(Node &node);
  void clearNodes();
  void forceRemove();

  inline void setAmbientLight(glm::vec4 const &color) { mAmbientLight = color; }
  inline glm::vec4 getAmbientLight() const { return mAmbientLight; };

  std::vector<Object *> getObjects();
  std::vector<LineObject *> getLineObjects();
  std::vector<PointObject *> getPointObjects();
  std::vector<Camera *> getCameras();
  std::vector<PointLight *> getPointLights();
  std::vector<DirectionalLight *> getDirectionalLights();
  std::vector<SpotLight *> getSpotLights();
  std::vector<TexturedLight *> getTexturedLights();

  void setEnvironmentMap(std::shared_ptr<resource::SVCubemap> map) {
    mEnvironmentMap = map;
  }

  std::shared_ptr<resource::SVCubemap> getEnvironmentMap() const {
    return mEnvironmentMap;
  }

  Scene();
  Scene(Scene const &other) = delete;
  Scene &operator=(Scene const &other) = delete;
  Scene(Scene &&other) = default;
  Scene &operator=(Scene &&other) = default;

  void uploadToDevice(core::Buffer &sceneBuffer,
                      StructDataLayout const &sceneLayout);
  void
  uploadShadowToDevice(core::Buffer &shadowBuffer,
                       std::vector<std::unique_ptr<core::Buffer>> &lightBuffers,
                       StructDataLayout const &shadowLayout);

  /** call exactly once per time frame to update the object matrices
   *  this function also copies current matrices into the previous matrices
   */
  void updateModelMatrices();

  /** called to order shadow lights before non-shadow lights */
  void reorderLights();

  uint64_t getVersion() const { return mVersion; }
  void updateVersion();
};

} // namespace scene
} // namespace svulkan2
