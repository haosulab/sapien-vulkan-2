#include "svulkan2/scene/scene_group.h"
#include "../common/logger.h"
#include "svulkan2/core/context.h"
#include <algorithm>

namespace svulkan2 {
namespace scene {

SceneGroup::SceneGroup(std::vector<std::shared_ptr<Scene>> const &scenes,
                       std::vector<Transform> const &transforms)
    : mScenes(scenes), mTransforms(transforms) {
  if (scenes.size() != transforms.size()) {
    throw std::runtime_error(
        "failed to create scene group: scenes and transforms should have the same length");
  }
}

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

void SceneGroup::uploadObjectTransforms() {
  if (mTransformBufferRenderVersion == mRenderVersion) {
    return;
  }

  prepareObjectTransformBuffer();

  std::vector<glm::mat4> data;

  // collect data
  for (uint32_t i = 0; i < mScenes.size(); ++i) {
    auto sceneMat = mTransforms[i].matrix();
    auto objs = mScenes[i]->getObjects();
    for (auto obj : objs) {
      data.push_back(sceneMat * obj->getTransform().worldModelMatrix);
    }
  }
  auto objs = Scene::getObjects();
  for (auto obj : objs) {
    data.push_back(obj->getTransform().worldModelMatrix);
  }

  auto lineObjects = getLineObjects();
  auto pointObjects = getPointObjects();
  for (auto obj : lineObjects) {
    data.push_back(obj->getTransform().worldModelMatrix);
  }
  for (auto obj : pointObjects) {
    data.push_back(obj->getTransform().worldModelMatrix);
  }

  // rendering empty scene
  if (data.empty()) {
    return;
  }

  // put data on CPU
  mTransformBufferCpu->upload(data);

  if (!mTransformUpdateCommandBuffer) {
    mTransformUpdateCommandBuffer = getCommandPool().allocateCommandBuffer();
  }
  mTransformUpdateCommandBuffer->reset();
  mTransformUpdateCommandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  vk::BufferCopy region(0, 0, data.size() * sizeof(glm::mat4));
  mTransformUpdateCommandBuffer->copyBuffer(mTransformBufferCpu->getVulkanBuffer(),
                                            mTransformBuffer->getVulkanBuffer(), region);

  vk::BufferMemoryBarrier barrier(
      vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, mTransformBuffer->getVulkanBuffer(), 0, VK_WHOLE_SIZE);
  mTransformUpdateCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                 vk::PipelineStageFlagBits::eVertexShader |
                                                     vk::PipelineStageFlagBits::eFragmentShader,
                                                 {}, {}, barrier, {});

  mTransformUpdateCommandBuffer->end();

  core::Context::Get()->getQueue().submit(mTransformUpdateCommandBuffer.get(), {});

  mTransformBufferRenderVersion = mRenderVersion;
}

} // namespace scene
} // namespace svulkan2
