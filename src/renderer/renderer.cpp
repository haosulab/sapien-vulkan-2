#include "svulkan2/renderer/renderer.h"

namespace svulkan2 {
namespace renderer {

static void updateDescriptorSets(
    vk::Device device, vk::DescriptorSet descriptorSet,
    std::vector<std::tuple<vk::DescriptorType, vk::Buffer,
                           vk::BufferView>> const &bufferData,
    std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData,
    uint32_t bindingOffset) {

  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(bufferData.size());

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(bufferData.size() + textureData.size() ? 1 : 0);

  uint32_t dstBinding = bindingOffset;
  for (auto const &bd : bufferData) {
    bufferInfos.push_back(
        vk::DescriptorBufferInfo(std::get<1>(bd), 0, VK_WHOLE_SIZE));
    writeDescriptorSets.push_back(vk::WriteDescriptorSet(
        descriptorSet, dstBinding++, 0, 1, std::get<0>(bd), nullptr,
        &bufferInfos.back(), std::get<2>(bd) ? &std::get<2>(bd) : nullptr));
  }

  std::vector<vk::DescriptorImageInfo> imageInfos;
  for (auto const &tex : textureData) {
    imageInfos.push_back(
        vk::DescriptorImageInfo(std::get<1>(tex), std::get<0>(tex),
                                vk::ImageLayout::eShaderReadOnlyOptimal));
  }
  if (imageInfos.size()) {
    writeDescriptorSets.push_back(vk::WriteDescriptorSet(
        descriptorSet, dstBinding, 0, imageInfos.size(),
        vk::DescriptorType::eCombinedImageSampler, imageInfos.data()));
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

Renderer::Renderer(core::Context &context,
                   std::shared_ptr<RendererConfig> config)
    : mContext(&context), mConfig(config) {
  mShaderManager = std::make_unique<shader::ShaderManager>(config);
  mShaderManager->processShadersInFolder(config->shaderDir);

  vk::DescriptorPoolSize pool_sizes[] = {
      {vk::DescriptorType::eCombinedImageSampler,
       100}, // render targets and input textures TODO: configure instead of 100
      {vk::DescriptorType::eUniformBuffer, config->maxNumObjects}};
  auto info = vk::DescriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      100 + config->maxNumObjects, 2, pool_sizes);
  mDescriptorPool = mContext->getDevice().createDescriptorPoolUnique(info);
}

void Renderer::prepareRenderTargets(uint32_t width, uint32_t height) {
  auto renderTargetFormats = mShaderManager->getRenderTargetFormats();
  for (auto &[name, format] : renderTargetFormats) {
    mRenderTargets[name] =
        std::make_shared<resource::SVRenderTarget>(name, width, height, format);
    mRenderTargets[name]->createDeviceResources(*mContext);
  }
}

void Renderer::preparePipelines(int numDirectionalLights, int numPointLights) {
  mShaderManager->createPipelines(*mContext, mConfig->culling,
                                  vk::FrontFace::eCounterClockwise,
                                  numDirectionalLights, numPointLights);
}

void Renderer::prepareFramebuffers(uint32_t width, uint32_t height) {
  mFramebuffers.clear();
  auto parsers = mShaderManager->getAllPasses();
  for (uint32_t i = 0; i < parsers.size(); ++i) {
    auto names = parsers[i]->getRenderTargetNames();
    std::vector<vk::ImageView> attachments;
    for (auto &name : names) {
      attachments.push_back(mRenderTargets[name]->getImageView());
    }
    vk::FramebufferCreateInfo info({}, parsers[i]->getRenderPass(),
                                   attachments.size(), attachments.data(),
                                   width, height);
    mFramebuffers.push_back(
        mContext->getDevice().createFramebufferUnique(info));
  }
}

void Renderer::resize(int width, int height) {
  mWidth = width;
  mHeight = height;
  mRequiresRebuild = true;
}

void Renderer::render(vk::CommandBuffer commandBuffer, scene::Scene &scene,
                      resource::SVCamera &camera) {
  if (mRequiresRebuild) {
    mRequiresRebuild = false;
    prepareRenderTargets(mWidth, mHeight);
    int numPointLights = scene.getSVScene()->getPointLights().size();
    int numDirectionalLights =
        scene.getSVScene()->getDirectionalLights().size();
    preparePipelines(numDirectionalLights, numPointLights);
    prepareFramebuffers(mWidth, mHeight);
  }
  auto objects = scene.getObjects();
  prepareObjectBuffers(objects.size());
  prepareSceneBuffer();
  prepareCameaBuffer();

  // load objects to CPU
  for (auto obj : objects) {
    obj->getModel()->load();
  }

  // upload objects to GPU
  for (auto obj : objects) {
    for (auto shape : obj->getModel()->getShapes()) {
      shape->material->uploadToDevice(*mContext);
      shape->mesh->uploadToDevice(*mContext);
    }
  }
}

void Renderer::prepareSceneBuffer() {
  if (!mSceneBuffer) {
    mSceneBuffer = mContext->getAllocator().allocateUniformBuffer(
        mShaderManager->getShaderConfig()->sceneBufferLayout->size);
    auto layout = mShaderManager->getSceneDescriptorSetLayout();
    mSceneSet = std::move(
        mContext->getDevice()
            .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                mDescriptorPool.get(), 1, &layout))
            .front());
    updateDescriptorSets(mContext->getDevice(), mSceneSet.get(),
                         {{vk::DescriptorType::eUniformBuffer,
                           mSceneBuffer->getVulkanBuffer(), nullptr}},
                         {}, 0);
  }
}

void Renderer::prepareObjectBuffers(uint32_t numObjects) {
  // we have too many object buffers
  if (numObjects * 2 < mObjectBuffers.size()) {
    uint32_t newSize = mObjectBuffers.size() / 2;
    mObjectBuffers.resize(newSize);
    mObjectSet.resize(newSize);
  }

  // we have too few object buffers
  for (uint32_t i = mObjectBuffers.size(); i < numObjects; ++i) {
    auto layout = mShaderManager->getObjectDescriptorSetLayout();
    mObjectBuffers.push_back(mContext->getAllocator().allocateUniformBuffer(
        mShaderManager->getShaderConfig()->objectBufferLayout->size));
    auto objectSet = std::move(
        mContext->getDevice()
            .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                mDescriptorPool.get(), 1, &layout))
            .front());
    updateDescriptorSets(mContext->getDevice(), objectSet.get(),
                         {{vk::DescriptorType::eUniformBuffer,
                           mSceneBuffer->getVulkanBuffer(), nullptr}},
                         {}, 0);
    mObjectSet.push_back(std::move(objectSet));
  }
}

void Renderer::prepareCameaBuffer() {
  if (!mCameraBuffer) {
    mCameraBuffer = mContext->getAllocator().allocateUniformBuffer(
        mShaderManager->getShaderConfig()->cameraBufferLayout->size);
    auto layout = mShaderManager->getCameraDescriptorSetLayout();
    mCameraSet = std::move(
        mContext->getDevice()
            .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                mDescriptorPool.get(), 1, &layout))
            .front());
    updateDescriptorSets(mContext->getDevice(), mCameraSet.get(),
                         {{vk::DescriptorType::eUniformBuffer,
                           mSceneBuffer->getVulkanBuffer(), nullptr}},
                         {}, 0);
  }
}

} // namespace renderer
} // namespace svulkan2
