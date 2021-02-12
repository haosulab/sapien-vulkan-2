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

  mContext->getResourceManager().setMaterialPipelineType(
      mShaderManager->getShaderConfig()->materialPipeline);

  mContext->getResourceManager().setVertexLayout(
      mShaderManager->getShaderConfig()->vertexLayout);

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
  mRenderTargetFinalLayouts = mShaderManager->getRenderTargetFinalLayouts();
}

void Renderer::preparePipelines(int numDirectionalLights, int numPointLights) {
  mShaderManager->createPipelines(*mContext, numDirectionalLights,
                                  numPointLights);
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
                                   width, height, 1);
    mFramebuffers.push_back(
        mContext->getDevice().createFramebufferUnique(info));
  }
}

void Renderer::resize(int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error(
        "failed to resize: width and height must be positive.");
  }
  mWidth = width;
  mHeight = height;
  mRequiresRebuild = true;
}

void Renderer::render(vk::CommandBuffer commandBuffer, scene::Scene &scene,
                      resource::SVCamera &camera) {
  if (mWidth <= 0 || mHeight <= 0) {
    throw std::runtime_error(
        "failed to render: resize must be called before rendering.");
  }
  int numPointLights = scene.getSVScene()->getPointLights().size();
  int numDirectionalLights = scene.getSVScene()->getDirectionalLights().size();
  bool numLightsChanged = (numPointLights != mLastNumPointLights) ||
                          (numDirectionalLights != mLastNumDirectionalLights);
  if (mRequiresRebuild || numLightsChanged) {
    mRequiresRebuild = false;
    mLastNumPointLights = numPointLights;
    mLastNumDirectionalLights = numDirectionalLights;

    prepareRenderTargets(mWidth, mHeight);
    preparePipelines(numDirectionalLights, numPointLights);
    prepareFramebuffers(mWidth, mHeight);
    prepareDeferredDescriptorSets();
  }
  auto objects = scene.getObjects();
  prepareObjectBuffers(objects.size());
  prepareSceneBuffer();
  prepareCameaBuffer();

  // load objects to CPU, if not already loaded
  std::vector<std::future<void>> futures;
  for (auto obj : objects) {
    futures.push_back(obj->getModel()->loadAsync());
  }
  for (auto &f : futures) {
    f.get();
  }

  // upload objects to GPU, if not up-to-date
  for (auto obj : objects) {
    for (auto shape : obj->getModel()->getShapes()) {
      shape->material->uploadToDevice(*mContext);
      shape->mesh->uploadToDevice(*mContext);
    }
  }

  // update camera
  camera.uploadToDevice(*mCameraBuffer,
                        *mShaderManager->getShaderConfig()->cameraBufferLayout);
  // update scene
  scene.getSVScene()->uploadToDevice(
      *mSceneBuffer, *mShaderManager->getShaderConfig()->sceneBufferLayout);

  // update objects
  for (uint32_t i = 0; i < objects.size(); ++i) {
    objects[i]->uploadToDevice(
        *mObjectBuffers[i],
        *mShaderManager->getShaderConfig()->objectBufferLayout);
  }

  auto passes = mShaderManager->getAllPasses();
  uint32_t composite_pass_index = 0;
  vk::Viewport viewport{
      0.f, 0.f, static_cast<float>(mWidth), static_cast<float>(mHeight),
      0.f, 1.f};
  vk::Rect2D scissor{vk::Offset2D{0u, 0u},
                     vk::Extent2D{static_cast<uint32_t>(mWidth),
                                  static_cast<uint32_t>(mHeight)}};
  for (uint32_t pass_index = 0; pass_index < passes.size(); ++pass_index) {
    auto pass = passes[pass_index];
    std::vector<vk::ClearValue> clearValues(
        pass->getTextureOutputLayout()->elements.size(),
        vk::ClearColorValue(std::array<float, 4>{0.f, 0.f, 0.f, 0.f}));
    if (auto p = std::dynamic_pointer_cast<shader::GbufferPassParser>(pass)) {
      clearValues.push_back(vk::ClearDepthStencilValue(1.0f, 0));
      vk::RenderPassBeginInfo renderPassBeginInfo{
          pass->getRenderPass(), mFramebuffers[pass_index].get(),
          vk::Rect2D({0, 0}, {static_cast<uint32_t>(mWidth),
                              static_cast<uint32_t>(mHeight)}),
          static_cast<uint32_t>(clearValues.size()), clearValues.data()};
      commandBuffer.beginRenderPass(renderPassBeginInfo,
                                    vk::SubpassContents::eInline);
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pass->getPipeline());
      commandBuffer.setViewport(0, viewport);
      commandBuffer.setScissor(0, scissor);

      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       pass->getPipelineLayout(), 0,
                                       mCameraSet.get(), nullptr);

      uint32_t i = 0;
      for (auto obj : objects) {
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                         pass->getPipelineLayout(), 1,
                                         mObjectSet[i].get(), nullptr);
        auto shapes = obj->getModel()->getShapes();
        for (auto shape : shapes) {
          commandBuffer.bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(), 2,
              shape->material->getDescriptorSet(), nullptr);
          commandBuffer.bindVertexBuffers(
              0, shape->mesh->getVertexBuffer().getVulkanBuffer(),
              std::vector<vk::DeviceSize>(1, 0));
          commandBuffer.bindIndexBuffer(
              shape->mesh->getIndexBuffer().getVulkanBuffer(), 0,
              vk::IndexType::eUint32);
          commandBuffer.drawIndexed(shape->mesh->getIndexCount(), 1, 0, 0, 0);
        }
      }
      commandBuffer.endRenderPass();
    } else if (auto p = std::dynamic_pointer_cast<shader::DeferredPassParser>(
                   pass)) {
      vk::RenderPassBeginInfo renderPassBeginInfo{
          pass->getRenderPass(), mFramebuffers[pass_index].get(),
          vk::Rect2D{
              {0, 0},
              {static_cast<uint32_t>(mWidth), static_cast<uint32_t>(mHeight)}},
          static_cast<uint32_t>(clearValues.size()), clearValues.data()};

      commandBuffer.beginRenderPass(renderPassBeginInfo,
                                    vk::SubpassContents::eInline);
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pass->getPipeline());
      commandBuffer.setViewport(0, viewport);
      commandBuffer.setScissor(0, scissor);
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       pass->getPipelineLayout(), 0,
                                       mSceneSet.get(), nullptr);
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       pass->getPipelineLayout(), 1,
                                       mCameraSet.get(), nullptr);
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       pass->getPipelineLayout(), 2,
                                       mDeferredSet.get(), nullptr);
      commandBuffer.draw(3, 1, 0, 0);
      commandBuffer.endRenderPass();
    } else if (auto p = std::dynamic_pointer_cast<shader::CompositePassParser>(
                   pass)) {
      vk::RenderPassBeginInfo renderPassBeginInfo(
          pass->getRenderPass(), mFramebuffers[pass_index].get(),
          vk::Rect2D{
              {0, 0},
              {static_cast<uint32_t>(mWidth), static_cast<uint32_t>(mHeight)}},
          static_cast<uint32_t>(clearValues.size()), clearValues.data());

      commandBuffer.beginRenderPass(renderPassBeginInfo,
                                    vk::SubpassContents::eInline);
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 pass->getPipeline());
      commandBuffer.setViewport(0, viewport);
      commandBuffer.setScissor(0, scissor);
      commandBuffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(), 0,
          mCompositeSets[composite_pass_index++].get(), nullptr);
      commandBuffer.draw(3, 1, 0, 0);
      commandBuffer.endRenderPass();
    } else {
      throw std::runtime_error("Unknown pass");
    }
  }
  for (auto& [name, target]: mRenderTargets) {
    target->getImage().setCurrentLayout(mRenderTargetFinalLayouts[name]);
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

void Renderer::prepareDeferredDescriptorSets() {
  // deferred set
  {
    auto layout = mShaderManager->getDeferredDescriptorSetLayout();
    mDeferredSet = std::move(
        mContext->getDevice()
            .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                mDescriptorPool.get(), 1, &layout))
            .front());
    std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData;
    auto names = mShaderManager->getDeferredPass()->getInputTextureNames();
    for (auto &name : names) {
      textureData.push_back({mRenderTargets[name]->getImageView(),
                             mRenderTargets[name]->getSampler()});
    }
    updateDescriptorSets(mContext->getDevice(), mDeferredSet.get(), {},
                         textureData, 0);
  }

  // composite sets
  auto compositePasses = mShaderManager->getCompositePasses();
  auto compositeLayouts = mShaderManager->getCompositeDescriptorSetLayouts();
  mCompositeSets.clear();
  mCompositeSets.reserve(compositePasses.size());
  for (uint32_t i = 0; i < compositePasses.size(); ++i) {
    auto layout = compositeLayouts[i];
    mCompositeSets.push_back(std::move(
        mContext->getDevice()
            .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                mDescriptorPool.get(), 1, &layout))
            .front()));
    std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData;
    auto names = compositePasses[i]->getInputTextureNames();
    for (auto &name : names) {
      textureData.push_back({mRenderTargets[name]->getImageView(),
                             mRenderTargets[name]->getSampler()});
    }
    updateDescriptorSets(mContext->getDevice(), mCompositeSets.back().get(), {},
                         textureData, 0);
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
