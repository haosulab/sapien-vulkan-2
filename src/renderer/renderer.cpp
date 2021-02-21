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

void Renderer::preparePipelines() {
  mShaderManager->createPipelines(*mContext, mSpecializationConstants);
}

void Renderer::prepareFramebuffers(uint32_t width, uint32_t height) {
  mFramebuffers.clear();
  auto parsers = mShaderManager->getAllPasses();
  for (uint32_t i = 0; i < parsers.size(); ++i) {
    auto names = parsers[i]->getColorRenderTargetNames();
    std::vector<vk::ImageView> attachments;
    for (auto &name : names) {
      attachments.push_back(mRenderTargets[name]->getImageView());
    }
    auto depthName = parsers[i]->getDepthRenderTargetName();
    if (depthName.has_value()) {
      attachments.push_back(mRenderTargets[depthName.value()]->getImageView());
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

void Renderer::setSpecializationConstantInt(std::string const &name,
                                            int value) {
  if (mSpecializationConstants.contains(name)) {
    if (mSpecializationConstants[name].dtype != DataType::eINT) {
      throw std::runtime_error("failed to set specialization constant: the "
                               "same constant can only have a single type");
    }
  }
  if (mSpecializationConstants[name].intValue != value) {
    mSpecializationConstants[name].dtype = DataType::eINT;
    mSpecializationConstantsChanged = true;
    mSpecializationConstants[name].intValue = value;
  }
}

void Renderer::setSpecializationConstantFloat(std::string const &name,
                                              float value) {
  if (mSpecializationConstants.contains(name)) {
    if (mSpecializationConstants[name].dtype != DataType::eFLOAT) {
      throw std::runtime_error("failed to set specialization constant: the "
                               "same constant can only have a single type");
    }
  }
  mSpecializationConstants[name].dtype = DataType::eFLOAT;
  mSpecializationConstantsChanged = true;
  mSpecializationConstants[name].floatValue = value;
}

void Renderer::render(vk::CommandBuffer commandBuffer, scene::Scene &scene,
                      scene::Camera &camera) {
  if (mWidth <= 0 || mHeight <= 0) {
    throw std::runtime_error(
        "failed to render: resize must be called before rendering.");
  }
  int numPointLights = scene.getSVScene()->getPointLights().size();
  int numDirectionalLights = scene.getSVScene()->getDirectionalLights().size();
  setSpecializationConstantInt("NUM_POINT_LIGHTS", numPointLights);
  setSpecializationConstantInt("NUM_DIRECTIONAL_LIGHTS", numDirectionalLights);

  if (mRequiresRebuild || mSpecializationConstantsChanged) {
    prepareRenderTargets(mWidth, mHeight);
    preparePipelines();
    prepareFramebuffers(mWidth, mHeight);
    prepareInputTextureDescriptorSets();
    mSpecializationConstantsChanged = false;
    mRequiresRebuild = false;
  }

  scene.updateModelMatrices();

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
    // if (auto p = std::dynamic_pointer_cast<shader::GbufferPassParser>(pass))
    // {
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

    int objectBinding = -1;
    int materialBinding = -1;
    auto types = pass->getUniformBindingTypes();
    for (uint32_t i = 0; i < types.size(); ++i) {
      switch (types[i]) {
      case shader::UniformBindingType::eCamera:
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                         pass->getPipelineLayout(), i,
                                         mCameraSet.get(), nullptr);
        break;
      case shader::UniformBindingType::eScene:
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                         pass->getPipelineLayout(), i,
                                         mSceneSet.get(), nullptr);
        break;
      case shader::UniformBindingType::eTextures:
        commandBuffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(), i,
            mInputTextureSets[pass_index].get(), nullptr);
        break;
      case shader::UniformBindingType::eObject:
        objectBinding = i;
        break;
      case shader::UniformBindingType::eMaterial:
        materialBinding = i;
        break;
      default:
        throw std::runtime_error("not implemented");
      }
    }

    if (objectBinding >= 0) {
      uint32_t i = 0;
      for (auto obj : objects) {
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                         pass->getPipelineLayout(), 1,
                                         mObjectSet[i++].get(), nullptr);
        auto shapes = obj->getModel()->getShapes();
        for (auto shape : shapes) {
          if (materialBinding >= 0) {
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(),
                materialBinding, shape->material->getDescriptorSet(), nullptr);
          }
          commandBuffer.bindVertexBuffers(
              0, shape->mesh->getVertexBuffer().getVulkanBuffer(),
              std::vector<vk::DeviceSize>(1, 0));
          commandBuffer.bindIndexBuffer(
              shape->mesh->getIndexBuffer().getVulkanBuffer(), 0,
              vk::IndexType::eUint32);
          commandBuffer.drawIndexed(shape->mesh->getIndexCount(), 1, 0, 0, 0);
        }
      }
    } else {
      commandBuffer.draw(3, 1, 0, 0);
    }
    commandBuffer.endRenderPass();
  }
  for (auto &[name, target] : mRenderTargets) {
    target->getImage().setCurrentLayout(mRenderTargetFinalLayouts[name]);
  }
}

void Renderer::display(vk::CommandBuffer commandBuffer,
                       std::string const &renderTargetName,
                       vk::Image backBuffer, vk::Format format, uint32_t width,
                       uint32_t height) {
  auto &renderTarget = mRenderTargets.at(renderTargetName);
  auto targetFormat = renderTarget->getImage().getFormat();
  if (targetFormat != vk::Format::eR8G8B8A8Unorm &&
      targetFormat != vk::Format::eR32G32B32A32Sfloat) {
    throw std::runtime_error(
        "failed to display: only color textures are supported in display");
  };
  auto layout = mRenderTargetFinalLayouts.at(renderTargetName);
  vk::AccessFlags sourceAccessMask;
  vk::PipelineStageFlags sourceStage;
  if (layout == vk::ImageLayout::eColorAttachmentOptimal) {
    sourceAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  } else if (layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
    sourceAccessMask = {};
    sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
  } else {
    throw std::runtime_error("invalid layout");
  }

  // transfer render target
  renderTarget->getImage().transitionLayout(
      commandBuffer, layout, vk::ImageLayout::eTransferSrcOptimal,
      sourceAccessMask, vk::AccessFlagBits::eTransferRead, sourceStage,
      vk::PipelineStageFlagBits::eTransfer);

  // transfer swap chain
  {
    vk::ImageSubresourceRange imageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::ImageMemoryBarrier barrier(
        {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, backBuffer, imageSubresourceRange);
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                  vk::PipelineStageFlagBits::eTransfer, {},
                                  nullptr, nullptr, barrier);
  }
  vk::ImageSubresourceLayers imageSubresourceLayers(
      vk::ImageAspectFlagBits::eColor, 0, 0, 1);
  vk::ImageBlit imageBlit(
      imageSubresourceLayers,
      {{vk::Offset3D{0, 0, 0}, vk::Offset3D{mWidth, mHeight, 1}}},
      imageSubresourceLayers,
      {{vk::Offset3D{0, 0, 0},
        vk::Offset3D{static_cast<int>(width), static_cast<int>(height), 1}}});
  commandBuffer.blitImage(renderTarget->getImage().getVulkanImage(),
                          vk::ImageLayout::eTransferSrcOptimal, backBuffer,
                          vk::ImageLayout::eTransferDstOptimal, imageBlit,
                          vk::Filter::eNearest);

  // transfer swap chain back
  {
    vk::ImageSubresourceRange imageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::ImageMemoryBarrier barrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eColorAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, backBuffer, imageSubresourceRange);
    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eAllCommands, {},
                                  nullptr, nullptr, barrier);
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
                           mObjectBuffers[i]->getVulkanBuffer(), nullptr}},
                         {}, 0);
    mObjectSet.push_back(std::move(objectSet));
  }
}

void Renderer::prepareInputTextureDescriptorSets() {
  auto layouts = mShaderManager->getInputTextureLayouts();
  auto passes = mShaderManager->getAllPasses();
  mInputTextureSets.clear();
  for (uint32_t i = 0; i < passes.size(); ++i) {
    auto layout = layouts[i];
    vk::UniqueDescriptorSet set{};
    if (layout) {
      mInputTextureSets.push_back(std::move(
          mContext->getDevice()
              .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                  mDescriptorPool.get(), 1, &layout))
              .front()));

      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData;
      auto names = passes[i]->getInputTextureNames();
      for (auto name : names) {
        // TODO: allow custom textures
        textureData.push_back({mRenderTargets[name]->getImageView(),
                               mRenderTargets[name]->getSampler()});
      }
      updateDescriptorSets(mContext->getDevice(),
                           mInputTextureSets.back().get(), {}, textureData, 0);

    } else {
      mInputTextureSets.push_back({});
    }
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
                           mCameraBuffer->getVulkanBuffer(), nullptr}},
                         {}, 0);
  }
}

} // namespace renderer
} // namespace svulkan2
