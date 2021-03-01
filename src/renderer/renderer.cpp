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

void Renderer::prepareShadowRenderTargets() {
  uint32_t shadowSize = mConfig->shadowMapSize;
  auto format = mConfig->depthFormat;

  mDirectionalShadowWriteTargets.clear();
  mPointShadowWriteTargets.clear();
  mCustomShadowWriteTargets.clear();

  auto commandBuffer = mContext->createCommandBuffer();
  commandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  vk::ComponentMapping componentMapping(
      vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
      vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);
  {
    uint32_t size = mNumPointLightShadows ? shadowSize : 1;
    uint32_t num = mNumPointLightShadows ? mNumPointLightShadows : 1;
    auto pointShadowImage = std::make_shared<core::Image>(
        *mContext, vk::Extent3D{size, size, 1}, format,
        vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eSampled,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
        1, 6 * num, vk::ImageTiling::eOptimal,
        vk::ImageCreateFlagBits::eCubeCompatible);

    // HACK: transition to shader read to stop validation from complaining dummy
    // texture is not shader read
    pointShadowImage->transitionLayout(
        commandBuffer.get(), vk::ImageLayout::eUndefined,
        vk::ImageLayout::eShaderReadOnlyOptimal, {},
        vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        vk::PipelineStageFlagBits::eFragmentShader);

    vk::ImageViewCreateInfo viewInfo(
        {}, pointShadowImage->getVulkanImage(), vk::ImageViewType::eCubeArray,
        format, componentMapping,
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                  6 * num));
    auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
    auto sampler = mContext->getDevice().createSamplerUnique(
        vk::SamplerCreateInfo({}, vk::Filter::eNearest, vk::Filter::eNearest,
                              vk::SamplerMipmapMode::eNearest,
                              vk::SamplerAddressMode::eClampToBorder,
                              vk::SamplerAddressMode::eClampToBorder,
                              vk::SamplerAddressMode::eClampToBorder, 0.f,
                              false, 0.f, false, vk::CompareOp::eNever, 0.f,
                              0.f, vk::BorderColor::eFloatOpaqueWhite));
    mPointShadowReadTarget = std::make_shared<resource::SVRenderTarget>(
        "PointShadow", size, size, pointShadowImage, std::move(imageView),
        std::move(sampler));

    for (uint32_t shadowIndex = 0; shadowIndex < num; ++shadowIndex) {
      for (uint32_t faceIndex = 0; faceIndex < 6; ++faceIndex) {
        vk::ImageViewCreateInfo viewInfo(
            {}, pointShadowImage->getVulkanImage(), vk::ImageViewType::e2D,
            format, componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1,
                                      6 * shadowIndex + faceIndex, 1));
        auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
        auto renderTarget = std::make_shared<resource::SVRenderTarget>(
            "PointShadow", size, size, pointShadowImage, std::move(imageView),
            vk::UniqueSampler{});
        mPointShadowWriteTargets.push_back(renderTarget);
      }
    }
  }

  {
    uint32_t size = mNumDirectionalLightShadows ? shadowSize : 1;
    uint32_t num =
        mNumDirectionalLightShadows ? mNumDirectionalLightShadows : 1;
    auto directionalShadowImage = std::make_shared<core::Image>(
        *mContext, vk::Extent3D{size, size, 1}, format,
        vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eSampled,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
        1, num);

    directionalShadowImage->transitionLayout(
        commandBuffer.get(), vk::ImageLayout::eUndefined,
        vk::ImageLayout::eShaderReadOnlyOptimal, {},
        vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        vk::PipelineStageFlagBits::eFragmentShader);

    vk::ImageViewCreateInfo viewInfo(
        {}, directionalShadowImage->getVulkanImage(),
        vk::ImageViewType::e2DArray, format, componentMapping,
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                  num));
    auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
    auto sampler = mContext->getDevice().createSamplerUnique(
        vk::SamplerCreateInfo({}, vk::Filter::eNearest, vk::Filter::eNearest,
                              vk::SamplerMipmapMode::eNearest,
                              vk::SamplerAddressMode::eClampToBorder,
                              vk::SamplerAddressMode::eClampToBorder,
                              vk::SamplerAddressMode::eClampToBorder, 0.f,
                              false, 0.f, false, vk::CompareOp::eNever, 0.f,
                              0.f, vk::BorderColor::eFloatOpaqueWhite));
    mDirectionalShadowReadTarget = std::make_shared<resource::SVRenderTarget>(
        "DirectionalShadow", size, size, directionalShadowImage,
        std::move(imageView), std::move(sampler));

    for (uint32_t shadowIndex = 0; shadowIndex < num; ++shadowIndex) {
      vk::ImageViewCreateInfo viewInfo(
          {}, directionalShadowImage->getVulkanImage(), vk::ImageViewType::e2D,
          format, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1,
                                    shadowIndex, 1));
      auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
      auto renderTarget = std::make_shared<resource::SVRenderTarget>(
          "DirectionalShadow", size, size, directionalShadowImage,
          std::move(imageView), vk::UniqueSampler{});
      mDirectionalShadowWriteTargets.push_back(renderTarget);
    }
  }

  {
    uint32_t size = mNumCustomShadows ? shadowSize : 1;
    uint32_t num = mNumCustomShadows ? mNumCustomShadows : 1;
    auto customShadowImage = std::make_shared<core::Image>(
        *mContext, vk::Extent3D{size, size, 1}, format,
        vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eSampled,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY, vk::SampleCountFlagBits::e1,
        1, num);

    customShadowImage->transitionLayout(
        commandBuffer.get(), vk::ImageLayout::eUndefined,
        vk::ImageLayout::eShaderReadOnlyOptimal, {},
        vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        vk::PipelineStageFlagBits::eFragmentShader);

    vk::ImageViewCreateInfo viewInfo(
        {}, customShadowImage->getVulkanImage(), vk::ImageViewType::e2DArray,
        format, componentMapping,
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                  num));
    auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
    auto sampler = mContext->getDevice().createSamplerUnique(
        vk::SamplerCreateInfo({}, vk::Filter::eNearest, vk::Filter::eNearest,
                              vk::SamplerMipmapMode::eNearest,
                              vk::SamplerAddressMode::eClampToBorder,
                              vk::SamplerAddressMode::eClampToBorder,
                              vk::SamplerAddressMode::eClampToBorder, 0.f,
                              false, 0.f, false, vk::CompareOp::eNever, 0.f,
                              0.f, vk::BorderColor::eFloatOpaqueWhite));
    mCustomShadowReadTarget = std::make_shared<resource::SVRenderTarget>(
        "CustomShadow", size, size, customShadowImage, std::move(imageView),
        std::move(sampler));

    for (uint32_t shadowIndex = 0; shadowIndex < num; ++shadowIndex) {
      vk::ImageViewCreateInfo viewInfo(
          {}, customShadowImage->getVulkanImage(), vk::ImageViewType::e2D,
          format, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1,
                                    shadowIndex, 1));
      auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
      auto renderTarget = std::make_shared<resource::SVRenderTarget>(
          "CustomShadow", size, size, customShadowImage, std::move(imageView),
          vk::UniqueSampler{});
      mCustomShadowWriteTargets.push_back(renderTarget);
    }
  }
  commandBuffer->end();
  mContext->submitCommandBufferAndWait(commandBuffer.get());
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

void Renderer::prepareShadowFramebuffers() {
  mShadowFramebuffers.clear();
  std::vector<std::shared_ptr<resource::SVRenderTarget>> targets;
  targets.insert(targets.end(), mDirectionalShadowWriteTargets.begin(),
                 mDirectionalShadowWriteTargets.begin() +
                     mNumDirectionalLightShadows);
  targets.insert(targets.end(), mPointShadowWriteTargets.begin(),
                 mPointShadowWriteTargets.begin() + mNumPointLightShadows * 6);
  targets.insert(targets.end(), mCustomShadowWriteTargets.begin(),
                 mCustomShadowWriteTargets.begin() + mNumCustomShadows);
  for (auto &target : targets) {
    vk::ImageView view = target->getImageView();
    vk::FramebufferCreateInfo info(
        {}, mShaderManager->getShadowPass()->getRenderPass(), 1, &view,
        target->getWidth(), target->getHeight(), 1);
    mShadowFramebuffers.push_back(
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
  } else {
    mSpecializationConstants[name].dtype = DataType::eINT;
    mSpecializationConstantsChanged = true;
  }
  if (mSpecializationConstants[name].intValue != value) {
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

void Renderer::renderShadows(vk::CommandBuffer commandBuffer,
                             scene::Scene &scene) {
  // render shadow passes
  if (mShaderManager->isShadowEnabled()) {
    auto objects = scene.getObjects();
    auto shadowPass = mShaderManager->getShadowPass();
    auto size = mConfig->shadowMapSize;
    vk::Viewport viewport{
        0.f, 0.f, static_cast<float>(size), static_cast<float>(size), 0.f, 1.f};
    vk::Rect2D scissor{
        vk::Offset2D{0u, 0u},
        vk::Extent2D{static_cast<uint32_t>(size), static_cast<uint32_t>(size)}};
    std::vector<vk::ClearValue> clearValues;
    clearValues.push_back(vk::ClearDepthStencilValue(1.0f, 0));

    for (uint32_t shadowIdx = 0;
         shadowIdx < mNumDirectionalLightShadows + 6 * mNumPointLightShadows +
                         mNumCustomShadows;
         ++shadowIdx) {
      vk::RenderPassBeginInfo renderPassBeginInfo{
          shadowPass->getRenderPass(), mShadowFramebuffers[shadowIdx].get(),
          vk::Rect2D({0, 0}, {static_cast<uint32_t>(size),
                              static_cast<uint32_t>(size)}),
          static_cast<uint32_t>(clearValues.size()), clearValues.data()};
      commandBuffer.beginRenderPass(renderPassBeginInfo,
                                    vk::SubpassContents::eInline);
      commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                 shadowPass->getPipeline());
      commandBuffer.setViewport(0, viewport);
      commandBuffer.setScissor(0, scissor);

      int objectBinding = -1;
      auto types = shadowPass->getUniformBindingTypes();
      for (uint32_t bindingIdx = 0; bindingIdx < types.size(); ++bindingIdx) {
        switch (types[bindingIdx]) {
        case shader::UniformBindingType::eObject:
          objectBinding = bindingIdx;
          break;
        case shader::UniformBindingType::eLight:
          commandBuffer.bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics, shadowPass->getPipelineLayout(),
              bindingIdx, mLightSets[shadowIdx].get(), nullptr);
          break;
        default:
          throw std::runtime_error(
              "shadow pass may only use object and light buffer");
        }
      }

      for (uint32_t objIdx = 0; objIdx < objects.size(); ++objIdx) {
        for (auto &shape : objects[objIdx]->getModel()->getShapes()) {
          if (objectBinding >= 0) {
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                             shadowPass->getPipelineLayout(),
                                             objectBinding,
                                             mObjectSet[objIdx].get(), nullptr);
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
      commandBuffer.endRenderPass();
    }
  }
}

void Renderer::render(vk::CommandBuffer commandBuffer, scene::Scene &scene,
                      scene::Camera &camera) {
  if (mWidth <= 0 || mHeight <= 0) {
    throw std::runtime_error(
        "failed to render: resize must be called before rendering.");
  }
  auto pointLights = scene.getPointLights();
  auto directionalLights = scene.getDirectionalLights();
  int numPointLights = pointLights.size();
  int numDirectionalLights = directionalLights.size();

  mNumPointLightShadows = 0;
  mNumDirectionalLightShadows = 0;
  mNumCustomShadows = scene.getCustomLights().size();

  for (auto l : pointLights) {
    if (l->isShadowEnabled()) {
      mNumPointLightShadows += 1;
    }
  }
  for (auto l : directionalLights) {
    if (l->isShadowEnabled()) {
      mNumDirectionalLightShadows += 1;
    }
  }

  {
    // load custom textures
    std::vector<std::future<void>> futures;
    for (auto t : mCustomTextures) {
      futures.push_back(t.second->loadAsync());
    }
    for (auto &f : futures) {
      f.get();
    }
  }

  setSpecializationConstantInt("NUM_POINT_LIGHTS", numPointLights);
  setSpecializationConstantInt("NUM_DIRECTIONAL_LIGHTS", numDirectionalLights);
  setSpecializationConstantInt("NUM_POINT_LIGHT_SHADOWS",
                               mNumPointLightShadows);
  setSpecializationConstantInt("NUM_DIRECTIONAL_LIGHT_SHADOWS",
                               mNumDirectionalLightShadows);
  setSpecializationConstantInt("NUM_CUSTOM_LIGHT_SHADOWS", mNumCustomShadows);

  if (mRequiresRebuild || mSpecializationConstantsChanged) {
    preparePipelines();
    prepareRenderTargets(mWidth, mHeight);
    if (mShaderManager->isShadowEnabled()) {
      prepareShadowRenderTargets();
      prepareShadowFramebuffers();
    }
    prepareFramebuffers(mWidth, mHeight);
    prepareInputTextureDescriptorSets();
    mSpecializationConstantsChanged = false;
    mRequiresRebuild = false;

    if (mShaderManager->isShadowEnabled()) {
      prepareLightBuffers();
    }
    prepareSceneBuffer();
    prepareCameaBuffer();
  }

  scene.updateModelMatrices();
  auto objects = scene.getObjects();
  prepareObjectBuffers(objects.size());

  // classify shapes
  uint32_t numGbufferPasses = mShaderManager->getNumGbufferPasses();
  std::vector<std::vector<std::shared_ptr<resource::SVShape>>> shapes(
      numGbufferPasses);

  std::vector<std::vector<uint32_t>> shapeObjectIndex(numGbufferPasses);

  int defaultShadingMode = 0;
  int transparencyShadingMode = numGbufferPasses > 1 ? 1 : 0;

  for (uint32_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex) {
    int shadingMode = objects[objectIndex]->getShadingMode();
    if (static_cast<uint32_t>(shadingMode) >= shapes.size()) {
      shadingMode = defaultShadingMode;
    }
    if (shadingMode == 0 && objects[objectIndex]->getTransparency() != 0) {
      shadingMode = transparencyShadingMode;
    }
    for (auto shape : objects[objectIndex]->getModel()->getShapes()) {
      int shapeShadingMode = shadingMode;
      if (shape->material->getOpacity() != 1 && shadingMode == 0) {
        shapeShadingMode = transparencyShadingMode;
      }
      shapes[shapeShadingMode].push_back(shape);
      shapeObjectIndex[shapeShadingMode].push_back(objectIndex);
    }
  }

  // load objects to CPU, if not already loaded
  {
    std::vector<std::future<void>> futures;
    for (auto obj : objects) {
      futures.push_back(obj->getModel()->loadAsync());
    }
    for (auto &f : futures) {
      f.get();
    }
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
  scene.uploadToDevice(*mSceneBuffer,
                       *mShaderManager->getShaderConfig()->sceneBufferLayout);

  if (mShaderManager->isShadowEnabled()) {
    scene.uploadShadowToDevice(
        *mShadowBuffer, mLightBuffers,
        *mShaderManager->getShaderConfig()->shadowBufferLayout);
  }

  // update objects
  for (uint32_t i = 0; i < objects.size(); ++i) {
    objects[i]->uploadToDevice(
        *mObjectBuffers[i],
        *mShaderManager->getShaderConfig()->objectBufferLayout);
  }

  renderShadows(commandBuffer, scene);

  uint32_t gbufferIndex = 0;
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

    if (auto gbufferPass =
            std::dynamic_pointer_cast<shader::GbufferPassParser>(pass)) {
      for (uint32_t i = 0; i < shapes[gbufferIndex].size(); ++i) {
        auto shape = shapes[gbufferIndex][i];
        uint32_t objectIndex = shapeObjectIndex[gbufferIndex][i];
        if (objectBinding >= 0) {
          commandBuffer.bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(),
              objectBinding, mObjectSet[objectIndex].get(), nullptr);
        }
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
      gbufferIndex++;
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
  if (mShaderManager->isShadowEnabled()) {
    mShadowBuffer = mContext->getAllocator().allocateUniformBuffer(
        mShaderManager->getShaderConfig()->shadowBufferLayout->size);
  }
  mSceneBuffer = mContext->getAllocator().allocateUniformBuffer(
      mShaderManager->getShaderConfig()->sceneBufferLayout->size);
  auto layout = mShaderManager->getSceneDescriptorSetLayout();
  mSceneSet =
      std::move(mContext->getDevice()
                    .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                        mDescriptorPool.get(), 1, &layout))
                    .front());
  auto &setDesc = mShaderManager->getSceneSetDesc();
  for (uint32_t bindingIndex = 0; bindingIndex < setDesc.bindings.size();
       ++bindingIndex) {
    auto binding = setDesc.bindings.at(bindingIndex);
    if (binding.name == "SceneBuffer") {
      updateDescriptorSets(mContext->getDevice(), mSceneSet.get(),
                           {{vk::DescriptorType::eUniformBuffer,
                             mSceneBuffer->getVulkanBuffer(), nullptr}},
                           {}, bindingIndex);
    } else if (binding.name == "ShadowBuffer") {
      updateDescriptorSets(mContext->getDevice(), mSceneSet.get(),
                           {{vk::DescriptorType::eUniformBuffer,
                             mShadowBuffer->getVulkanBuffer(), nullptr}},
                           {}, bindingIndex);
    } else if (binding.name == "samplerPointLightDepths") {
      updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                           {{mPointShadowReadTarget->getImageView(),
                             mPointShadowReadTarget->getSampler()}},
                           bindingIndex);
    } else if (binding.name == "samplerDirectionalLightDepths") {
      updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                           {{mDirectionalShadowReadTarget->getImageView(),
                             mDirectionalShadowReadTarget->getSampler()}},
                           bindingIndex);
    } else if (binding.name == "samplerCustomLightDepths") {
      updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                           {{mCustomShadowReadTarget->getImageView(),
                             mCustomShadowReadTarget->getSampler()}},
                           bindingIndex);
    } else if (binding.type == vk::DescriptorType::eCombinedImageSampler &&
               binding.name.starts_with("sampler")) {
      std::string customTextureName = binding.name.substr(7);
      if (customTextureName.starts_with("Random")) {
        auto randomTex = mContext->getResourceManager().CreateRandomTexture(
            customTextureName);
        randomTex->uploadToDevice(*mContext);
        updateDescriptorSets(
            mContext->getDevice(), mSceneSet.get(), {},
            {{randomTex->getImageView(), randomTex->getSampler()}},
            bindingIndex);
      } else if (mCustomTextures.contains(customTextureName)) {
        mCustomTextures[customTextureName]->uploadToDevice(*mContext);
        updateDescriptorSets(
            mContext->getDevice(), mSceneSet.get(), {},
            {{mCustomTextures[customTextureName]->getImageView(),
              mCustomTextures[customTextureName]->getSampler()}},
            bindingIndex);
      } else {
        throw std::runtime_error("custom sampler \"" + customTextureName +
                                 "\" is not set in the renderer");
      }
    } else {
      throw std::runtime_error("unrecognized uniform binding in scene \"" +
                               binding.name + "\"");
    }
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

void Renderer::prepareLightBuffers() {
  auto lightBufferLayout = mShaderManager->getShaderConfig()->lightBufferLayout;
  uint32_t numShadows = mNumPointLightShadows * 6 +
                        mNumDirectionalLightShadows + mNumCustomShadows;
  // too many shadow sets
  if (numShadows * 2 < mLightSets.size()) {
    uint32_t newSize = numShadows;
    mLightSets.resize(newSize);
    mLightBuffers.resize(newSize);
  }

  // too few shadow sets
  for (uint32_t i = mLightSets.size(); i < numShadows; ++i) {
    auto layout = mShaderManager->getLightDescriptorSetLayout();
    mLightBuffers.push_back(mContext->getAllocator().allocateUniformBuffer(
        mShaderManager->getShaderConfig()->lightBufferLayout->size));
    auto shadowSet = std::move(
        mContext->getDevice()
            .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                mDescriptorPool.get(), 1, &layout))
            .front());
    mLightSets.push_back(std::move(shadowSet));
    updateDescriptorSets(mContext->getDevice(), mLightSets.back().get(),
                         {{vk::DescriptorType::eUniformBuffer,
                           mLightBuffers.back()->getVulkanBuffer(), nullptr}},
                         {}, 0);
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
        if (mRenderTargets.contains(name)) {
          textureData.push_back({mRenderTargets[name]->getImageView(),
                                 mRenderTargets[name]->getSampler()});
        } else if (name.starts_with("Random")) {
          auto randomTex =
              mContext->getResourceManager().CreateRandomTexture(name);
          randomTex->uploadToDevice(*mContext);
          textureData.push_back(
              {randomTex->getImageView(), randomTex->getSampler()});
        } else if (mCustomTextures.contains(name)) {
          mCustomTextures[name]->uploadToDevice(*mContext);
          textureData.push_back({mCustomTextures[name]->getImageView(),
                                 mCustomTextures[name]->getSampler()});
        } else {
          throw std::runtime_error("custom sampler \"" + name +
                                   "\" is not set in the renderer");
        }
      }
      updateDescriptorSets(mContext->getDevice(),
                           mInputTextureSets.back().get(), {}, textureData, 0);
    } else {
      mInputTextureSets.push_back({});
    }
  }
}

void Renderer::prepareCameaBuffer() {
  mCameraBuffer = mContext->getAllocator().allocateUniformBuffer(
      mShaderManager->getShaderConfig()->cameraBufferLayout->size);
  auto layout = mShaderManager->getCameraDescriptorSetLayout();
  mCameraSet =
      std::move(mContext->getDevice()
                    .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                        mDescriptorPool.get(), 1, &layout))
                    .front());
  updateDescriptorSets(mContext->getDevice(), mCameraSet.get(),
                       {{vk::DescriptorType::eUniformBuffer,
                         mCameraBuffer->getVulkanBuffer(), nullptr}},
                       {}, 0);
}

void Renderer::setCustomTexture(std::string const &name,
                                std::shared_ptr<resource::SVTexture> texture) {
  mCustomTextures[name] = texture;
}

} // namespace renderer
} // namespace svulkan2
