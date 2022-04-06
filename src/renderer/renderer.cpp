#include "svulkan2/renderer/renderer.h"
#include <easy/profiler.h>

namespace svulkan2 {
namespace renderer {

static void updateArrayDescriptorSets(
    vk::Device device, vk::DescriptorSet descriptorSet,
    std::vector<std::tuple<vk::ImageView, vk::Sampler>> textureData,
    uint32_t binding) {

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(textureData.size());

  std::vector<vk::DescriptorImageInfo> imageInfos;
  for (auto const &tex : textureData) {
    imageInfos.push_back(
        vk::DescriptorImageInfo(std::get<1>(tex), std::get<0>(tex),
                                vk::ImageLayout::eShaderReadOnlyOptimal));
  }
  for (uint32_t idx = 0; idx < imageInfos.size(); ++idx) {
    writeDescriptorSets.push_back(vk::WriteDescriptorSet(
        descriptorSet, binding, idx, 1,
        vk::DescriptorType::eCombinedImageSampler, &imageInfos[idx]));
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

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

Renderer::Renderer(std::shared_ptr<RendererConfig> config) : mConfig(config) {
  mContext = core::Context::Get();
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  mShaderManager = std::make_unique<shader::ShaderManager>(config);

  mContext->getResourceManager()->setVertexLayout(
      mShaderManager->getShaderConfig()->vertexLayout);

  if (mShaderManager->getShaderConfig()->primitiveVertexLayout) {
    mContext->getResourceManager()->setLineVertexLayout(
        mShaderManager->getShaderConfig()->primitiveVertexLayout);
  }

  vk::DescriptorPoolSize pool_sizes[] = {
      {vk::DescriptorType::eCombinedImageSampler,
       100}, // render targets and input textures TODO: configure instead of 100
  };
  auto info = vk::DescriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 100, 1, pool_sizes);
  mDescriptorPool = mContext->getDevice().createDescriptorPoolUnique(info);

  mObjectPool = std::make_unique<core::DynamicDescriptorPool>(
      std::vector<vk::DescriptorPoolSize>{
          {vk::DescriptorType::eUniformBuffer, 1000}});
}

void Renderer::prepareRenderTargets(uint32_t width, uint32_t height) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  auto renderTargetFormats = mShaderManager->getRenderTargetFormats();
  auto renderTargetScales = mShaderManager->getRenderTargetScales();
  for (auto &[name, format] : renderTargetFormats) {
    float scale = renderTargetScales[name];
    mRenderTargets[name] = std::make_shared<resource::SVRenderTarget>(
        name, static_cast<uint32_t>(width * scale),
        static_cast<uint32_t>(height * scale), format);
    mRenderTargets[name]->createDeviceResources();
  }
  mRenderTargetFinalLayouts = mShaderManager->getRenderTargetFinalLayouts();
}

void Renderer::prepareShadowRenderTargets() {
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  auto format = mConfig->depthFormat;

  mDirectionalShadowWriteTargets.clear();
  mPointShadowWriteTargets.clear();
  mSpotShadowWriteTargets.clear();
  mTexturedLightShadowWriteTargets.clear();

  mDirectionalShadowReadTargets.clear();
  mPointShadowReadTargets.clear();
  mSpotShadowReadTargets.clear();
  mTexturedLightShadowReadTargets.clear();

  auto commandBuffer = mContext->createCommandBuffer();
  commandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  vk::ComponentMapping componentMapping(
      vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
      vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA);

  // point light
  {
    for (uint32_t size : mPointLightShadowSizes) {
      auto pointShadowImage = std::make_shared<core::Image>(
          vk::Extent3D{size, size, 1}, format,
          vk::ImageUsageFlagBits::eDepthStencilAttachment |
              vk::ImageUsageFlagBits::eSampled,
          VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
          vk::SampleCountFlagBits::e1, 1, 6, vk::ImageTiling::eOptimal,
          vk::ImageCreateFlagBits::eCubeCompatible);

      // HACK: transition to shader read to stop validation from complaining
      // dummy texture is not shader read
      pointShadowImage->transitionLayout(
          commandBuffer.get(), vk::ImageLayout::eUndefined,
          vk::ImageLayout::eShaderReadOnlyOptimal, {},
          vk::AccessFlagBits::eShaderRead,
          vk::PipelineStageFlagBits::eBottomOfPipe,
          vk::PipelineStageFlagBits::eFragmentShader);

      // read targets
      vk::ImageViewCreateInfo viewInfo(
          {}, pointShadowImage->getVulkanImage(), vk::ImageViewType::eCube,
          format, componentMapping,
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                    6));
      auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
      auto sampler = mContext->getDevice().createSamplerUnique(
          vk::SamplerCreateInfo({}, vk::Filter::eNearest, vk::Filter::eNearest,
                                vk::SamplerMipmapMode::eNearest,
                                vk::SamplerAddressMode::eClampToBorder,
                                vk::SamplerAddressMode::eClampToBorder,
                                vk::SamplerAddressMode::eClampToBorder, 0.f,
                                false, 0.f, false, vk::CompareOp::eNever, 0.f,
                                0.f, vk::BorderColor::eFloatOpaqueWhite));
      mPointShadowReadTargets.push_back(
          std::make_shared<resource::SVRenderTarget>(
              "PointShadow", size, size, pointShadowImage, std::move(imageView),
              std::move(sampler)));

      // write targets
      for (uint32_t faceIndex = 0; faceIndex < 6; ++faceIndex) {
        vk::ImageViewCreateInfo viewInfo(
            {}, pointShadowImage->getVulkanImage(), vk::ImageViewType::e2D,
            format, componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1,
                                      faceIndex, 1));
        auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
        mPointShadowWriteTargets.push_back(
            std::make_shared<resource::SVRenderTarget>(
                "PointShadow", size, size, pointShadowImage,
                std::move(imageView), vk::UniqueSampler{}));
      }
    }
  }

  // directional light
  {
    for (uint32_t size : mDirectionalLightShadowSizes) {
      auto directionalShadowImage = std::make_shared<core::Image>(
          vk::Extent3D{size, size, 1}, format,
          vk::ImageUsageFlagBits::eDepthStencilAttachment |
              vk::ImageUsageFlagBits::eSampled,
          VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
          vk::SampleCountFlagBits::e1, 1, 1);

      directionalShadowImage->transitionLayout(
          commandBuffer.get(), vk::ImageLayout::eUndefined,
          vk::ImageLayout::eShaderReadOnlyOptimal, {},
          vk::AccessFlagBits::eShaderRead,
          vk::PipelineStageFlagBits::eBottomOfPipe,
          vk::PipelineStageFlagBits::eFragmentShader);

      {
        vk::ImageViewCreateInfo viewInfo(
            {}, directionalShadowImage->getVulkanImage(),
            vk::ImageViewType::e2D, format, componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                      1));
        auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
        auto sampler =
            mContext->getDevice().createSamplerUnique(vk::SamplerCreateInfo(
                {}, vk::Filter::eNearest, vk::Filter::eNearest,
                vk::SamplerMipmapMode::eNearest,
                vk::SamplerAddressMode::eClampToBorder,
                vk::SamplerAddressMode::eClampToBorder,
                vk::SamplerAddressMode::eClampToBorder, 0.f, false, 0.f, false,
                vk::CompareOp::eNever, 0.f, 0.f,
                vk::BorderColor::eFloatOpaqueWhite));
        mDirectionalShadowReadTargets.push_back(
            std::make_shared<resource::SVRenderTarget>(
                "DirectionalShadow", size, size, directionalShadowImage,
                std::move(imageView), std::move(sampler)));
      }

      {
        vk::ImageViewCreateInfo viewInfo(
            {}, directionalShadowImage->getVulkanImage(),
            vk::ImageViewType::e2D, format, componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                      1));
        auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
        mDirectionalShadowWriteTargets.push_back(
            std::make_shared<resource::SVRenderTarget>(
                "DirectionalShadow", size, size, directionalShadowImage,
                std::move(imageView), vk::UniqueSampler{}));
      }
    }
  }

  // spot lights
  {
    for (uint32_t size : mSpotLightShadowSizes) {
      auto shadowImage = std::make_shared<core::Image>(
          vk::Extent3D{size, size, 1}, format,
          vk::ImageUsageFlagBits::eDepthStencilAttachment |
              vk::ImageUsageFlagBits::eSampled,
          VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
          vk::SampleCountFlagBits::e1, 1, 1);

      shadowImage->transitionLayout(commandBuffer.get(),
                                    vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::eShaderReadOnlyOptimal, {},
                                    vk::AccessFlagBits::eShaderRead,
                                    vk::PipelineStageFlagBits::eBottomOfPipe,
                                    vk::PipelineStageFlagBits::eFragmentShader);

      {
        vk::ImageViewCreateInfo viewInfo(
            {}, shadowImage->getVulkanImage(), vk::ImageViewType::e2D, format,
            componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                      1));
        auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
        auto sampler =
            mContext->getDevice().createSamplerUnique(vk::SamplerCreateInfo(
                {}, vk::Filter::eNearest, vk::Filter::eNearest,
                vk::SamplerMipmapMode::eNearest,
                vk::SamplerAddressMode::eClampToBorder,
                vk::SamplerAddressMode::eClampToBorder,
                vk::SamplerAddressMode::eClampToBorder, 0.f, false, 0.f, false,
                vk::CompareOp::eNever, 0.f, 0.f,
                vk::BorderColor::eFloatOpaqueWhite));
        mSpotShadowReadTargets.push_back(
            std::make_shared<resource::SVRenderTarget>(
                "SpotShadow", size, size, shadowImage, std::move(imageView),
                std::move(sampler)));
      }
      {
        vk::ImageViewCreateInfo viewInfo(
            {}, shadowImage->getVulkanImage(), vk::ImageViewType::e2D, format,
            componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                      1));
        auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
        mSpotShadowWriteTargets.push_back(
            std::make_shared<resource::SVRenderTarget>(
                "SpotShadow", size, size, shadowImage, std::move(imageView),
                vk::UniqueSampler{}));
      }
    }
  }

  // active lights
  {
    for (uint32_t size : mTexturedLightShadowSizes) {
      auto shadowImage = std::make_shared<core::Image>(
          vk::Extent3D{size, size, 1}, format,
          vk::ImageUsageFlagBits::eDepthStencilAttachment |
              vk::ImageUsageFlagBits::eSampled,
          VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
          vk::SampleCountFlagBits::e1, 1, 1);

      shadowImage->transitionLayout(commandBuffer.get(),
                                    vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::eShaderReadOnlyOptimal, {},
                                    vk::AccessFlagBits::eShaderRead,
                                    vk::PipelineStageFlagBits::eBottomOfPipe,
                                    vk::PipelineStageFlagBits::eFragmentShader);

      {
        vk::ImageViewCreateInfo viewInfo(
            {}, shadowImage->getVulkanImage(), vk::ImageViewType::e2D, format,
            componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                      1));
        auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
        auto sampler =
            mContext->getDevice().createSamplerUnique(vk::SamplerCreateInfo(
                {}, vk::Filter::eNearest, vk::Filter::eNearest,
                vk::SamplerMipmapMode::eNearest,
                vk::SamplerAddressMode::eClampToBorder,
                vk::SamplerAddressMode::eClampToBorder,
                vk::SamplerAddressMode::eClampToBorder, 0.f, false, 0.f, false,
                vk::CompareOp::eNever, 0.f, 0.f,
                vk::BorderColor::eFloatOpaqueWhite));
        mTexturedLightShadowReadTargets.push_back(
            std::make_shared<resource::SVRenderTarget>(
                "TexturedLightShadow", size, size, shadowImage,
                std::move(imageView), std::move(sampler)));
      }
      {
        vk::ImageViewCreateInfo viewInfo(
            {}, shadowImage->getVulkanImage(), vk::ImageViewType::e2D, format,
            componentMapping,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                      1));
        auto imageView = mContext->getDevice().createImageViewUnique(viewInfo);
        mTexturedLightShadowWriteTargets.push_back(
            std::make_shared<resource::SVRenderTarget>(
                "TexturedLightShadow", size, size, shadowImage,
                std::move(imageView), vk::UniqueSampler{}));
      }
    }
  }

  commandBuffer->end();
  mContext->submitCommandBufferAndWait(commandBuffer.get());
}

void Renderer::preparePipelines() {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  mShaderManager->createPipelines(mSpecializationConstants);
}

void Renderer::prepareFramebuffers(uint32_t width, uint32_t height) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  mFramebuffers.clear();
  auto parsers = mShaderManager->getAllPasses();
  for (uint32_t i = 0; i < parsers.size(); ++i) {
    float scale = parsers[i]->getResolutionScale();
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
                                   static_cast<uint32_t>(scale * width),
                                   static_cast<uint32_t>(scale * height), 1);
    mFramebuffers.push_back(
        mContext->getDevice().createFramebufferUnique(info));
  }
}

void Renderer::prepareShadowFramebuffers() {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  mShadowFramebuffers.clear();
  mShadowSizes.clear();
  std::vector<std::shared_ptr<resource::SVRenderTarget>> targets;
  targets.insert(targets.end(), mDirectionalShadowWriteTargets.begin(),
                 mDirectionalShadowWriteTargets.begin() +
                     mDirectionalLightShadowSizes.size());
  for (uint32_t i = 0; i < mDirectionalLightShadowSizes.size(); ++i) {
    mShadowSizes.push_back(mDirectionalLightShadowSizes[i]);
  }

  targets.insert(targets.end(), mPointShadowWriteTargets.begin(),
                 mPointShadowWriteTargets.begin() +
                     mPointLightShadowSizes.size() * 6);
  for (uint32_t i = 0; i < mPointLightShadowSizes.size(); ++i) {
    for (uint32_t j = 0; j < 6; ++j) {
      mShadowSizes.push_back(mPointLightShadowSizes[i]);
    }
  }

  targets.insert(targets.end(), mSpotShadowWriteTargets.begin(),
                 mSpotShadowWriteTargets.begin() +
                     mSpotLightShadowSizes.size());
  for (uint32_t i = 0; i < mSpotLightShadowSizes.size(); ++i) {
    mShadowSizes.push_back(mSpotLightShadowSizes[i]);
  }

  targets.insert(targets.end(), mTexturedLightShadowWriteTargets.begin(),
                 mTexturedLightShadowWriteTargets.begin() +
                     mTexturedLightShadowSizes.size());
  for (uint32_t i = 0; i < mTexturedLightShadowSizes.size(); ++i) {
    mShadowSizes.push_back(mTexturedLightShadowSizes[i]);
  }

  for (auto &target : targets) {
    vk::ImageView view = target->getImageView();
    vk::FramebufferCreateInfo info(
        {}, mShaderManager->getShadowPass()->getRenderPass(), 1, &view,
        target->getWidth(), target->getHeight(), 1);
    mShadowFramebuffers.push_back(
        mContext->getDevice().createFramebufferUnique(info));
  }
  assert(mShadowSizes.size() == mShadowFramebuffers.size());
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
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (mSpecializationConstants.find(name) != mSpecializationConstants.end()) {
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
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (mSpecializationConstants.find(name) != mSpecializationConstants.end()) {
    if (mSpecializationConstants[name].dtype != DataType::eFLOAT) {
      throw std::runtime_error("failed to set specialization constant: the "
                               "same constant can only have a single type");
    }
  }
  mSpecializationConstants[name].dtype = DataType::eFLOAT;
  mSpecializationConstantsChanged = true;
  mSpecializationConstants[name].floatValue = value;
}

void Renderer::recordShadows(scene::Scene &scene) {
  mShadowCommandBuffer =
      mContext->createCommandBuffer(vk::CommandBufferLevel::ePrimary);
  mShadowCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {}));
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  // render shadow passes
  if (mShaderManager->isShadowEnabled()) {
    auto objects = scene.getObjects();
    auto shadowPass = mShaderManager->getShadowPass();

    std::vector<vk::ClearValue> clearValues;
    clearValues.push_back(vk::ClearDepthStencilValue(1.0f, 0));

    for (uint32_t shadowIdx = 0;
         shadowIdx < mDirectionalLightShadowSizes.size() +
                         6 * mPointLightShadowSizes.size() +
                         mSpotLightShadowSizes.size() +
                         mTexturedLightShadowSizes.size();
         ++shadowIdx) {
      uint32_t size = mShadowSizes[shadowIdx];

      vk::Viewport viewport{
          0.f, 0.f, static_cast<float>(size), static_cast<float>(size),
          0.f, 1.f};
      vk::Rect2D scissor{vk::Offset2D{0u, 0u},
                         vk::Extent2D{static_cast<uint32_t>(size),
                                      static_cast<uint32_t>(size)}};

      vk::RenderPassBeginInfo renderPassBeginInfo{
          shadowPass->getRenderPass(), mShadowFramebuffers[shadowIdx].get(),
          vk::Rect2D({0, 0}, {static_cast<uint32_t>(size),
                              static_cast<uint32_t>(size)}),
          static_cast<uint32_t>(clearValues.size()), clearValues.data()};
      mShadowCommandBuffer->beginRenderPass(renderPassBeginInfo,
                                            vk::SubpassContents::eInline);
      mShadowCommandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics,
                                         shadowPass->getPipeline());
      mShadowCommandBuffer->setViewport(0, viewport);
      mShadowCommandBuffer->setScissor(0, scissor);

      int objectBinding = -1;
      auto types = shadowPass->getUniformBindingTypes();
      for (uint32_t bindingIdx = 0; bindingIdx < types.size(); ++bindingIdx) {
        switch (types[bindingIdx]) {
        case shader::UniformBindingType::eObject:
          objectBinding = bindingIdx;
          break;
        case shader::UniformBindingType::eLight:
          mShadowCommandBuffer->bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics, shadowPass->getPipelineLayout(),
              bindingIdx, mLightSets[shadowIdx].get(), nullptr);
          break;
        default:
          throw std::runtime_error(
              "shadow pass may only use object and light buffer");
        }
      }

      for (uint32_t objIdx = 0; objIdx < objects.size(); ++objIdx) {
        if (objects[objIdx]->getTransparency() >= 1 ||
            !objects[objIdx]->getCastShadow()) {
          continue;
        }

        for (auto &shape : objects[objIdx]->getModel()->getShapes()) {
          if (objectBinding >= 0) {
            mShadowCommandBuffer->bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                shadowPass->getPipelineLayout(), objectBinding,
                mObjectSet[objIdx].get(), nullptr);
          }
          mShadowCommandBuffer->bindVertexBuffers(
              0, shape->mesh->getVertexBuffer().getVulkanBuffer(),
              std::vector<vk::DeviceSize>(1, 0));
          mShadowCommandBuffer->bindIndexBuffer(
              shape->mesh->getIndexBuffer().getVulkanBuffer(), 0,
              vk::IndexType::eUint32);
          mShadowCommandBuffer->drawIndexed(shape->mesh->getIndexCount(), 1, 0,
                                            0, 0);
        }
      }
      mShadowCommandBuffer->endRenderPass();
    }
  }

  if (mShaderManager->isPointShadowEnabled()) {
    auto objects = scene.getPointObjects();
    auto pointShadowPass = mShaderManager->getPointShadowPass();
    std::vector<vk::ClearValue> clearValues;
    clearValues.push_back(vk::ClearDepthStencilValue(1.0f, 0));

    for (uint32_t shadowIdx = 0;
         shadowIdx < mDirectionalLightShadowSizes.size() +
                         6 * mPointLightShadowSizes.size() +
                         mSpotLightShadowSizes.size() +
                         mTexturedLightShadowSizes.size();
         ++shadowIdx) {
      uint32_t size = mShadowSizes[shadowIdx];

      vk::Viewport viewport{
          0.f, 0.f, static_cast<float>(size), static_cast<float>(size),
          0.f, 1.f};
      vk::Rect2D scissor{vk::Offset2D{0u, 0u},
                         vk::Extent2D{static_cast<uint32_t>(size),
                                      static_cast<uint32_t>(size)}};

      vk::RenderPassBeginInfo renderPassBeginInfo{
          pointShadowPass->getRenderPass(),
          mShadowFramebuffers[shadowIdx].get(),
          vk::Rect2D({0, 0}, {static_cast<uint32_t>(size),
                              static_cast<uint32_t>(size)}),
          static_cast<uint32_t>(clearValues.size()), clearValues.data()};
      mShadowCommandBuffer->beginRenderPass(renderPassBeginInfo,
                                            vk::SubpassContents::eInline);
      mShadowCommandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics,
                                         pointShadowPass->getPipeline());
      mShadowCommandBuffer->setViewport(0, viewport);
      mShadowCommandBuffer->setScissor(0, scissor);

      int objectBinding = -1;
      auto types = pointShadowPass->getUniformBindingTypes();
      for (uint32_t bindingIdx = 0; bindingIdx < types.size(); ++bindingIdx) {
        switch (types[bindingIdx]) {
        case shader::UniformBindingType::eObject:
          objectBinding = bindingIdx;
          break;
        case shader::UniformBindingType::eLight:
          mShadowCommandBuffer->bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics,
              pointShadowPass->getPipelineLayout(), bindingIdx,
              mLightSets[shadowIdx].get(), nullptr);
          break;
        default:
          throw std::runtime_error(
              "point shadow pass may only use object and light buffer");
        }
      }

      for (uint32_t objIdx = 0; objIdx < objects.size(); ++objIdx) {
        if (objects[objIdx]->getTransparency() >= 1) {
          continue;
        }
        if (objectBinding >= 0) {
          mShadowCommandBuffer->bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics,
              pointShadowPass->getPipelineLayout(), objectBinding,
              mObjectSet[mPointObjectIndex + objIdx].get(), nullptr);
        }
        mShadowCommandBuffer->bindVertexBuffers(
            0,
            objects[objIdx]->getPointSet()->getVertexBuffer().getVulkanBuffer(),
            vk::DeviceSize(0));
        mShadowCommandBuffer->draw(
            objects[objIdx]->getPointSet()->getVertexCount(), 1, 0, 0);
      }
      mShadowCommandBuffer->endRenderPass();
    }
  }

  mShadowCommandBuffer->end();
}

void Renderer::prepareObjects(scene::Scene &scene) {
  EASY_BLOCK("Prepare objects");
  auto objects = mScene->getObjects();
  auto lineObjects = mScene->getLineObjects();
  auto pointObjects = mScene->getPointObjects();

  auto size = objects.size();
  if (mShaderManager->isLineEnabled()) {
    mLineObjectIndex = size;
    size += lineObjects.size();
  }
  if (mShaderManager->isPointEnabled()) {
    mPointObjectIndex = size;
    size += pointObjects.size();
  }

  prepareObjectBuffers(size);

  EASY_END_BLOCK;

  // load objects to CPU, if not already loaded
  {
    EASY_BLOCK("Load objects to CPU");
    std::vector<std::future<void>> futures;
    for (auto obj : objects) {
      futures.push_back(obj->getModel()->loadAsync());
    }
    for (auto &f : futures) {
      f.get();
    }
  }

  {
    EASY_BLOCK("Upload objects to GPU");
    // upload objects to GPU, if not up-to-date
    for (auto obj : objects) {
      for (auto shape : obj->getModel()->getShapes()) {
        shape->material->uploadToDevice();
        shape->mesh->uploadToDevice();
      }
    }
    if (mShaderManager->isLineEnabled()) {
      for (auto obj : lineObjects) {
        obj->getLineSet()->uploadToDevice();
      }
    }
    if (mShaderManager->isPointEnabled()) {
      for (auto obj : pointObjects) {
        obj->getPointSet()->uploadToDevice();
      }
    }
  }
}

void Renderer::recordRenderPasses(scene::Scene &scene) {
  mRenderCommandBuffer.reset();
  mModelCache.clear();
  mLineSetCache.clear();
  mPointSetCache.clear();

  mRenderCommandBuffer =
      mContext->createCommandBuffer(vk::CommandBufferLevel::ePrimary);
  mRenderCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {}));

  // classify shapes
  uint32_t numGbufferPasses = mShaderManager->getNumGbufferPasses();
  std::vector<std::vector<std::shared_ptr<resource::SVShape>>> shapes(
      numGbufferPasses);

  std::vector<std::vector<uint32_t>> shapeObjectIndex(numGbufferPasses);

  int transparencyShadingMode = numGbufferPasses > 1 ? 1 : 0;

  auto objects = mScene->getObjects();
  for (uint32_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex) {
    int shadingMode = objects[objectIndex]->getShadingMode();
    if (static_cast<uint32_t>(shadingMode) >= shapes.size()) {
      continue; // do not render
    }
    if (shadingMode == 0 && objects[objectIndex]->getTransparency() != 0) {
      shadingMode = transparencyShadingMode;
    }
    if (objects[objectIndex]->getTransparency() >= 1) {
      continue;
    }
    /* HACK: hold onto the models to make sure the underlying buffers are not
     * released until command buffer reset */
    mModelCache.insert(objects[objectIndex]->getModel());

    for (auto shape : objects[objectIndex]->getModel()->getShapes()) {
      int shapeShadingMode = shadingMode;
      if (shape->material->getOpacity() == 0) {
        continue;
      }
      if (shape->material->getOpacity() != 1 && shadingMode == 0) {
        shapeShadingMode = transparencyShadingMode;
      }
      shapes[shapeShadingMode].push_back(shape);
      shapeObjectIndex[shapeShadingMode].push_back(objectIndex);
    }
  }

  auto linesetObjects = mScene->getLineObjects();

  // classify point sets
  uint32_t numPointPasses = mShaderManager->getNumPointPasses();
  auto pointsetObjects = mScene->getPointObjects();
  std::vector<std::vector<uint32_t>> pointsets(numPointPasses);
  for (uint32_t objectIndex = 0; objectIndex < pointsetObjects.size();
       ++objectIndex) {
    auto mode = pointsetObjects[objectIndex]->getShadingMode();
    if (static_cast<uint32_t>(mode) >= pointsets.size()) {
      continue;
    }
    pointsets[mode].push_back(objectIndex);
  }

  uint32_t gbufferIndex = 0;
  uint32_t pointIndex = 0;
  auto passes = mShaderManager->getAllPasses();
  for (uint32_t pass_index = 0; pass_index < passes.size(); ++pass_index) {
    EASY_BLOCK("Record render pass" + std::to_string(pass_index));

    auto pass = passes[pass_index];

    float scale = pass->getResolutionScale();
    uint32_t passWidth = static_cast<uint32_t>(mWidth * scale);
    uint32_t passHeight = static_cast<uint32_t>(mHeight * scale);

    vk::Viewport viewport{
        0.f, 0.f, static_cast<float>(passWidth), static_cast<float>(passHeight),
        0.f, 1.f};
    vk::Rect2D scissor{vk::Offset2D{0u, 0u},
                       vk::Extent2D{static_cast<uint32_t>(passWidth),
                                    static_cast<uint32_t>(passHeight)}};

    std::vector<vk::ClearValue> clearValues(
        pass->getTextureOutputLayout()->elements.size(),
        vk::ClearColorValue(std::array<float, 4>{0.f, 0.f, 0.f, 0.f}));
    clearValues.push_back(vk::ClearDepthStencilValue(1.0f, 0));
    vk::RenderPassBeginInfo renderPassBeginInfo{
        pass->getRenderPass(), mFramebuffers[pass_index].get(),
        vk::Rect2D({0, 0}, {static_cast<uint32_t>(passWidth),
                            static_cast<uint32_t>(passHeight)}),
        static_cast<uint32_t>(clearValues.size()), clearValues.data()};
    mRenderCommandBuffer->beginRenderPass(renderPassBeginInfo,
                                          vk::SubpassContents::eInline);
    mRenderCommandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics,
                                       pass->getPipeline());
    mRenderCommandBuffer->setViewport(0, viewport);
    mRenderCommandBuffer->setScissor(0, scissor);

    int objectBinding = -1;
    int materialBinding = -1;
    auto types = pass->getUniformBindingTypes();
    for (uint32_t i = 0; i < types.size(); ++i) {
      switch (types[i]) {
      case shader::UniformBindingType::eCamera:
        mRenderCommandBuffer->bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(), i,
            mCameraSet.get(), nullptr);
        break;
      case shader::UniformBindingType::eScene:
        mRenderCommandBuffer->bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(), i,
            mSceneSet.get(), nullptr);
        break;
      case shader::UniformBindingType::eTextures:
        mRenderCommandBuffer->bindDescriptorSets(
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
          mRenderCommandBuffer->bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(),
              objectBinding, mObjectSet[objectIndex].get(), nullptr);
        }
        if (materialBinding >= 0) {
          mRenderCommandBuffer->bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(),
              materialBinding, shape->material->getDescriptorSet(), nullptr);
        }
        mRenderCommandBuffer->bindVertexBuffers(
            0, shape->mesh->getVertexBuffer().getVulkanBuffer(),
            std::vector<vk::DeviceSize>(1, 0));
        mRenderCommandBuffer->bindIndexBuffer(
            shape->mesh->getIndexBuffer().getVulkanBuffer(), 0,
            vk::IndexType::eUint32);
        mRenderCommandBuffer->drawIndexed(shape->mesh->getIndexCount(), 1, 0, 0,
                                          0);
      }
      gbufferIndex++;
    } else if (auto linePass =
                   std::dynamic_pointer_cast<shader::LinePassParser>(pass)) {
      for (uint32_t index = 0; index < linesetObjects.size(); ++index) {
        auto &lineObj = linesetObjects[index];
        if (objectBinding >= 0) {
          mRenderCommandBuffer->bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(),
              objectBinding, mObjectSet[mLineObjectIndex + index].get(),
              nullptr);
        }
        if (lineObj->getTransparency() < 1) {
          mLineSetCache.insert(lineObj->getLineSet());
          mRenderCommandBuffer->bindVertexBuffers(
              0, lineObj->getLineSet()->getVertexBuffer().getVulkanBuffer(),
              vk::DeviceSize(0));
          mRenderCommandBuffer->draw(lineObj->getLineSet()->getVertexCount(), 1,
                                     0, 0);
        }
      }
    } else if (auto pointPass =
                   std::dynamic_pointer_cast<shader::PointPassParser>(pass)) {
      for (uint32_t i = 0; i < pointsets[pointIndex].size(); ++i) {
        uint32_t objectIndex = pointsets[pointIndex][i];
        auto &pointObj = pointsetObjects[objectIndex];
        if (objectBinding >= 0) {
          mRenderCommandBuffer->bindDescriptorSets(
              vk::PipelineBindPoint::eGraphics, pass->getPipelineLayout(),
              objectBinding, mObjectSet[mPointObjectIndex + i].get(), nullptr);
        }
        if (pointObj->getTransparency() < 1) {
          mPointSetCache.insert(pointObj->getPointSet());
          mRenderCommandBuffer->bindVertexBuffers(
              0, pointObj->getPointSet()->getVertexBuffer().getVulkanBuffer(),
              vk::DeviceSize(0));
          mRenderCommandBuffer->draw(pointObj->getPointSet()->getVertexCount(),
                                     1, 0, 0);
        }
      }
      pointIndex++;
    } else {
      mRenderCommandBuffer->draw(3, 1, 0, 0);
    }
    mRenderCommandBuffer->endRenderPass();
  }
  mRenderCommandBuffer->end();
}

void Renderer::render(scene::Camera &camera,
                      std::vector<vk::Semaphore> const &waitSemaphores,
                      std::vector<vk::PipelineStageFlags> const &waitStages,
                      std::vector<vk::Semaphore> const &signalSemaphores,
                      vk::Fence fence) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }

  if (!mScene) {
    throw std::runtime_error("setScene must be called before rendering");
  }

  EASY_BLOCK("Record & Submit");
  if (mLastVersion != mScene->getVersion()) {
    mRequiresRecord = true;
  }

  if (mEnvironmentMap != mScene->getEnvironmentMap()) {
    mEnvironmentMap = mScene->getEnvironmentMap();
    mEnvironmentMap->load();
    mRequiresRebuild = true;
  }

  if (mWidth <= 0 || mHeight <= 0) {
    throw std::runtime_error(
        "failed to render: resize must be called before rendering.");
  }

  if (mRequiresRecord) {
    auto pointLights = mScene->getPointLights();
    auto directionalLights = mScene->getDirectionalLights();
    auto spotLights = mScene->getSpotLights();
    auto texturedLights = mScene->getTexturedLights();

    int numPointLights = pointLights.size();
    int numDirectionalLights = directionalLights.size();
    int numSpotLights = spotLights.size();

    mPointLightShadowSizes.clear();
    mDirectionalLightShadowSizes.clear();
    mSpotLightShadowSizes.clear();
    mTexturedLightShadowSizes.clear();

    for (auto l : pointLights) {
      if (l->isShadowEnabled()) {
        mPointLightShadowSizes.push_back(l->getShadowMapSize());
      }
    }
    for (auto l : directionalLights) {
      if (l->isShadowEnabled()) {
        mDirectionalLightShadowSizes.push_back(l->getShadowMapSize());
      }
    }
    for (auto l : spotLights) {
      if (l->isShadowEnabled()) {
        mSpotLightShadowSizes.push_back(l->getShadowMapSize());
      }
    }
    for (auto l : texturedLights) {
      mTexturedLightShadowSizes.push_back(l->getShadowMapSize());
    }

    {
      // load custom textures
      std::vector<std::future<void>> futures;
      for (auto t : mCustomTextures) {
        futures.push_back(t.second->loadAsync());
      }
      for (auto t : mCustomCubemaps) {
        futures.push_back(t.second->loadAsync());
      }
      for (auto l : mScene->getTexturedLights()) {
        if (l->getTexture()) {
          futures.push_back(l->getTexture()->loadAsync());
        }
      }
      for (auto &f : futures) {
        f.get();
      }
      for (auto l : mScene->getTexturedLights()) {
        if (l->getTexture()) {
          l->getTexture()->uploadToDevice();
        }
      }
    }

    setSpecializationConstantInt("NUM_POINT_LIGHTS", numPointLights);
    setSpecializationConstantInt("NUM_DIRECTIONAL_LIGHTS",
                                 numDirectionalLights);
    setSpecializationConstantInt("NUM_SPOT_LIGHTS", numSpotLights);

    setSpecializationConstantInt("NUM_POINT_LIGHT_SHADOWS",
                                 mPointLightShadowSizes.size());
    setSpecializationConstantInt("NUM_DIRECTIONAL_LIGHT_SHADOWS",
                                 mDirectionalLightShadowSizes.size());
    setSpecializationConstantInt("NUM_SPOT_LIGHT_SHADOWS",
                                 mSpotLightShadowSizes.size());
    setSpecializationConstantInt("NUM_TEXTURED_LIGHT_SHADOWS",
                                 mTexturedLightShadowSizes.size());
  }

  if (mRequiresRebuild || mSpecializationConstantsChanged) {
    EASY_BLOCK("Rebuilding Pipeline");
    mRequiresRecord = true;

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

    if (camera.getWidth() != mWidth || camera.getHeight() != mHeight) {
      throw std::runtime_error("Camera size and renderer size does not match. "
                               "Please resize camera first.");
    }
  }

  if (mRequiresRecord) {
    prepareObjects(*mScene);
  }

  {
    EASY_BLOCK("Update camera & scene");
    // update camera
    camera.uploadToDevice(
        *mCameraBuffer, *mShaderManager->getShaderConfig()->cameraBufferLayout);

    // update scene
    mScene->uploadToDevice(
        *mSceneBuffer, *mShaderManager->getShaderConfig()->sceneBufferLayout);

    if (mShaderManager->isShadowEnabled()) {
      mScene->uploadShadowToDevice(
          *mShadowBuffer, mLightBuffers,
          *mShaderManager->getShaderConfig()->shadowBufferLayout);
    }
  }

  {
    EASY_BLOCK("Update objects");
    auto bufferSize =
        mShaderManager->getShaderConfig()->objectBufferLayout->getAlignedSize(
            64);

    // update objects

    mObjectBuffer->map();
    auto objects = mScene->getObjects();
    for (uint32_t i = 0; i < objects.size(); ++i) {
      objects[i]->uploadToDevice(
          *mObjectBuffer, i * bufferSize,
          *mShaderManager->getShaderConfig()->objectBufferLayout);
    }
    if (mShaderManager->isLineEnabled()) {
      auto lineObjects = mScene->getLineObjects();
      for (uint32_t i = 0; i < lineObjects.size(); ++i) {
        lineObjects[i]->uploadToDevice(
            *mObjectBuffer, (mLineObjectIndex + i) * bufferSize,
            *mShaderManager->getShaderConfig()->objectBufferLayout);
      }
    }
    if (mShaderManager->isPointEnabled()) {
      auto pointObjects = mScene->getPointObjects();
      for (uint32_t i = 0; i < pointObjects.size(); ++i) {
        pointObjects[i]->uploadToDevice(
            *mObjectBuffer, (mPointObjectIndex + i) * bufferSize,
            *mShaderManager->getShaderConfig()->objectBufferLayout);
      }
    }
    mObjectBuffer->unmap();
  }

  {
    if (mRequiresRecord) {
      {
        EASY_BLOCK("Record shadow draw calls");
        recordShadows(*mScene);
      }
      {
        EASY_BLOCK("Record render draw calls");
        recordRenderPasses(*mScene);
      }
    }
  }

  vk::CommandBuffer cbs[2] = {mShadowCommandBuffer.get(),
                              mRenderCommandBuffer.get()};
  vk::SubmitInfo info(waitSemaphores.size(), waitSemaphores.data(),
                      waitStages.data(), 2, cbs, signalSemaphores.size(),
                      signalSemaphores.data());
  mContext->getQueue().submit(info, fence);
  mRequiresRecord = false;
  mLastVersion = mScene->getVersion();

  for (auto &[name, target] : mRenderTargets) {
    target->getImage().setCurrentLayout(mRenderTargetFinalLayouts[name]);
  }
}

std::vector<std::string> Renderer::getDisplayTargetNames() const {
  if (!mContext->isVulkanAvailable()) {
    return {};
  }
  std::vector<std::string> result;
  auto renderTargetFormats = mShaderManager->getRenderTargetFormats();
  for (auto &[name, format] : renderTargetFormats) {
    if (format == vk::Format::eR8G8B8A8Unorm ||
        format == vk::Format::eR32G32B32A32Sfloat) {
      result.push_back(name);
    }
  }
  return result;
}

std::vector<std::string> Renderer::getRenderTargetNames() const {
  if (!mContext->isVulkanAvailable()) {
    return {};
  }

  std::vector<std::string> result;
  auto renderTargetFormats = mShaderManager->getRenderTargetFormats();
  for (auto &[name, format] : renderTargetFormats) {
    result.push_back(name);
  }
  return result;
}

void Renderer::display(std::string const &renderTargetName,
                       vk::Image backBuffer, vk::Format format, uint32_t width,
                       uint32_t height,
                       std::vector<vk::Semaphore> const &waitSemaphores,
                       std::vector<vk::PipelineStageFlags> const &waitStages,
                       std::vector<vk::Semaphore> const &signalSemaphores,
                       vk::Fence fence) {
  if (!mContext->isPresentAvailable()) {
    throw std::runtime_error("Display failed: present is not enabled.");
  }

  if (!mDisplayCommandBuffer) {
    mDisplayCommandBuffer = mContext->createCommandBuffer();
  }
  mDisplayCommandBuffer->begin(
      {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  auto &renderTarget = mRenderTargets.at(renderTargetName);
  auto targetFormat = renderTarget->getFormat();
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
      mDisplayCommandBuffer.get(), layout, vk::ImageLayout::eTransferSrcOptimal,
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
    mDisplayCommandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, barrier);
  }
  vk::ImageSubresourceLayers imageSubresourceLayers(
      vk::ImageAspectFlagBits::eColor, 0, 0, 1);
  vk::ImageBlit imageBlit(
      imageSubresourceLayers,
      {{vk::Offset3D{0, 0, 0}, vk::Offset3D{mWidth, mHeight, 1}}},
      imageSubresourceLayers,
      {{vk::Offset3D{0, 0, 0},
        vk::Offset3D{static_cast<int>(width), static_cast<int>(height), 1}}});
  mDisplayCommandBuffer->blitImage(
      renderTarget->getImage().getVulkanImage(),
      vk::ImageLayout::eTransferSrcOptimal, backBuffer,
      vk::ImageLayout::eTransferDstOptimal, imageBlit, vk::Filter::eNearest);

  // transfer swap chain back
  {
    vk::ImageSubresourceRange imageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::ImageMemoryBarrier barrier(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eColorAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, backBuffer, imageSubresourceRange);
    mDisplayCommandBuffer->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eAllCommands, {}, nullptr, nullptr, barrier);
  }

  mDisplayCommandBuffer->end();

  vk::SubmitInfo info(waitSemaphores.size(), waitSemaphores.data(),
                      waitStages.data(), 1, &mDisplayCommandBuffer.get(),
                      signalSemaphores.size(), signalSemaphores.data());
  mContext->getQueue().submit(info, fence);
  renderTarget->getImage().setCurrentLayout(
      vk::ImageLayout::eTransferSrcOptimal);
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
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      for (auto t : mPointShadowReadTargets) {
        textures.push_back({t->getImageView(), t->getSampler()});
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(),
                                textures, bindingIndex);
    } else if (binding.name == "samplerDirectionalLightDepths") {
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      for (auto t : mDirectionalShadowReadTargets) {
        textures.push_back({t->getImageView(), t->getSampler()});
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(),
                                textures, bindingIndex);
    } else if (binding.name == "samplerSpotLightDepths") {
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      for (auto t : mSpotShadowReadTargets) {
        textures.push_back({t->getImageView(), t->getSampler()});
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(),
                                textures, bindingIndex);
    } else if (binding.name == "samplerTexturedLightDepths") {
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      for (auto t : mTexturedLightShadowReadTargets) {
        textures.push_back({t->getImageView(), t->getSampler()});
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(),
                                textures, bindingIndex);
    } else if (binding.name == "samplerTexturedLightTextures") {
      auto lights = mScene->getTexturedLights();
      std::vector<std::tuple<vk::ImageView, vk::Sampler>> textures;
      for (auto l : lights) {
        auto t = l->getTexture();
        if (!t) {
          log::error("A textured light does not have texture!");
          t = mContext->getResourceManager()->getDefaultTexture();
          t->loadAsync().get();
          t->uploadToDevice();
        }
        textures.push_back({t->getImageView(), t->getSampler()});
      }
      updateArrayDescriptorSets(mContext->getDevice(), mSceneSet.get(),
                                textures, bindingIndex);
    } else if (binding.type == vk::DescriptorType::eCombinedImageSampler &&
               binding.name.substr(0, 7) == "sampler") {
      std::string customTextureName = binding.name.substr(7);
      if (customTextureName.substr(0, 6) == "Random") {
        auto randomTex = mContext->getResourceManager()->CreateRandomTexture(
            customTextureName);
        randomTex->uploadToDevice();
        updateDescriptorSets(
            mContext->getDevice(), mSceneSet.get(), {},
            {{randomTex->getImageView(), randomTex->getSampler()}},
            bindingIndex);
      } else if (mCustomTextures.find(customTextureName) !=
                 mCustomTextures.end()) {
        mCustomTextures[customTextureName]->uploadToDevice();
        updateDescriptorSets(
            mContext->getDevice(), mSceneSet.get(), {},
            {{mCustomTextures[customTextureName]->getImageView(),
              mCustomTextures[customTextureName]->getSampler()}},
            bindingIndex);
      } else if (customTextureName == "BRDFLUT") {
        // generate if BRDFLUT is not supplied
        auto tex = mContext->getResourceManager()->getDefaultBRDFLUT();
        updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                             {{tex->getImageView(), tex->getSampler()}},
                             bindingIndex);
      } else if (mCustomCubemaps.find(customTextureName) !=
                 mCustomCubemaps.end()) {
        mCustomCubemaps[customTextureName]->uploadToDevice();
        updateDescriptorSets(
            mContext->getDevice(), mSceneSet.get(), {},
            {{mCustomCubemaps[customTextureName]->getImageView(),
              mCustomCubemaps[customTextureName]->getSampler()}},
            bindingIndex);
      } else if (customTextureName == "Environment") {
        auto cube = mEnvironmentMap;
        if (!cube) {
          cube = mContext->getResourceManager()->getDefaultCubemap();
        }
        cube->uploadToDevice();
        updateDescriptorSets(mContext->getDevice(), mSceneSet.get(), {},
                             {{cube->getImageView(), cube->getSampler()}},
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
} // namespace renderer

void Renderer::prepareObjectBuffers(uint32_t numObjects) {
  // TODO: read aligned size from physical device
  auto bufferSize =
      mShaderManager->getShaderConfig()->objectBufferLayout->getAlignedSize(64);
  bool updated{false};

  // shrink
  if (numObjects * 2 < mObjectSet.size()) {
    updated = true;
    uint32_t newSize = numObjects;

    // reallocate buffer
    mObjectBuffer = mContext->getAllocator().allocateUniformBuffer(
        bufferSize * newSize, false);

    mObjectSet.resize(newSize);
  }
  // expand
  if (numObjects > mObjectSet.size()) {
    updated = true;
    uint32_t newSize =
        std::max(numObjects, 2 * static_cast<uint32_t>(mObjectSet.size()));

    // reallocate buffer
    mObjectBuffer = mContext->getAllocator().allocateUniformBuffer(
        bufferSize * newSize, false);

    auto layout = mShaderManager->getObjectDescriptorSetLayout();
    for (uint32_t i = mObjectSet.size(); i < newSize; ++i) {
      mObjectSet.push_back(mObjectPool->allocateSet(layout));
    }
  }

  if (updated) {
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    std::vector<vk::DescriptorBufferInfo> bufferInfos(mObjectSet.size());

    for (uint32_t i = 0; i < mObjectSet.size(); ++i) {
      auto buffer = mObjectBuffer->getVulkanBuffer();
      bufferInfos[i] =
          vk::DescriptorBufferInfo(buffer, bufferSize * i, bufferSize);
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mObjectSet[i].get(), 0, 0, vk::DescriptorType::eUniformBuffer, {},
          bufferInfos[i], {}));
    }
    mContext->getDevice().updateDescriptorSets(writeDescriptorSets, nullptr);
  }
}

void Renderer::prepareLightBuffers() {
  auto lightBufferLayout = mShaderManager->getShaderConfig()->lightBufferLayout;
  uint32_t numShadows =
      mPointLightShadowSizes.size() * 6 + mDirectionalLightShadowSizes.size() +
      mSpotLightShadowSizes.size() + mTexturedLightShadowSizes.size();

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
        if (mRenderTargets.find(name) != mRenderTargets.end()) {
          textureData.push_back({mRenderTargets[name]->getImageView(),
                                 mRenderTargets[name]->getSampler()});
        } else if (name.substr(0, 6) == "Random") {
          auto randomTex =
              mContext->getResourceManager()->CreateRandomTexture(name);
          randomTex->uploadToDevice();
          textureData.push_back(
              {randomTex->getImageView(), randomTex->getSampler()});
        } else if (mCustomTextures.find(name) != mCustomTextures.end()) {
          mCustomTextures[name]->uploadToDevice();
          textureData.push_back({mCustomTextures[name]->getImageView(),
                                 mCustomTextures[name]->getSampler()});
        } else if (name == "BRDFLUT") {
          // generate if BRDFLUT is not supplied
          auto tex = mContext->getResourceManager()->getDefaultBRDFLUT();
          textureData.push_back({tex->getImageView(), tex->getSampler()});
        } else if (mCustomCubemaps.find(name) != mCustomCubemaps.end()) {
          mCustomCubemaps[name]->uploadToDevice();
          textureData.push_back({mCustomCubemaps[name]->getImageView(),
                                 mCustomCubemaps[name]->getSampler()});
        } else if (name == "Environment") {
          auto cube = mEnvironmentMap;
          if (!cube) {
            cube = mContext->getResourceManager()->getDefaultCubemap();
          }
          cube->uploadToDevice();
          textureData.push_back({cube->getImageView(), cube->getSampler()});
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

void Renderer::setCustomCubemap(std::string const &name,
                                std::shared_ptr<resource::SVCubemap> cubemap) {
  mCustomCubemaps[name] = cubemap;
}

std::shared_ptr<resource::SVRenderTarget>
Renderer::getRenderTarget(std::string const &name) const {
  return mRenderTargets.at(name);
}

std::tuple<std::shared_ptr<core::Buffer>, std::array<uint32_t, 2>, vk::Format>
Renderer::transferToBuffer(std::string const &targetName) {
  auto target = mRenderTargets.at(targetName);
  auto &img = target->getImage();
  auto extent = img.getExtent();
  vk::DeviceSize size = extent.width * extent.height * extent.depth *
                        getFormatSize(img.getFormat());
  {
    auto it = mTransferBuffers.find(targetName);
    if (it == mTransferBuffers.end() || it->second->getSize() != size) {
      EASY_BLOCK("Create Vulkan-cuda buffer");
      mTransferBuffers[targetName] = std::make_shared<core::Buffer>(
          size,
          vk::BufferUsageFlagBits::eTransferSrc |
              vk::BufferUsageFlagBits::eTransferDst,
          VMA_MEMORY_USAGE_GPU_ONLY);
      EASY_END_BLOCK;
    }
  }

  EASY_BLOCK("Wait for render and copy to Vulkan-Cuda buffer");
  img.copyToBuffer(mTransferBuffers[targetName]->getVulkanBuffer(), size,
                   {0, 0, 0}, extent);
  EASY_END_BLOCK;

  return {mTransferBuffers[targetName],
          std::array<uint32_t, 2>{extent.width, extent.height},
          img.getFormat()};
}

} // namespace renderer
} // namespace svulkan2
