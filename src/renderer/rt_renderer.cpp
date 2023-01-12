#include "svulkan2/renderer/rt_renderer.h"
#include "svulkan2/core/context.h"

namespace svulkan2 {
namespace renderer {
RTRenderer::RTRenderer(std::string const &shaderDir) : mShaderDir(shaderDir) {
  mContext = core::Context::Get();
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mContext->isRayTracingAvailable()) {
    log::error("The selected GPU does not support ray tracing");
    return;
  }
  mShaderPack = mContext->getResourceManager()->CreateRTShaderPack(shaderDir);

  mMaterialBufferLayout = mShaderPack->getMaterialBufferLayout();
  mTextureIndexBufferLayout = mShaderPack->getTextureIndexBufferLayout();
  mGeometryInstanceBufferLayout =
      mShaderPack->getGeometryInstanceBufferLayout();
  mCameraBufferLayout = mShaderPack->getCameraBufferLayout();

  // set default values
  mCustomPropertiesInt["spp"] = 1;
  mCustomPropertiesInt["maxDepth"] = 0;
  mCustomPropertiesInt["russianRoulette"] = 0;
  mCustomPropertiesInt["russianRouletteMinBounces"] = 2;
  mCustomPropertiesVec3["ambientLight"] = {0.f, 0.f, 0.f};
}

void RTRenderer::resize(int width, int height) {
  if (mWidth != width || mHeight != height) {
    mWidth = width;
    mHeight = height;
    mRequiresRebuild = true;
  }
}

void RTRenderer::setScene(scene::Scene &scene) {
  mScene = &scene;
  mSceneVersion = 0l;
  mSceneRenderVersion = 0l;
}

void RTRenderer::prepareRender(scene::Camera &camera) {
  if (mEnvironmentMap != mScene->getEnvironmentMap()) {
    mEnvironmentMap = mScene->getEnvironmentMap();
    mEnvironmentMap->load();
    mRequiresRebuild = true;
  }

  auto objects = camera.getScene()->getObjects();
  auto scene = camera.getScene();
  if (mSceneVersion != scene->getVersion()) {
    mRequiresRebuild = true;
  }

  if (mRequiresRebuild) {
    scene->buildRTResources(mMaterialBufferLayout, mTextureIndexBufferLayout,
                            mGeometryInstanceBufferLayout);

    mShaderPackInstance =
        std::make_shared<shader::RayTracingShaderPackInstance>(
            shader::RayTracingShaderPackInstanceDesc{
                .shaderDir = mShaderDir,
                .maxMeshes =
                    static_cast<uint32_t>(mScene->getRTVertexBuffers().size()),
                .maxMaterials = static_cast<uint32_t>(
                    mScene->getRTMaterialBuffers().size()),
                .maxTextures =
                    static_cast<uint32_t>(mScene->getRTTextures().size())});
    prepareOutput();
    prepareCamera();
    prepareScene();

    mSceneVersion = scene->getVersion();
    mRequiresRebuild = false;
  }

  if (mSceneRenderVersion != mScene->getRenderVersion()) {
    mScene->updateRTResources();                                // update TLAS
    camera.uploadToDevice(*mCameraBuffer, mCameraBufferLayout); // update camera
    mFrameCount = 0;
    mSceneRenderVersion = mScene->getRenderVersion();
  } else {
    mFrameCount += 1;
  }
  log::info("frame {}", mFrameCount);

  updatePushConstant();

  // TODO: test if this is a bottleneck
  recordRender();
}

void RTRenderer::updatePushConstant() {
  auto pushConstantLayout = mShaderPack->getPushConstantLayout();
  mPushConstantBuffer.resize(pushConstantLayout->size);

  auto layout = mShaderPack->getPushConstantLayout();
  for (auto &[name, value] : mCustomPropertiesInt) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::eINT) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value,
                  it->second.size);
    }
  }
  for (auto &[name, value] : mCustomPropertiesFloat) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::eFLOAT) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value,
                  it->second.size);
    }
  }
  for (auto &[name, value] : mCustomPropertiesVec3) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::eFLOAT3) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value,
                  it->second.size);
    }
  }
  for (auto &[name, value] : mCustomPropertiesVec4) {
    auto it = layout->elements.find(name);
    if (it != layout->elements.end() && it->second.dtype == DataType::eFLOAT4) {
      std::memcpy(mPushConstantBuffer.data() + it->second.offset, &value,
                  it->second.size);
    }
  }

  if (layout->elements.contains("pointLightCount")) {
    auto &elem = layout->elements.at("pointLightCount");
    if (elem.dtype != DataType::eINT) {
      throw std::runtime_error("pointLightCount must be type int");
    }
    uint32_t v = mScene->getPointLights().size();
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &v, elem.size);
  }

  if (layout->elements.contains("directionalLightCount")) {
    auto &elem = layout->elements.at("directionalLightCount");
    if (elem.dtype != DataType::eINT) {
      throw std::runtime_error("directionalLightCount must be type int");
    }
    uint32_t v = mScene->getDirectionalLights().size();
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &v, elem.size);
  }

  if (layout->elements.contains("spotLightCount")) {
    auto &elem = layout->elements.at("spotLightCount");
    if (elem.dtype != DataType::eINT) {
      throw std::runtime_error("spotLightCount must be type int");
    }
    uint32_t v =
        mScene->getSpotLights().size() + mScene->getTexturedLights().size();
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &v, elem.size);
  }

  if (layout->elements.contains("frameCount")) {
    auto &elem = layout->elements.at("frameCount");
    if (elem.dtype != DataType::eINT) {
      throw std::runtime_error("frameCount must be type int");
    }
    std::memcpy(mPushConstantBuffer.data() + elem.offset, &mFrameCount,
                elem.size);
  }
}

void RTRenderer::recordRender() {
  if (!mRenderCommandPool) {
    mRenderCommandPool = mContext->createCommandPool();
    mRenderCommandBuffer = mRenderCommandPool->allocateCommandBuffer();
  }

  mRenderCommandBuffer->reset();

  auto pushConstantLayout = mShaderPack->getPushConstantLayout();

  mRenderCommandBuffer->begin(vk::CommandBufferBeginInfo({}, {}));

  mRenderCommandBuffer->pushConstants(
      mShaderPackInstance->getPipelineLayout(),
      vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eMissKHR |
          vk::ShaderStageFlagBits::eClosestHitKHR |
          vk::ShaderStageFlagBits::eAnyHitKHR,
      0, pushConstantLayout->size, mPushConstantBuffer.data());
  mRenderCommandBuffer->bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                                     mShaderPackInstance->getPipeline());

  std::vector<vk::DescriptorSet> descriptorSets;
  auto const &resources = mShaderPack->getResources();
  for (uint32_t sid = 0; sid < resources.size(); ++sid) {
    if (resources.at(sid).type == shader::UniformBindingType::eRTOutput) {
      descriptorSets.push_back(mOutputSet.get());
    } else if (resources.at(sid).type == shader::UniformBindingType::eRTScene) {
      descriptorSets.push_back(mSceneSet.get());
    } else if (resources.at(sid).type ==
               shader::UniformBindingType::eRTCamera) {
      descriptorSets.push_back(mCameraSet.get());
    } else {
      throw std::runtime_error("unrecognized descriptor set");
    }
  }

  mRenderCommandBuffer->bindDescriptorSets(
      vk::PipelineBindPoint::eRayTracingKHR,
      mShaderPackInstance->getPipelineLayout(), 0, descriptorSets, {});

  mRenderCommandBuffer->traceRaysKHR(
      mShaderPackInstance->getRgenRegion(),
      mShaderPackInstance->getMissRegion(), mShaderPackInstance->getHitRegion(),
      mShaderPackInstance->getCallRegion(), mWidth, mHeight, 1);

  mRenderCommandBuffer->end();
}

void RTRenderer::prepareOutput() {
  auto &set = mShaderPack->getOutputDescription();
  if (mWidth <= 0 || mHeight <= 0) {
    return;
  }

  // create images
  std::vector<std::shared_ptr<resource::SVStorageImage>> images;
  for (auto &[bid, binding] : set.bindings) {
    if (binding.type == vk::DescriptorType::eStorageImage &&
        binding.name.starts_with("out")) {
      std::string texName = binding.name.substr(3);
      auto image = mRenderImages[texName] =
          std::make_shared<resource::SVStorageImage>(texName, mWidth, mHeight,
                                                     binding.format);
      images.push_back(image);
      image->createDeviceResources();
    }
  }

  auto pool = mContext->createCommandPool();
  auto commandBuffer = pool->allocateCommandBuffer();
  commandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  for (auto image : images) {
    image->getImage().transitionLayout(
        commandBuffer.get(), vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral, {},
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eRayTracingShaderKHR);
  }
  commandBuffer->end();
  mContext->getQueue().submitAndWait(commandBuffer.get());

  mOutputSet.reset();

  // make descriptor set
  mOutputPool = std::make_unique<core::DynamicDescriptorPool>(
      std::vector<vk::DescriptorPoolSize>{
          {vk::DescriptorType::eStorageImage,
           static_cast<uint32_t>(images.size())}});
  mOutputSet =
      mOutputPool->allocateSet(mShaderPackInstance->getOutputSetLayout());

  // write descriptor sets
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  std::vector<vk::DescriptorImageInfo> imageInfos;
  for (auto img : images) {
    imageInfos.push_back(vk::DescriptorImageInfo(
        VK_NULL_HANDLE, img->getImageView(), vk::ImageLayout::eGeneral));
  }
  for (uint32_t binding = 0; binding < imageInfos.size(); ++binding) {
    writeDescriptorSets.push_back(vk::WriteDescriptorSet(
        mOutputSet.get(), binding, 0, vk::DescriptorType::eStorageImage,
        imageInfos.at(binding)));
  }
  mContext->getDevice().updateDescriptorSets(writeDescriptorSets, {});
}

void RTRenderer::prepareScene() {
  auto &set = mShaderPack->getSceneDescription();
  uint32_t asCount{0};
  uint32_t storageBufferCount{0};
  uint32_t textureCount{0};
  for (auto &[bid, binding] : set.bindings) {
    switch (binding.type) {
    case vk::DescriptorType::eAccelerationStructureKHR:
      ++asCount;
      break;
    case vk::DescriptorType::eStorageBuffer:
      ++storageBufferCount;
      break;
    case vk::DescriptorType::eCombinedImageSampler:
      ++textureCount;
      break;
    default:
      throw std::runtime_error("scene set should only contain acceleration "
                               "structures, storage buffers, and textures");
    }
  }

  mSceneSet.reset();

  mScenePool = std::make_unique<core::DynamicDescriptorPool>(
      std::vector<vk::DescriptorPoolSize>{
          {vk::DescriptorType::eAccelerationStructureKHR, asCount},
          {vk::DescriptorType::eStorageBuffer, storageBufferCount},
          {vk::DescriptorType::eCombinedImageSampler, textureCount}});
  mSceneSet = mScenePool->allocateSet(mShaderPackInstance->getSceneSetLayout());

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  auto as = mScene->getTLAS()->getVulkanAS();
  vk::WriteDescriptorSetAccelerationStructureKHR asWrite(as);

  vk::DescriptorBufferInfo geometryInstanceBufferInfo(
      mScene->getRTGeometryInstanceBuffer(), 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo textureIndexBufferInfo(
      mScene->getRTTextureIndexBuffer(), 0, VK_WHOLE_SIZE);
  std::vector<vk::DescriptorBufferInfo> materialBufferInfos;
  for (auto buffer : mScene->getRTMaterialBuffers()) {
    materialBufferInfos.push_back({buffer, 0, VK_WHOLE_SIZE});
  }
  std::vector<vk::DescriptorImageInfo> textureInfos;
  for (auto [view, sampler] : mScene->getRTTextures()) {
    textureInfos.push_back(vk::DescriptorImageInfo(
        sampler, view, vk::ImageLayout::eShaderReadOnlyOptimal));
  }
  std::vector<vk::DescriptorBufferInfo> vertexBufferInfos;
  for (auto buffer : mScene->getRTVertexBuffers()) {
    vertexBufferInfos.push_back({buffer, 0, VK_WHOLE_SIZE});
  }
  std::vector<vk::DescriptorBufferInfo> indexBufferInfos;
  for (auto buffer : mScene->getRTIndexBuffers()) {
    indexBufferInfos.push_back({buffer, 0, VK_WHOLE_SIZE});
  }

  vk::DescriptorBufferInfo pointLightBufferInfo(mScene->getRTPointLightBuffer(),
                                                0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo directionalLightBufferInfo(
      mScene->getRTDirectionalLightBuffer(), 0, VK_WHOLE_SIZE);
  vk::DescriptorBufferInfo spotLightBufferInfo(mScene->getRTSpotLightBuffer(),
                                               0, VK_WHOLE_SIZE);

  auto cube = mEnvironmentMap;
  if (!cube) {
    cube = mContext->getResourceManager()->getDefaultCubemap();
  }
  cube->uploadToDevice();
  vk::DescriptorImageInfo cubeMapInfo(cube->getSampler(), cube->getImageView(),
                                      vk::ImageLayout::eShaderReadOnlyOptimal);

  for (auto &[bid, binding] : set.bindings) {
    if (binding.name == "tlas") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, 1,
          vk::DescriptorType::eAccelerationStructureKHR, {}, {}, {}, &asWrite));
    } else if (binding.name == "GeometryInstances") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {},
          geometryInstanceBufferInfo, {}));
    } else if (binding.name == "TextureIndices") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {},
          textureIndexBufferInfo, {}));
    } else if (binding.name == "Materials") {
      if (materialBufferInfos.size()) {
        writeDescriptorSets.push_back(vk::WriteDescriptorSet(
            mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {},
            materialBufferInfos, {}));
      }
    } else if (binding.name == "textures") {
      if (textureInfos.size()) {
        writeDescriptorSets.push_back(vk::WriteDescriptorSet(
            mSceneSet.get(), bid, 0, vk::DescriptorType::eCombinedImageSampler,
            textureInfos, {}, {}));
      }
    } else if (binding.name == "Vertices") {
      if (vertexBufferInfos.size()) {
        writeDescriptorSets.push_back(vk::WriteDescriptorSet(
            mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {},
            vertexBufferInfos, {}));
      }
    } else if (binding.name == "Indices") {
      if (indexBufferInfos.size()) {
        writeDescriptorSets.push_back(vk::WriteDescriptorSet(
            mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {},
            indexBufferInfos, {}));
      }
    } else if (binding.name == "PointLights") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {},
          pointLightBufferInfo, {}));
    } else if (binding.name == "DirectionalLights") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {},
          directionalLightBufferInfo, {}));
    } else if (binding.name == "SpotLights") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, vk::DescriptorType::eStorageBuffer, {},
          spotLightBufferInfo, {}));
    } else if (binding.name == "samplerEnvironment") {
      writeDescriptorSets.push_back(vk::WriteDescriptorSet(
          mSceneSet.get(), bid, 0, vk::DescriptorType::eCombinedImageSampler,
          cubeMapInfo, {}, {}));
    } else {
      throw std::runtime_error("unrecognized scene set binding " +
                               binding.name);
    }
  }
  mContext->getDevice().updateDescriptorSets(writeDescriptorSets, {});
}

void RTRenderer::prepareCamera() {
  auto &set = mShaderPack->getCameraDescription();
  for (auto &[bid, binding] : set.bindings) {
    if (binding.name == "CameraBuffer") {
      mCameraBuffer = mContext->getAllocator().allocateUniformBuffer(
          set.buffers.at(binding.arrayIndex)->size);
      if (bid != 0) {
        throw std::runtime_error("CameraBuffer must have binding 0");
      }
    }
  }

  mCameraSet.reset(); // TODO: manag
  mCameraPool = std::make_unique<core::DynamicDescriptorPool>(
      std::vector<vk::DescriptorPoolSize>{
          {vk::DescriptorType::eUniformBuffer, 1}}); // TODO adapt
  mCameraSet =
      mCameraPool->allocateSet(mShaderPackInstance->getCameraSetLayout());

  // write
  vk::DescriptorBufferInfo bufferInfo(mCameraBuffer->getVulkanBuffer(), 0,
                                      VK_WHOLE_SIZE);
  vk::WriteDescriptorSet write(mCameraSet.get(), 0, 0,
                               vk::DescriptorType::eUniformBuffer, {},
                               bufferInfo);
  mContext->getDevice().updateDescriptorSets(write, {});
}

void RTRenderer::render(scene::Camera &camera,
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

  prepareRender(camera);

  // mContext->getQueue().submit(mRenderCommandBuffer.get(), waitSemaphores,
  //                             waitStages, signalSemaphores, fence);
  mContext->getQueue().submitAndWait(mRenderCommandBuffer.get());
}

void RTRenderer::render(
    scene::Camera &camera,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &waitSemaphores,
    vk::ArrayProxyNoTemporaries<vk::PipelineStageFlags const> const
        &waitStageMasks,
    vk::ArrayProxyNoTemporaries<uint64_t const> const &waitValues,
    vk::ArrayProxyNoTemporaries<vk::Semaphore const> const &signalSemaphores,
    vk::ArrayProxyNoTemporaries<uint64_t const> const &signalValues) {
  if (!mContext->isVulkanAvailable()) {
    return;
  }
  if (!mScene) {
    throw std::runtime_error("setScene must be called before rendering");
  }

  prepareRender(camera);
  mContext->getQueue().submit(mRenderCommandBuffer.get(), waitSemaphores,
                              waitStageMasks, waitValues, signalSemaphores,
                              signalValues, {});
}

void RTRenderer::display(std::string const &imageName, vk::Image backBuffer,
                         vk::Format format, uint32_t width, uint32_t height,
                         std::vector<vk::Semaphore> const &waitSemaphores,
                         std::vector<vk::PipelineStageFlags> const &waitStages,
                         std::vector<vk::Semaphore> const &signalSemaphores,
                         vk::Fence fence) {
  if (!mContext->isPresentAvailable()) {
    throw std::runtime_error("Display failed: present is not enabled.");
  }

  if (!mDisplayCommandBuffer) {
    mDisplayCommandPool = mContext->createCommandPool();
    mDisplayCommandBuffer = mDisplayCommandPool->allocateCommandBuffer();
  }
  mDisplayCommandBuffer->begin(
      {vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  auto renderTarget = mRenderImages.at(imageName);
  auto targetFormat = renderTarget->getFormat();
  if (targetFormat != vk::Format::eR8G8B8A8Unorm &&
      targetFormat != vk::Format::eR32G32B32A32Sfloat) {
    throw std::runtime_error(
        "failed to display: only color textures are supported in display");
  };
  auto layout = vk::ImageLayout::eGeneral;
  vk::AccessFlags sourceAccessMask =
      vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
  vk::PipelineStageFlags sourceStage =
      vk::PipelineStageFlagBits::eRayTracingShaderKHR;

  renderTarget->getImage().transitionLayout(
      mDisplayCommandBuffer.get(), layout, vk::ImageLayout::eTransferSrcOptimal,
      sourceAccessMask, vk::AccessFlagBits::eTransferRead, sourceStage,
      vk::PipelineStageFlagBits::eTransfer);

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
  log::info("blit image {:x}", renderTarget->getImage().getVulkanImage());
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

  // transfer image layout back
  renderTarget->getImage().transitionLayout(
      mDisplayCommandBuffer.get(), vk::ImageLayout::eTransferSrcOptimal,
      vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferRead,
      vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eRayTracingShaderKHR);

  mDisplayCommandBuffer->end();
  mContext->getQueue().submit(mDisplayCommandBuffer.get(), waitSemaphores,
                              waitStages, signalSemaphores, fence);
}

std::vector<std::string> RTRenderer::getRenderTargetNames() const {
  std::vector<std::string> names;
  for (auto &[bid, binding] : mShaderPack->getOutputDescription().bindings) {
    if (binding.type == vk::DescriptorType::eStorageImage &&
        binding.name.starts_with("out")) {
      std::string texName = binding.name.substr(3);
      names.push_back(texName);
    }
  }
  return names;
}

} // namespace renderer
} // namespace svulkan2
