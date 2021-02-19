#include "svulkan2/common/fs.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include "svulkan2/renderer/renderer.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/shader.h"
#include "svulkan2/ui/ui.h"
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "camera_controller.hpp"
#include <stb_image_write.h>

static bool gSwapchainRebuild = true;
static int gSwapchainResizeWidth = 800;
static int gSwapchainResizeHeight = 600;
static bool gClosed = false;

static void glfw_resize_callback(GLFWwindow *, int w, int h) {
  gSwapchainRebuild = true;
  gSwapchainResizeWidth = w;
  gSwapchainResizeHeight = h;
}

static void window_close_callback(GLFWwindow *window) {
  gClosed = true;
  glfwSetWindowShouldClose(window, GLFW_FALSE);
}

using namespace svulkan2;

int main() {
  svulkan2::log::getLogger()->set_level(spdlog::level::info);

  svulkan2::core::Context context(VK_API_VERSION_1_1, true, 5000, 5000, 4);

  auto config = std::make_shared<RendererConfig>();
  config->shaderDir = "../test/assets/shader/full";
  config->colorFormat = vk::Format::eR8G8B8A8Unorm;
  renderer::Renderer renderer(context, config);

  svulkan2::scene::Scene scene;

  scene.setSVScene(std::make_shared<resource::SVScene>());
  scene.getSVScene()->addPointLight({{0, 0.5, 0, 1}, {0, 0, 1, 1}});
  scene.getSVScene()->addDirectionalLight({{0, -1, -1, 1}, {1, 1, 1, 1}});
  scene.getSVScene()->setAmbeintLight({0.1, 0.1, 0.1, 0});

  auto model = context.getResourceManager().CreateModelFromFile(
      "../test/assets/scene/sponza/sponza.obj");
  scene.addObject(model).setTransform(
      scene::Transform{.scale = {0.001, 0.001, 0.001}});

  auto &cameraNode = scene.addCamera();
  cameraNode.setPerspectiveParameters(0.05, 10, 1, 4.f / 3);
  FPSCameraController controller(cameraNode, {0, 0, -1}, {0, 1, 0});
  controller.setXYZ(0, 0.5, 0);

  renderer.resize(800, 600);

  auto window = context.createWindow(800, 600);
  window->initImgui();
  context.getDevice().waitIdle();
  vk::UniqueSemaphore sceneRenderSemaphore =
      context.getDevice().createSemaphoreUnique({});
  vk::UniqueFence sceneRenderFence = context.getDevice().createFenceUnique(
      {vk::FenceCreateFlagBits::eSignaled});
  auto commandBuffer = context.createCommandBuffer();

  glfwSetWindowSizeCallback(window->getGLFWWindow(), glfw_resize_callback);
  glfwSetWindowCloseCallback(window->getGLFWWindow(), window_close_callback);

  auto uiWindow =
      ui::Widget::Create<ui::Window>()
          ->Size(400, 400)
          ->Name("main window")
          ->append(ui::Widget::Create<ui::DisplayText>()->Text("Hello!"))
          ->append(ui::Widget::Create<ui::InputText>()->Label("Input##1"))
          ->append(ui::Widget::Create<ui::InputFloat>()->Label("Input##2"))
          ->append(ui::Widget::Create<ui::InputFloat2>()->Label("Input##3"))
          ->append(ui::Widget::Create<ui::InputFloat3>()->Label("Input##4"))
          ->append(ui::Widget::Create<ui::InputFloat4>()->Label("Input##5"));

  model->loadAsync().get();
  model->getShapes()[0]->material->uploadToDevice(context);

  while (!window->isClosed()) {
    if (gSwapchainRebuild) {
      gSwapchainRebuild = false;
      context.getDevice().waitIdle();
      window->updateSize(gSwapchainResizeWidth, gSwapchainResizeHeight);
      renderer.resize(window->getWidth(), window->getHeight());
      context.getDevice().waitIdle();
      cameraNode.setPerspectiveParameters(
          0.05, 10, 1,
          static_cast<float>(window->getWidth()) / window->getHeight());
      continue;
    }
    try {
      window->newFrame();
    } catch (vk::OutOfDateKHRError &e) {
      gSwapchainRebuild = true;
      context.getDevice().waitIdle();
      continue;
    }

    ImGui::NewFrame();

    uiWindow->build();
    ImGui::ShowDemoWindow();

    ImGui::Render();

    // wait for previous frame to finish
    {
      context.getDevice().waitForFences(sceneRenderFence.get(), VK_TRUE,
                                        UINT64_MAX);
      context.getDevice().resetFences(sceneRenderFence.get());
    }
    // draw
    {
      commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
      renderer.render(commandBuffer.get(), scene, cameraNode);
      renderer.display(commandBuffer.get(), "Color", window->getBackbuffer(),
                       window->getBackBufferFormat(), window->getWidth(),
                       window->getHeight());
      commandBuffer->end();

      auto imageAcquiredSemaphore = window->getImageAcquiredSemaphore();
      vk::PipelineStageFlags waitStage =
          vk::PipelineStageFlagBits::eColorAttachmentOutput;
      vk::SubmitInfo info(1, &imageAcquiredSemaphore, &waitStage, 1,
                          &commandBuffer.get(), 1, &sceneRenderSemaphore.get());
      context.getQueue().submit(info, {});
    }

    auto swapchain = window->getSwapchain();
    auto fidx = window->getFrameIndex();
    vk::PresentInfoKHR info(1, &sceneRenderSemaphore.get(), 1, &swapchain,
                            &fidx);
    try {
      window->presentFrameWithImgui(sceneRenderSemaphore.get(),
                                    sceneRenderFence.get());
    } catch (vk::OutOfDateKHRError &e) {
      gSwapchainRebuild = true;
    }
    // required since we only use 1 set of uniform buffers
    context.getDevice().waitIdle();
    // break;
    if (gClosed) {
      window->close();
    }
    if (window->isKeyDown('q')) {
      window->close();
    }

    if (window->isMouseKeyDown(1)) {
      auto [x, y] = window->getMouseDelta();
      float r = 1e-3;
      controller.rotate(0, -r * y, -r * x);
    }

    constexpr float r = 1e-3;
    if (window->isKeyDown('w')) {
      controller.move(r, 0, 0);
    }
    if (window->isKeyDown('s')) {
      controller.move(-r, 0, 0);
    }
    if (window->isKeyDown('a')) {
      controller.move(0, r, 0);
    }
    if (window->isKeyDown('d')) {
      controller.move(0, -r, 0);
    }
  }

  scene.clearNodes();
  model.reset();

  context.getDevice().waitIdle();
  log::info("finish");

  return 0;
}
