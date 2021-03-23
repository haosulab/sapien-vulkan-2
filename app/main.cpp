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

  svulkan2::core::Context context(
      VK_API_VERSION_1_1, true, 5000, 5000, 4);

  auto config = std::make_shared<RendererConfig>();
  // config->shaderDir = "../shader/full_no_shadow";
  // config->shaderDir = "../shader/full";
  // config->shaderDir = "../shader/forward";
  config->shaderDir = "../shader/full2";
  config->colorFormat = vk::Format::eR8G8B8A8Unorm;
  renderer::Renderer renderer(context, config);

  svulkan2::scene::Scene scene;

  auto &pointLight = scene.addPointLight();
  pointLight.setTransform({.position = glm::vec4{0, 0.5, 0, 1}});
  pointLight.setColor({1, 0, 0, 1});
  pointLight.enableShadow(true);
  pointLight.setShadowParameters(0.05, 5);

  // auto &p2 = scene.addPointLight();
  // p2.setTransform({.position = glm::vec4{0.5, 0.5, 0, 1}});
  // p2.setColor({0, 1, 0, 1});
  // p2.enableShadow(true);
  // p2.setShadowParameters(0.05, 5);

  // auto &p3 = scene.addPointLight();
  // p3.setTransform({.position = glm::vec4{-0.5, 0.5, 0, 1}});
  // p3.setColor({0, 0, 1, 1});
  // p3.enableShadow(true);
  // p3.setShadowParameters(0.05, 5);

  auto &dl = scene.addDirectionalLight();
  dl.setTransform({.position = {0, 0, 0}});
  dl.setDirection({0, -5, -1});
  dl.setColor({1, 1, 1, 1});
  dl.enableShadow(true);
  dl.setShadowParameters(-5, 5, 3);

  // auto &dl2 = scene.addDirectionalLight();
  // dl2.setTransform({.position = {0, 0, 0}});
  // dl2.setDirection({1, -1, 0.1});
  // dl2.setColor({1, 0, 0, 1});
  // dl2.enableShadow(true);
  // dl2.setShadowParameters(-5, 5, 3);

  scene.setAmbientLight({0.7f, 0.7f, 0.7f, 0});

  auto material =
      std::make_shared<resource::SVMetallicMaterial>(glm::vec4{1, 1, 1, 1});
  auto shape = std::make_shared<resource::SVShape>();
  // shape->mesh = resource::SVMesh::createCapsule(0.1, 0.2, 32, 8);
  // shape->mesh = resource::SVMesh::createUVSphere(32, 16);
  // shape->mesh = resource::SVMesh::createCone(32);

  // shape->mesh = resource::SVMesh::CreateCube();
  // shape->material = material;
  // shape->mesh->exportToFile("cube.obj");
  // auto sphere = resource::SVModel::FromData({{shape}});
  // scene.addObject(sphere).setTransform(
  //     {.position = {0, 0.25, 0}, .scale = {0.1, 0.1, 0.1}});

  auto model = context.getResourceManager().CreateModelFromFile(
      "../test/assets/scene/sponza/sponza2.obj");
  scene.addObject(model).setTransform(
      scene::Transform{.scale = {0.001, 0.001, 0.001}});

  // auto dragon = context.getResourceManager().CreateModelFromFile(
  //     "../test/assets/scene/dragon/dragon.obj");
  // scene.addObject(dragon).setTransform(
  //     scene::Transform{.position = {0, 0.1, 0}, .scale = {0.3, 0.3, 0.3}});

  auto &cameraNode = scene.addCamera();
  cameraNode.setPerspectiveParameters(0.05, 10, 1, 4.f / 3);
  FPSCameraController controller(cameraNode, {0, 0, -1}, {0, 1, 0});
  controller.setXYZ(0, 0.5, 0);

  // auto &customLight = scene.addCustomLight(cameraNode);
  // customLight.setTransform({.position = {0.1, 0, 0}});
  // customLight.setShadowProjectionMatrix(glm::perspective(0.7f, 1.f, 0.1f, 5.f));

  // renderer.setCustomTexture("LightMap",
  //                           context.getResourceManager().CreateTextureFromFile(
  //                               "../test/assets/image/flashlight.jpg", 1));

  auto window = context.createWindow(1600, 1200);

  int width, height;
  glfwGetFramebufferSize(window->getGLFWWindow(), &width, &height);

  renderer.resize(width, height);

  window->initImgui();
  context.getDevice().waitIdle();
  vk::UniqueSemaphore sceneRenderSemaphore =
      context.getDevice().createSemaphoreUnique({});
  vk::UniqueFence sceneRenderFence = context.getDevice().createFenceUnique(
      {vk::FenceCreateFlagBits::eSignaled});
  auto commandBuffer = context.createCommandBuffer();

  glfwSetFramebufferSizeCallback(window->getGLFWWindow(), glfw_resize_callback);
  glfwSetWindowCloseCallback(window->getGLFWWindow(), window_close_callback);

  auto uiWindow =
      ui::Widget::Create<ui::Window>()
          ->Size({400, 400})
          ->Label("main window")
          ->append(ui::Widget::Create<ui::DisplayText>()->Text("Hello!"))
          ->append(ui::Widget::Create<ui::InputText>()->Label("Input##1"))
          ->append(ui::Widget::Create<ui::InputFloat>()->Label("Input##2"))
          ->append(ui::Widget::Create<ui::InputFloat2>()->Label("Input##3"))
          ->append(ui::Widget::Create<ui::InputFloat3>()->Label("Input##4"))
          ->append(ui::Widget::Create<ui::InputFloat4>()->Label("Input##5"))
          ->append(ui::Widget::Create<ui::SliderFloat>()
                       ->Label("SliderFloat")
                       ->Min(10)
                       ->Max(20)
                       ->Value(15))
          ->append(ui::Widget::Create<ui::SliderAngle>()
                       ->Label("SliderAngle")
                       ->Min(1)
                       ->Max(90)
                       ->Value(1))
          ->append(ui::Widget::Create<ui::Checkbox>()
                       ->Label("Checkbox")
                       ->Checked(true));

  int count = 0;
  while (!window->isClosed()) {
    count += 1;

    // float T = std::abs((count % 120 - 60) / 60.f) - 1e-3;
    // dragonObj.setTransparency(T);

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
    // float scaling = ImGui::GetWindowDpiScale();
    // // log::info("Window DPI scale: {}", scaling);
    // applyStyle(1.f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_MenuBar;
    flags |= ImGuiWindowFlags_NoDocking;
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::SetNextWindowBgAlpha(0.f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", 0, flags);
    ImGui::PopStyleVar();

    ImGui::DockSpace(ImGui::GetID("Dockspace"), ImVec2(0, 0),
                     ImGuiDockNodeFlags_PassthruCentralNode |
                         ImGuiDockNodeFlags_NoDockingInCentralNode);
    ImGui::End();
    ImGui::PopStyleVar();

    uiWindow->build();
    ImGui::ShowDemoWindow();

    ImGui::Render();

    // wait for previous frame to finish
    {
      if (context.getDevice().waitForFences(sceneRenderFence.get(), VK_TRUE,
                                            UINT64_MAX) !=
          vk::Result::eSuccess) {
        throw std::runtime_error("Failed on wait for fence.");
      }
      context.getDevice().resetFences(sceneRenderFence.get());
    }

    scene.updateModelMatrices();
    // draw
    {
      commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
      renderer.render(commandBuffer.get(), scene, cameraNode);
      renderer.display(commandBuffer.get(), "ColorSSR", window->getBackbuffer(),
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
    if (window->isKeyDown("q")) {
      window->close();
    }

    float r = 1e-2;
    if (window->isMouseKeyDown(1)) {
      auto [x, y] = window->getMouseDelta();
      controller.rotate(0, -r * y, -r * x);
    }
    if (window->isKeyDown("w")) {
      controller.move(r, 0, 0);
    }
    if (window->isKeyDown("s")) {
      controller.move(-r, 0, 0);
    }
    if (window->isKeyDown("a")) {
      controller.move(0, r, 0);
    }
    if (window->isKeyDown("d")) {
      controller.move(0, -r, 0);
    }
  }

  scene.clearNodes();
  model.reset();

  context.getDevice().waitIdle();
  log::info("finish");

  return 0;
}
