#include "svulkan2/common/fs.h"
#include "svulkan2/common/log.h"
#include "svulkan2/core/context.h"
#include "svulkan2/renderer/renderer.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/compute.h"
#include "svulkan2/shader/shader.h"
#include "svulkan2/ui/ui.h"
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "camera_controller.hpp"
#include <stb_image_write.h>

using namespace svulkan2;

static bool gSwapchainRebuild = true;
static bool gClosed = false;
static std::string srcBase = "";

static void glfw_resize_callback(GLFWwindow *, int w, int h) {
  gSwapchainRebuild = true;
}

static void window_close_callback(GLFWwindow *window) {
  gClosed = true;
  glfwSetWindowShouldClose(window, GLFW_FALSE);
}

static void createSphereArray(svulkan2::scene::Scene &scene) {
  for (uint32_t i = 0; i < 10; ++i) {
    for (uint32_t j = 0; j < 2; ++j) {
      float metallic = i / 9.f;
      float roughness = j / 9.f;
      auto shape = resource::SVShape::Create(
          resource::SVMesh::CreateUVSphere(32, 16),
          std::make_shared<resource::SVMetallicMaterial>(
              glm::vec4{0, 0, 0, 1}, glm::vec4{1, 1, 1, 1}, 0, roughness,
              metallic));
      scene.addObject(
          resource::SVModel::FromData({shape}),
          {.position = {i / 8.f, j / 8.f, 0}, .scale = {0.05, 0.05, 0.05}});
    }
  }

  {
    auto shape = resource::SVShape::Create(
        resource::SVMesh::CreateCube(),
        std::make_shared<resource::SVMetallicMaterial>(
            glm::vec4{0, 0, 0, 1}, glm::vec4{1, 1, 1, 1}, 0, 1, 0));
    scene.addObject(resource::SVModel::FromData({shape}),
                    {.position = {0, -0.1, 0}, .scale = {10, 0.1, 10}});
  }
}

int main() {
  svulkan2::log::getLogger()->set_level(spdlog::level::info);

  auto context = svulkan2::core::Context::Create(true, 5000, 5000, 4);
  auto manager = context->createResourceManager();

  if (srcBase.length() == 0) {
    std::cout << "Using default srcBase" << std::endl;
    srcBase = "../";
  }

  auto config = std::make_shared<RendererConfig>();
  config->shaderDir = srcBase + "shader/ibl";

  config->colorFormat = vk::Format::eR32G32B32A32Sfloat;
  renderer::Renderer renderer(context, config);

  // auto image = shader::generateBRDFLUT(context, 512);
  // auto sampler =
  // context->getDevice().createSamplerUnique(vk::SamplerCreateInfo(
  //     {}, vk::Filter::eLinear, vk::Filter::eLinear,
  //     vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
  //     vk::SamplerAddressMode::eClampToEdge,
  //     vk::SamplerAddressMode::eClampToEdge, 0.f, false, 0.f, false,
  //     vk::CompareOp::eNever, 0.f, 0.f, vk::BorderColor::eFloatOpaqueWhite));
  // auto view =
  //     context->getDevice().createImageViewUnique(vk::ImageViewCreateInfo(
  //         {}, image->getVulkanImage(), vk::ImageViewType::e2D,
  //         image->getFormat(), vk::ComponentSwizzle::eIdentity,
  //         vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
  //                                   1)));
  // auto lutTexture = resource::SVTexture::FromImage(
  //     resource::SVImage::FromDeviceImage(std::move(image)), std::move(view),
  //     std::move(sampler));

  svulkan2::scene::Scene scene;

  // createSphereArray(scene);

  auto &spotLight1 = scene.addSpotLight();
  spotLight1.setPosition({0, 0.5, 0});
  spotLight1.setDirection({1, 0, 0});
  spotLight1.setFov(1);
  spotLight1.setColor({1, 0, 0});
  spotLight1.enableShadow(true);
  spotLight1.setShadowParameters(0.05, 5);

  auto &spotLight2 = scene.addSpotLight();
  spotLight2.setPosition({0.5, 0.5, 0.7});
  spotLight2.setDirection({0, 0, -1});
  spotLight2.setFov(2);
  spotLight2.setColor({4, 4, 4});
  spotLight2.enableShadow(true);
  spotLight2.setShadowParameters(0.05, 10);

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

  // auto &dl = scene.addDirectionalLight();
  // dl.setPosition({0, 0, 0});
  // dl.setDirection({-1, -5, -1});
  // dl.setColor({1, 1, 1, 1});
  // dl.enableShadow(true);
  // dl.setShadowParameters(-10, 10, 5);

  auto &dl2 = scene.addDirectionalLight();
  dl2.setTransform({.position = {0, 0, 0}});
  dl2.setDirection({1, -1, 0.1});
  dl2.setColor({1, 0, 0});
  dl2.enableShadow(true);
  dl2.setShadowParameters(-5, 5, 3);

  scene.setAmbientLight({0.f, 0.f, 0.f, 0});

  // auto material =
  //     std::make_shared<resource::SVMetallicMaterial>(glm::vec4{1, 1, 1, 1});
  // auto shape = std::make_shared<resource::SVShape>();
  // shape->mesh = resource::SVMesh::createCapsule(0.1, 0.2, 32, 8);
  // shape->mesh = resource::SVMesh::createUVSphere(32, 16);
  // shape->mesh = resource::SVMesh::createCone(32);

  // shape->mesh = resource::SVMesh::CreateCube();
  // shape->material = material;
  // shape->mesh->exportToFile("cube.obj");
  // auto sphere = resource::SVModel::FromData({{shape}});
  // scene.addObject(sphere).setTransform(
  //     {.position = {0, 0.25, 0}, .scale = {0.1, 0.1, 0.1}});

  // auto model = context->getResourceManager()->CreateModelFromFile(
  //     srcBase + "test/assets/scene/sponza/sponza2.obj");
  // scene.addObject(model).setTransform(
  //     scene::Transform{.scale = {0.001, 0.001, 0.001}});

  auto model = context->getResourceManager()->CreateModelFromFile(
      "/home/fx/blender-data/test_pbr.glb");
  scene.addObject(model);

  // auto model = context->getResourceManager()->CreateModelFromFile(
  //     "/home/fx/datasets/RIS/objects/glass.gltf");
  // scene.addObject(model);

  auto lineset = std::make_shared<resource::SVLineSet>();
  lineset->setVertexAttribute(
      "position", {0.1, 0.1, 0.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 0.1, 0.1});
  lineset->setVertexAttribute("color", {
                                           1,
                                           0,
                                           0,
                                           1,
                                           1,
                                           0,
                                           0,
                                           1,
                                           1,
                                           0,
                                           0,
                                           1,
                                           1,
                                           0,
                                           0,
                                           1,
                                       });
  scene.addLineObject(lineset).setPosition({0, 1, 0});

  auto pointset = std::make_shared<resource::SVPointSet>();
  pointset->setVertexAttribute("position",
                               {0.1, 0.1, 0.1, 1.1, 1.1, 1.1, 1.1, 0.1, 0.1});
  pointset->setVertexAttribute("color", {
                                            0,
                                            1,
                                            0,
                                            1,
                                            0,
                                            1,
                                            0,
                                            1,
                                            0,
                                            1,
                                            0,
                                            1,
                                        });
  scene.addPointObject(pointset);

  auto &cameraNode = scene.addCamera();
  cameraNode.setPerspectiveParameters(0.05, 50, 1, 400, 300);
  FPSCameraController controller(cameraNode, {0, 0, -1}, {0, 1, 0});
  controller.setXYZ(0, 0.5, 3);

  // auto &customLight = scene.addCustomLight(cameraNode);
  // customLight.setTransform({.position = {0.1, 0, 0}});
  // customLight.setShadowProjectionMatrix(math::perspective(0.7f, 1.f,
  // 0.1f, 5.f));

  // renderer.setCustomTexture("LightMap",
  //                           context.getResourceManager().CreateTextureFromFile(
  //                               "../test/assets/image/flashlight.jpg", 1));

  // renderer.setCustomTexture("BRDFLUT", lutTexture);

  auto cubemap =
      context->getResourceManager()->CreateCubemapFromKTX("input.ktx", 5);
  scene.setEnvironmentMap(cubemap);

  // auto cubemap = context->getResourceManager()->CreateCubemapFromFiles(
  //     {
  //         "../test/assets/image/cube2/px.png",
  //         "../test/assets/image/cube2/nx.png",
  //         "../test/assets/image/cube2/py.png",
  //         "../test/assets/image/cube2/ny.png",
  //         "../test/assets/image/cube2/pz.png",
  //         "../test/assets/image/cube2/nz.png",
  //     },
  //     5);
  // cubemap->load();
  // cubemap->uploadToDevice(context);
  // scene.setEnvironmentMap(cubemap);

  // cubemap->exportKTX("output.ktx");

  // renderer.setCustomCubemap("Environment", cubemap);

  auto window = context->createWindow(1024, 1024);
  // glfwGetFramebufferSize(window->getGLFWWindow(), &gSwapchainResizeWidth,
  //                        &gSwapchainResizeHeight);
  // renderer.resize(gSwapchainResizeWidth, gSwapchainResizeHeight);

  window->initImgui();
  context->getDevice().waitIdle();
  vk::UniqueSemaphore sceneRenderSemaphore =
      context->getDevice().createSemaphoreUnique({});
  vk::UniqueFence sceneRenderFence = context->getDevice().createFenceUnique(
      {vk::FenceCreateFlagBits::eSignaled});
  auto commandBuffer = context->createCommandBuffer();

  glfwSetFramebufferSizeCallback(window->getGLFWWindow(), glfw_resize_callback);
  glfwSetWindowCloseCallback(window->getGLFWWindow(), window_close_callback);

  // auto uiWindow =
  //     ui::Widget::Create<ui::Window>()
  //         ->Size({400, 400})
  //         ->Label("main window")
  //         ->append(ui::Widget::Create<ui::DisplayText>()->Text("Hello!"))
  //         ->append(ui::Widget::Create<ui::InputText>()->Label("Input##1"))
  //         ->append(ui::Widget::Create<ui::InputFloat>()->Label("Input##2"))
  //         ->append(ui::Widget::Create<ui::InputFloat2>()->Label("Input##3"))
  //         ->append(ui::Widget::Create<ui::InputFloat3>()->Label("Input##4"))
  //         ->append(ui::Widget::Create<ui::InputFloat4>()->Label("Input##5"))
  //         ->append(ui::Widget::Create<ui::SliderFloat>()
  //                      ->Label("SliderFloat")
  //                      ->Min(10)
  //                      ->Max(20)
  //                      ->Value(15))
  //         ->append(ui::Widget::Create<ui::SliderAngle>()
  //                      ->Label("SliderAngle")
  //                      ->Min(1)
  //                      ->Max(90)
  //                      ->Value(1))
  //         ->append(ui::Widget::Create<ui::Checkbox>()
  //                      ->Label("Checkbox")
  //                      ->Checked(true));

  auto gizmo = ui::Widget::Create<ui::Gizmo>()->Matrix(glm::mat4(1));
  auto uiWindow = ui::Widget::Create<ui::Window>()
                      ->Size({400, 400})
                      ->Label("main window")
                      ->append(gizmo);

  renderer.setScene(scene);

  int count = 0;
  while (!window->isClosed()) {
    count += 1;
    // spotLight.setDirection({glm::cos(count / 50.f), 0, glm::sin(count
    // / 50.f)});
    if (count == 120) {
      auto &l = scene.addSpotLight();
      l.enableShadow(true);
      scene.removeNode(l);
    }

    if (gSwapchainRebuild) {
      context->getDevice().waitIdle();
      int width, height;
      glfwGetFramebufferSize(window->getGLFWWindow(), &width, &height);
      if (!window->updateSize(width, height)) {
        continue;
      }
      gSwapchainRebuild = false;
      renderer.resize(width, height);
      context->getDevice().waitIdle();
      cameraNode.setPerspectiveParameters(0.05, 50, 1, window->getWidth(),
                                          window->getHeight());
      continue;
    }
    try {
      window->newFrame();
    } catch (vk::OutOfDateKHRError &e) {
      gSwapchainRebuild = true;
      context->getDevice().waitIdle();
      continue;
    }

    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

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
      if (context->getDevice().waitForFences(sceneRenderFence.get(), VK_TRUE,
                                             UINT64_MAX) !=
          vk::Result::eSuccess) {
        throw std::runtime_error("Failed on wait for fence.");
      }
      context->getDevice().resetFences(sceneRenderFence.get());
    }

    scene.updateModelMatrices();
    // draw
    {
      renderer.render(cameraNode, {}, {}, {}, {});
      auto imageAcquiredSemaphore = window->getImageAcquiredSemaphore();
      renderer.display("Color", window->getBackbuffer(),
                       window->getBackBufferFormat(), window->getWidth(),
                       window->getHeight(), {imageAcquiredSemaphore},
                       {vk::PipelineStageFlagBits::eColorAttachmentOutput},
                       {sceneRenderSemaphore.get()}, {});
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
      context->getDevice().waitIdle();
      continue;
    }
    // required since we only use 1 set of uniform buffers
    context->getDevice().waitIdle();
    // auto cuda = renderer.transferToCuda("Color");
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

    auto model = cameraNode.computeWorldModelMatrix();
    auto view = glm::affineInverse(model);
    auto proj = cameraNode.getProjectionMatrix();
    proj[1][1] *= -1;
    proj[2][1] *= -1;
    gizmo->setCameraParameters(view, proj);
  }

  context->getDevice().waitIdle();
  log::info("finish");

  return 0;
}
