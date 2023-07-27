#include "camera_controller.hpp"
#include "svulkan2/common/fs.h"
#include "svulkan2/core/context.h"
#include "svulkan2/renderer/renderer.h"
#include "svulkan2/renderer/rt_renderer.h"
#include "svulkan2/scene/scene.h"
#include "svulkan2/shader/compute.h"
#include "svulkan2/shader/shader.h"
#include "svulkan2/ui/ui.h"
#include <GLFW/glfw3.h>
#include <chrono>
#include <iostream>

// clang-format off
#include <imgui.h>
#include <ImGuizmo.h>
#include <ImGuiFileDialog.h>
// clang-format on

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include <stb_image_write.h>

using namespace svulkan2;

static bool gSwapchainRebuild = true;
static bool gClosed = false;
static std::string srcBase = "";

static void glfw_resize_callback(GLFWwindow *, int w, int h) { gSwapchainRebuild = true; }

static void window_close_callback(GLFWwindow *window) {
  gClosed = true;
  glfwSetWindowShouldClose(window, GLFW_FALSE);
}

static void createSphereArray(svulkan2::scene::Scene &scene) {
  for (uint32_t i = 0; i < 10; ++i) {
    for (uint32_t j = 0; j < 10; ++j) {
      float metallic = i / 9.f;
      float roughness = j / 9.f;
      float specular = 0.5;
      auto shape = resource::SVShape::Create(
          resource::SVMesh::CreateUVSphere(32, 16),
          std::make_shared<resource::SVMetallicMaterial>(
              glm::vec4{0, 0, 0, 1}, glm::vec4{1, 1, 1, 1}, specular, roughness, metallic));
      auto &o =
          scene.addObject(resource::SVModel::FromData({shape}),
                          {.position = {i / 8.f, 0.2 + j / 8.f, 0}, .scale = {0.05, 0.05, 0.05}});

      // o.setShadeFlat(true);
      o.setSegmentation({0, 0, i, j});
    }
  }

  {
    auto shape = resource::SVShape::Create(
        resource::SVMesh::CreateCube(),
        std::make_shared<resource::SVMetallicMaterial>(glm::vec4{0, 0, 0, 1}, // emission
                                                       glm::vec4{1, 1, 1, 1}, // base color
                                                       0,                     // fresnel
                                                       0.0,                   // roughness
                                                       0.0,                   // metallic
                                                       0.9,                   // transmission
                                                       3.0,                   // ior
                                                       0.1 // transmission roughness
                                                       ));

    scene.addObject(resource::SVModel::FromData({shape}),
                    {.position = {0, 0.5, 0.3}, .scale = {0.1, 0.1, 0.02}});
  }

  {
    auto shape = resource::SVShape::Create(
        resource::SVMesh::CreateCube(),
        std::make_shared<resource::SVMetallicMaterial>(glm::vec4{0, 0, 0, 1}, // emission
                                                       glm::vec4{1, 1, 1, 1}, // base color
                                                       0,                     // fresnel
                                                       1.0, 0.0));
    scene.addObject(resource::SVModel::FromData({shape}),
                    {.position = {0, -0.1, 0}, .scale = {10, 0.1, 10}});
  }
}

static void setupScene(svulkan2::scene::Scene &scene) {
  {
    auto shape = resource::SVShape::Create(
        resource::SVMesh::CreateUVSphere(32, 16),
        std::make_shared<resource::SVMetallicMaterial>(glm::vec4{0, 0, 0, 1},
                                                       glm::vec4{1, 1, 1, 1}, 0.f, 1.f, 0.f));
    auto &o = scene.addObject(resource::SVModel::FromData({shape}),
                              {.position = {0.f, 0.15f, 0.f}, .scale = {0.05, 0.05, 0.05}});
  }

  {
    auto shape = resource::SVShape::Create(
        resource::SVMesh::CreateCube(),
        std::make_shared<resource::SVMetallicMaterial>(glm::vec4{0, 0, 0, 1},       // emission
                                                       glm::vec4{0.9, 0.9, 0.9, 1}, // base color
                                                       0,                           // fresnel
                                                       1.0, 0.0));
    scene.addObject(resource::SVModel::FromData({shape}),
                    {.position = {0, -0.1, 0}, .scale = {10, 0.1, 10}});
  }

  auto &light = scene.addParallelogramLight();
  light.setColor({10, 10, 10});
  light.setTransform(scene::Transform{.position = {0, 0.3f, 0.0}, .rotation = {1, 0, 0, 0}});
  light.setShape({0, 0, 0.1}, {0.1, 0, 0});
  // auto &light = scene.addPointLight();
  // light.setColor({1.0, 1.0, 1.0});
  // light.setPosition({0, 0, 0.4});
}

int main() {
  svulkan2::logger::setLogLevel("info");

  auto context = svulkan2::core::Context::Create(true, 5000, 5000, 4);
  auto manager = context->createResourceManager();

  // auto config = std::make_shared<RendererConfig>();
  // config->shaderDir = "../shader/ibl";
  // config->culling = vk::CullModeFlagBits::eNone;
  // config->msaa = vk::SampleCountFlagBits::e1;
  // config->colorFormat4 = vk::Format::eR32G32B32A32Sfloat;
  // renderer::Renderer renderer(config);

  renderer::RTRenderer renderer("../shader/rt");
  renderer.setCustomProperty("maxDepth", 6);
  renderer.setCustomProperty("russianRoulette", 1);
  renderer.setCustomProperty("russianRouletteMinBounces", 3);
  renderer.enableDenoiser("HdrColor", "Albedo", "Normal");

  auto scene = std::make_shared<svulkan2::scene::Scene>();

  // createSphereArray(*scene);

  setupScene(*scene);

  // {
  //   auto model =
  //       manager->CreateModelFromFile("../test/assets/scene/sponza/sponza.obj");
  //   auto &obj = scene.addObject(model);
  //   obj.setScale({0.001, 0.001, 0.001});
  // }

  // {
  //   auto model =
  //       manager->CreateModelFromFile("/home/fx/Downloads/sapien_cube.glb");
  //   auto &obj = scene.addObject(model);
  //   obj.setScale({0.1, 0.1, 0.1});
  //   obj.setPosition({0.5, 0.5, 0.5});
  // }
  // {
  //   auto model =
  //       manager->CreateModelFromFile("/home/fx/Downloads/sapien_cube_2.glb");
  //   auto &obj = scene.addObject(model);
  //   obj.setScale({0.05, 0.05, 0.05});
  //   obj.setPosition({-0.5, 0.5, 0.5});
  // }
  // {
  //   auto model =
  //       manager->CreateModelFromFile("/home/fx/Downloads/sapien_cube_2.glb");
  //   auto &obj = scene.addObject(model);
  //   obj.setScale({0.05, 0.05, 0.05});
  //   obj.setPosition({-1.0, 0.5, 0.5});
  //   obj.setRotation(glm::quat(0.9238795, 0.0, 0.3826834, 0.0));
  // }

  // auto model = manager->CreateModelFromFile(
  //     "/home/fx/source/sapien-project-web/storage/models/partnet/"
  //     "0af05cd6-8494-4454-b7d0-7f6a3cda41f1/model.gltf");
  // auto &obj = scene.addObject(model);
  // obj.setScale({0.1, 0.1, 0.1});
  // obj.setPosition({0.5, 0.5, 0.5});

  // auto &spotLight1 = scene.addSpotLight();
  // spotLight1.setPosition({0.5, 0.5, 1.0});
  // spotLight1.setDirection({0, 0, -1});
  // spotLight1.setFov(1);
  // spotLight1.setFovSmall(1);
  // spotLight1.setColor({4, 4, 4});
  // spotLight1.enableShadow(true);
  // spotLight1.setShadowParameters(0.05, 5, 2048);

  // auto &spotLight2 = scene.addSpotLight();
  // spotLight2.setPosition({0.5, 0.5, 0.7});
  // spotLight2.setDirection({0, 0, -1});
  // spotLight2.setFov(2);
  // spotLight2.setColor({4, 4, 4});
  // spotLight2.enableShadow(true);
  // spotLight2.setShadowParameters(0.05, 10, 2048);

  // auto &spotLight2 = scene.addTexturedLight();
  // spotLight2.setPosition({1, 0.5, 1});
  // spotLight2.setDirection({0, -1, 0});
  // spotLight2.setFov(1.5);
  // spotLight2.setFovSmall(1.5);
  // spotLight2.setColor({1, 0, 0});
  // spotLight2.enableShadow(true);
  // spotLight2.setShadowParameters(0.05, 10, 2048);
  // spotLight2.setTexture(context->getResourceManager()->CreateTextureFromFile(
  //     "../test/assets/image/flashlight.jpg", 1, vk::Filter::eLinear,
  //     vk::Filter::eLinear, vk::SamplerAddressMode::eClampToBorder,
  //     vk::SamplerAddressMode::eClampToBorder, true));

  // auto &pl = scene.addPointLight();
  // pl.setPosition({0.5, 0.5, 0.5});
  // pl.setColor({1, 1, 1});
  // pl.enableShadow(true);
  // pl.setShadowParameters(0.01, 10, 2048);

  // auto &dl = scene.addDirectionalLight();
  // dl.setPosition({0, 0, 0});
  // // dl.setDirection({-1, -5, -1});
  // dl.setDirection({0, -5, -5});
  // dl.setColor({1, 1, 1});
  // dl.enableShadow(true);
  // dl.setShadowParameters(-10, 10, 5, 2048);

  scene->setAmbientLight({0.5f, 0.5f, 0.5f, 0});

  // auto material =
  //     std::make_shared<resource::SVMetallicMaterial>(glm::vec4{1, 1, 1,
  //     1});
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

  // auto model = context->getResourceManager()->CreateModelFromFile(
  //     "/home/fx/blender-data/test_pbr.glb");
  // scene.addObject(model);

  // auto model = context->getResourceManager()->CreateModelFromFile(
  //     "/home/fx/datasets/RIS/objects/glass.gltf");
  // scene.addObject(model);

  // auto lineset = std::make_shared<resource::SVLineSet>();
  // lineset->setVertexAttribute(
  //     "position", {0.1, 0.1, 0.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 0.1,
  //     0.1});
  // lineset->setVertexAttribute("color", {
  //                                          1,
  //                                          0,
  //                                          0,
  //                                          1,
  //                                          1,
  //                                          0,
  //                                          0,
  //                                          1,
  //                                          1,
  //                                          0,
  //                                          0,
  //                                          1,
  //                                          1,
  //                                          0,
  //                                          0,
  //                                          1,
  //                                      });
  // scene.addLineObject(lineset).setPosition({0, 1, 0});

  std::vector<float> positions;
  std::vector<float> scales;
  std::vector<float> colors;
  float r = 0.01;
  for (uint32_t i = 0; i < 40; ++i) {
    for (uint32_t j = 0; j < 40; ++j) {
      for (uint32_t k = 0; k < 20; ++k) {
        positions.push_back(i * r);
        positions.push_back(j * r);
        positions.push_back(k * r);
        scales.push_back(r);
        colors.push_back(0);
        colors.push_back(1);
        colors.push_back(1);
        colors.push_back(1);
      }
    }
  }

  // auto pointset = std::make_shared<resource::SVPointSet>();
  // pointset->setVertexAttribute("position", positions);
  // pointset->setVertexAttribute("scale", scales);
  // pointset->setVertexAttribute("color", colors);
  // pointset->uploadToDevice();

  // scene.addPointObject(pointset).setShadingMode(0);

  auto &cameraNode = scene->addCamera();
  cameraNode.setPerspectiveParameters(0.05, 50, 1, 400, 300);
  FPSCameraController controller(cameraNode, {0, 0, -1}, {0, 1, 0});
  // controller.setXYZ(0, 0.5, 0);
  controller.setXYZ(0, 0.5, 3);

  // auto &customLight = scene.addCustomLight(cameraNode);
  // customLight.setTransform({.position = {0.1, 0, 0}});
  // customLight.setShadowProjectionMatrix(math::perspective(0.7f, 1.f,
  // 0.1f, 5.f));

  // auto flashlight = context->getResourceManager()->CreateTextureFromFile(
  //     "../test/assets/image/flashlight.jpg", 1);

  // auto &l = scene->addTexturedLight();
  // l.setColor({1, 1, 1});
  // l.setPosition({0.5, 0.5, 0.5});
  // l.enableShadow(true);
  // l.setFov(2);
  // l.setFovSmall(1);
  // l.setTexture(flashlight);
  // l.setDirection({0, -1, 0});
  // scene->updateModelMatrices();

  // renderer.setCustomTexture("BRDFLUT", lutTexture);

  // auto cubemap =
  //     context->getResourceManager()->CreateCubemapFromKTX("input.ktx", 5);
  // scene.setEnvironmentMap(cubemap);
  //

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

  // auto cubemap = context->getResourceManager()->CreateCubemapFromKTX(
  //     "../test/assets/image/cube2/out.ktx", 5);
  // renderer.setCustomCubemap("Environment", cubemap);
  // scene->setEnvironmentMap(cubemap);

  auto window = context->createWindow(1024, 1024);
  // glfwGetFramebufferSize(window->getGLFWWindow(), &gSwapchainResizeWidth,
  //                        &gSwapchainResizeHeight);
  // renderer.resize(gSwapchainResizeWidth, gSwapchainResizeHeight);

  window->initImgui();
  context->getDevice().waitIdle();
  vk::UniqueSemaphore sceneRenderSemaphore = context->getDevice().createSemaphoreUnique({});
  vk::UniqueFence sceneRenderFence =
      context->getDevice().createFenceUnique({vk::FenceCreateFlagBits::eSignaled});

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
  auto uiWindow =
      ui::Widget::Create<ui::Window>()->Size({400, 400})->Label("main window")->append(gizmo);

  renderer.setScene(scene);

  scene->updateModelMatrices();

  int count = 0;
  while (!window->isClosed()) {
    count += 1;
    // spot
    // / 50.f)});

    if (count == 3) {
      // auto &l = scene.addSpotLight();
      // l.setColor({1, 1, 1});
      // l.setPosition({0.5, 0.5, 0.5});
      // l.enableShadow(true);
      // l.setFov(2);
      // l.setFovSmall(1);
      // scene.updateModelMatrices();
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
      cameraNode.setPerspectiveParameters(0.05, 50, 1, window->getWidth(), window->getHeight());
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

    ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".cpp,.h,.hpp",
                                            ".");

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
    flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

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
      if (context->getDevice().waitForFences(sceneRenderFence.get(), VK_TRUE, UINT64_MAX) !=
          vk::Result::eSuccess) {
        throw std::runtime_error("Failed on wait for fence.");
      }
      context->getDevice().resetFences(sceneRenderFence.get());
    }

    std::async(std::launch::async, [&]() {
      {
        renderer.render(cameraNode, std::vector<vk::Semaphore>{}, {}, {}, {});

        auto imageAcquiredSemaphore = window->getImageAcquiredSemaphore();

        renderer.display("Color", window->getBackbuffer(), window->getBackBufferFormat(),
                         window->getWidth(), window->getHeight(), {imageAcquiredSemaphore},
                         {vk::PipelineStageFlagBits::eColorAttachmentOutput},
                         {sceneRenderSemaphore.get()}, {});
      }

      try {
        window->presentFrameWithImgui(sceneRenderSemaphore.get(), sceneRenderFence.get());
      } catch (vk::OutOfDateKHRError &e) {
        gSwapchainRebuild = true;
        context->getDevice().waitIdle();
      }
    }).get();

    // required since we only use 1 set of uniform buffers
    context->getDevice().waitIdle();

    // auto [buffer, size, _] = renderer.transferToBuffer("Color");
    // auto now1 = std::chrono::system_clock::now();
    // auto through_transfer = buffer->download<float>();
    // auto now2 = std::chrono::system_clock::now();
    // std::chrono::duration<double> dur = now2 - now1;
    // std::cout << "transfer" << dur.count() << std::endl;

    // // time
    // now1 = std::chrono::system_clock::now();
    // void *cudaPtr = buffer->getCudaPtr();
    // float *dest = new float[buffer->getSize() / 4];
    // cudaMemcpy(dest, cudaPtr, buffer->getSize(), cudaMemcpyDeviceToHost);
    // now2 = std::chrono::system_clock::now();
    // dur = now2 - now1;
    // std::cout << "cuda copy" << dur.count() << std::endl;
    // break;

    if (gClosed) {
      window->close();
      break;
    }
    if (window->isKeyDown("q")) {
      window->close();
      break;
    }

    bool update = false;
    float r = 1e-2;
    if (window->isMouseKeyDown(1)) {
      auto [x, y] = window->getMouseDelta();
      controller.rotate(0, -r * y, -r * x);
      update = true;
    }
    if (window->isKeyDown("w")) {
      controller.move(r, 0, 0);
      update = true;
    }
    if (window->isKeyDown("s")) {
      controller.move(-r, 0, 0);
      update = true;
    }
    if (window->isKeyDown("a")) {
      controller.move(0, r, 0);
      update = true;
    }
    if (window->isKeyDown("d")) {
      controller.move(0, -r, 0);
      update = true;
    }

    if (update) {
      scene->updateModelMatrices();
    }

    auto model = cameraNode.computeWorldModelMatrix();
    auto view = glm::affineInverse(model);
    auto proj = cameraNode.getProjectionMatrix();
    proj[1][1] *= -1;
    proj[2][1] *= -1;
    gizmo->setCameraParameters(view, proj);
  }

  context->getDevice().waitIdle();
  // logger::info("finish");

  return 0;
}
