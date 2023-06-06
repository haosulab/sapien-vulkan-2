#include "svulkan2/shader/shader_pack.h"
#include "../common/logger.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/shader/primitive.h"
#include "svulkan2/shader/primitive_shadow.h"
#include "svulkan2/shader/shadow.h"
#include <filesystem>
#include <set>

namespace fs = std::filesystem;
namespace svulkan2 {
namespace shader {

ShaderPack::ShaderPack(std::string const &dirname) {
  fs::path path(dirname);
  check_dir_required(path, "[shader pack] " + dirname + " is not a directory");
  check_file_required(path / "gbuffer.vert", "[shader pack] gbuffer.vert is required");
  check_file_required(path / "gbuffer.frag", "[shader pack] gbuffer.frag is required");

  // GLSL compiler starts

  std::vector<std::future<void>> futures;

  // shadow pass
  if (check_file(path / "shadow.vert")) {
    mShadowPass = std::make_shared<ShadowPassParser>();
    mShadowPass->setName("Shadow");
    futures.push_back(mShadowPass->loadGLSLFilesAsync((path / "shadow.vert").string(), ""));
  }

  // point shadow pass
  if (check_file(path / "shadow_point.vert")) {
    mPointShadowPass = std::make_shared<PointShadowParser>();
    mPointShadowPass->setName("PointShadow");

    auto vsFile = path / "shadow_point.vert";
    auto fsFile = path / "shadow_point.frag";
    auto gsFile = path / "shadow_point.geom";
    fsFile = check_file(fsFile) ? fsFile : "";
    gsFile = check_file(gsFile) ? gsFile : "";
    futures.push_back(
        mPointShadowPass->loadGLSLFilesAsync(vsFile.string(), fsFile.string(), gsFile.string()));
  }

  mHasDeferred = check_file(path / "deferred.vert") && check_file(path / "deferred.frag");

  mNonShadowPasses = {};
  mGbufferPasses = {};
  mPointPasses = {};
  mLinePasses = {};

  int numGbufferPasses = 0;
  int numPointPasses = 0;
  for (const auto &entry : fs::directory_iterator(path)) {
    std::string filename = entry.path().filename().string();
    if (filename.starts_with("gbuffer") && filename.ends_with(".frag")) {
      numGbufferPasses++;
    }
    if (filename.starts_with("point") && filename.ends_with(".frag")) {
      numPointPasses++;
    }
  }

  // first gbuffer pass
  std::shared_ptr<GbufferPassParser> firstGbufferPass = std::make_shared<GbufferPassParser>();
  {
    firstGbufferPass->setName("Gbuffer");
    mNonShadowPasses.push_back(firstGbufferPass);
    mGbufferPasses.push_back(firstGbufferPass);
    firstGbufferPass->setIndex(mNonShadowPasses.size() - 1);
    std::string vsFile = (path / ("gbuffer.vert")).string();
    std::string fsFile = (path / ("gbuffer.frag")).string();
    futures.push_back(firstGbufferPass->loadGLSLFilesAsync(vsFile, fsFile));
  }

  // ao pass
  {
    std::string vsFile = (path / "ao.vert").string();
    std::string fsFile = (path / "ao.frag").string();
    if (check_file(vsFile) && check_file(fsFile)) {
      auto aoPass = std::make_shared<DeferredPassParser>();
      aoPass->setName("AO");
      mNonShadowPasses.push_back(aoPass);
      aoPass->setIndex(mNonShadowPasses.size() - 1);
      futures.push_back(aoPass->loadGLSLFilesAsync(vsFile, fsFile));
    }
  }

  // deferred pass
  std::string vsFile = (path / "deferred.vert").string();
  std::string fsFile = (path / "deferred.frag").string();
  if (check_file(vsFile) && check_file(fsFile)) {
    auto deferredPass = std::make_shared<DeferredPassParser>();
    deferredPass->setName("Deferred");
    mNonShadowPasses.push_back(deferredPass);
    deferredPass->setIndex(mNonShadowPasses.size() - 1);
    futures.push_back(deferredPass->loadGLSLFilesAsync(vsFile, fsFile));
  }

  // gbuffer passes
  for (int i = 1; i < numGbufferPasses; ++i) {
    std::string suffix = std::to_string(i);
    auto gbufferPass = std::make_shared<GbufferPassParser>();

    gbufferPass->setName("Gbuffer" + suffix);
    mNonShadowPasses.push_back(gbufferPass);
    mGbufferPasses.push_back(gbufferPass);
    gbufferPass->setIndex(mNonShadowPasses.size() - 1);

    std::string vsFile = (path / ("gbuffer" + suffix + ".vert")).string();
    std::string fsFile = (path / ("gbuffer" + suffix + ".frag")).string();
    futures.push_back(gbufferPass->loadGLSLFilesAsync(vsFile, fsFile));
  }

  // point pass
  std::shared_ptr<PointPassParser> firstPointPass{};
  for (int i = 0; i < numPointPasses; ++i) {
    std::string suffix = i == 0 ? "" : std::to_string(i);
    auto pointPass = std::make_shared<PointPassParser>();
    if (i == 0) {
      firstPointPass = pointPass;
    }
    pointPass->setName("Point" + suffix);
    mNonShadowPasses.push_back(pointPass);
    mPointPasses.push_back(pointPass);
    pointPass->setIndex(mNonShadowPasses.size() - 1);

    std::string vsFile = (path / ("point" + suffix + ".vert")).string();
    std::string fsFile = (path / ("point" + suffix + ".frag")).string();
    std::string gsFile = (path / ("point" + suffix + ".geom")).string();
    gsFile = check_file(gsFile) ? gsFile : "";

    futures.push_back(pointPass->loadGLSLFilesAsync(vsFile, fsFile, gsFile));
  }

  std::shared_ptr<LinePassParser> linePass{};
  // line pass
  {
    std::string vsFile = (path / "line.vert").string();
    std::string fsFile = (path / "line.frag").string();
    if (check_file(vsFile) && check_file(fsFile)) {
      linePass = std::make_shared<LinePassParser>();
      linePass->setName("Line");
      linePass->setLineWidth(2.f);
      mNonShadowPasses.push_back(linePass);
      mLinePasses.push_back(linePass);
      linePass->setIndex(mNonShadowPasses.size() - 1);
      futures.push_back(linePass->loadGLSLFilesAsync(vsFile, fsFile));
    }
  }

  int numCompositePasses = 0;
  for (const auto &entry : fs::directory_iterator(path)) {
    std::string filename = entry.path().filename().string();
    if (filename.starts_with("composite") && filename.ends_with(".frag"))
      numCompositePasses++;
  }

  vsFile = (path / "composite.vert").string();
  for (int i = 0; i < numCompositePasses; i++) {
    fsFile = (path / ("composite" + std::to_string(i) + ".frag")).string();
    auto compositePass = std::make_shared<DeferredPassParser>();
    compositePass->setName("Composite" + std::to_string(i));
    mNonShadowPasses.push_back(compositePass);
    compositePass->setIndex(mNonShadowPasses.size() - 1);
    futures.push_back(compositePass->loadGLSLFilesAsync(vsFile, fsFile));
  }
  for (auto &f : futures) {
    f.get();
  }
  // GLSL compiler ends

  mShaderInputLayouts = generateShaderInputLayouts();
}

std::shared_ptr<ShaderConfig> ShaderPack::generateShaderInputLayouts() const {
  DescriptorSetDescription cameraSetDesc;
  DescriptorSetDescription objectSetDesc;
  DescriptorSetDescription sceneSetDesc;
  DescriptorSetDescription lightSetDesc;

  auto layouts = std::make_shared<ShaderConfig>();

  ASSERT(mGbufferPasses.size() > 0, "There must be at least 1 gbuffer pass in a shader pack.");
  layouts->vertexLayout = mGbufferPasses.at(0)->getVertexInputLayout();

  if (mLinePasses.size()) {
    layouts->primitiveVertexLayout = mLinePasses.at(0)->getVertexInputLayout();
  }

  if (mPointPasses.size()) {
    if (layouts->primitiveVertexLayout) {
      ASSERT(*layouts->primitiveVertexLayout == *mPointPasses.at(0)->getVertexInputLayout(),
             "All primitive passes must share the same vertex layout");
    } else {
      layouts->primitiveVertexLayout = mPointPasses.at(0)->getVertexInputLayout();
    }
  }

  std::vector<std::shared_ptr<BaseParser>> passes = mNonShadowPasses;
  if (mShadowPass) {
    passes.push_back(mShadowPass);
  }
  if (mPointShadowPass) {
    passes.push_back(mPointShadowPass);
  }

  for (auto &pass : passes) {
    auto descs = pass->getDescriptorSetDescriptions();
    for (auto &desc : descs) {
      switch (desc.type) {
      case UniformBindingType::eCamera:
        if (cameraSetDesc.type == UniformBindingType::eUnknown) {
          cameraSetDesc = desc;
        } else {
          cameraSetDesc = cameraSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eObject:
        if (objectSetDesc.type == UniformBindingType::eUnknown) {
          objectSetDesc = desc;
        } else {
          objectSetDesc = objectSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eScene:
        if (sceneSetDesc.type == UniformBindingType::eUnknown) {
          sceneSetDesc = desc;
        } else {
          sceneSetDesc = sceneSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eLight:
        if (lightSetDesc.type == UniformBindingType::eUnknown) {
          lightSetDesc = desc;
        } else {
          lightSetDesc = lightSetDesc.merge(desc);
        }
        break;
      case UniformBindingType::eMaterial:
      case UniformBindingType::eTextures:
        break;
      case UniformBindingType::eNone:
      case UniformBindingType::eUnknown:
      default:
        throw std::runtime_error("invalid descriptor set");
      }
    }
  }

  layouts->cameraBufferLayout = cameraSetDesc.buffers.at(cameraSetDesc.bindings.at(0).arrayIndex);
  layouts->objectBufferLayout = objectSetDesc.buffers.at(objectSetDesc.bindings.at(0).arrayIndex);
  layouts->sceneBufferLayout = sceneSetDesc.buffers.at(sceneSetDesc.bindings.at(0).arrayIndex);

  if (mShadowPass) {
    layouts->lightBufferLayout = lightSetDesc.buffers.at(lightSetDesc.bindings.at(0).arrayIndex);
    for (auto &binding : sceneSetDesc.bindings) {
      if (binding.second.name == "ShadowBuffer") {
        layouts->shadowBufferLayout = sceneSetDesc.buffers.at(binding.second.arrayIndex);
        break;
      }
    }
    if (!layouts->shadowBufferLayout) {
      throw std::runtime_error("Scene must declare ShadowBuffer");
    }
  }

  layouts->sceneSetDescription = sceneSetDesc;
  layouts->cameraSetDescription = cameraSetDesc;
  layouts->objectSetDescription = objectSetDesc;
  layouts->lightSetDescription = lightSetDesc;

  return layouts;
}

} // namespace shader
} // namespace svulkan2
