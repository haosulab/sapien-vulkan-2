#include "svulkan2/shader/shader_pack.h"
#include "../common/logger.h"
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
  if (!fs::is_directory(path)) {
    throw std::runtime_error("[shader pack] " + dirname +
                             " is not a directory");
  }

  if (!fs::is_regular_file(path / "gbuffer.vert")) {
    throw std::runtime_error("[shader pack] gbuffer.vert is required");
  }
  if (!fs::is_regular_file(path / "gbuffer.frag")) {
    throw std::runtime_error("[shader pack] gbuffer.frag is required");
  }

  // GLSL compiler starts

  std::vector<std::future<void>> futures;

  // shadow pass
  if (fs::is_regular_file(path / "shadow.vert")) {
    mShadowPass = std::make_shared<ShadowPassParser>();
    mShadowPass->setName("Shadow");
    futures.push_back(
        mShadowPass->loadGLSLFilesAsync((path / "shadow.vert").string(), ""));
  }

  // point shadow pass
  if (fs::is_regular_file(path / "shadow_point.vert")) {
    mPointShadowPass = std::make_shared<PointShadowParser>();
    mPointShadowPass->setName("PointShadow");

    auto vsFile = path / "shadow_point.vert";
    auto fsFile = path / "shadow_point.frag";
    auto gsFile = path / "shadow_point.geom";
    if (!fs::is_regular_file(fsFile)) {
      fsFile = "";
    }
    if (!fs::is_regular_file(gsFile)) {
      gsFile = "";
    }
    futures.push_back(
        mPointShadowPass->loadGLSLFilesAsync(vsFile.string(), fsFile.string(), gsFile.string()));
  }

  mHasDeferred = fs::is_regular_file(path / "deferred.vert") &&
                 fs::is_regular_file(path / "deferred.frag");

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

  // gbuffer passes
  std::shared_ptr<GbufferPassParser> firstGbufferPass;
  for (int i = 0; i < numGbufferPasses; ++i) {
    std::string suffix = i == 0 ? "" : std::to_string(i);
    auto gbufferPass = std::make_shared<GbufferPassParser>();
    if (i == 0) {
      firstGbufferPass = gbufferPass;
    }

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
    if (!fs::is_regular_file(gsFile)) {
      gsFile = "";
    }
    futures.push_back(pointPass->loadGLSLFilesAsync(vsFile, fsFile, gsFile));
  }

  std::shared_ptr<LinePassParser> linePass{};
  // line pass
  {
    std::string vsFile = (path / "line.vert").string();
    std::string fsFile = (path / "line.frag").string();
    if (fs::is_regular_file(vsFile) && fs::is_regular_file(fsFile)) {
      linePass = std::make_shared<LinePassParser>();
      linePass->setName("Line");
      linePass->setLineWidth(2.f);
      mNonShadowPasses.push_back(linePass);
      mLinePasses.push_back(linePass);
      linePass->setIndex(mNonShadowPasses.size() - 1);
      futures.push_back(linePass->loadGLSLFilesAsync(vsFile, fsFile));
    }
  }

  {
    std::string vsFile = (path / "ao.vert").string();
    std::string fsFile = (path / "ao.frag").string();
    if (fs::is_regular_file(vsFile) && fs::is_regular_file(fsFile)) {
      auto aoPass = std::make_shared<DeferredPassParser>();
      aoPass->setName("AO");
      mNonShadowPasses.push_back(aoPass);
      aoPass->setIndex(mNonShadowPasses.size() - 1);
      futures.push_back(aoPass->loadGLSLFilesAsync(vsFile, fsFile));
    }
  }

  // load deferred pass
  std::string vsFile = (path / "deferred.vert").string();
  std::string fsFile = (path / "deferred.frag").string();
  if (fs::is_regular_file(vsFile) && fs::is_regular_file(fsFile)) {
    auto deferredPass = std::make_shared<DeferredPassParser>();
    deferredPass->setName("Deferred");
    mNonShadowPasses.push_back(deferredPass);
    deferredPass->setIndex(mNonShadowPasses.size() - 1);
    futures.push_back(deferredPass->loadGLSLFilesAsync(vsFile, fsFile));
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
  mTextureOperationTable = generateTextureOperationTable();
}

std::shared_ptr<ShaderConfig> ShaderPack::generateShaderInputLayouts() const {
  DescriptorSetDescription cameraSetDesc;
  DescriptorSetDescription objectSetDesc;
  DescriptorSetDescription sceneSetDesc;
  DescriptorSetDescription lightSetDesc;

  auto layouts = std::make_shared<ShaderConfig>();

  ASSERT(mGbufferPasses.size() > 0,
         "There must be at least 1 gbuffer pass in a shader pack.");
  layouts->vertexLayout = mGbufferPasses.at(0)->getVertexInputLayout();

  if (mLinePasses.size()) {
    layouts->primitiveVertexLayout = mLinePasses.at(0)->getVertexInputLayout();
  }

  if (mPointPasses.size()) {
    if (layouts->primitiveVertexLayout) {
      ASSERT(*layouts->primitiveVertexLayout ==
                 *mPointPasses.at(0)->getVertexInputLayout(),
             "All primitive passes must share the same vertex layout");
    } else {
      layouts->primitiveVertexLayout =
          mPointPasses.at(0)->getVertexInputLayout();
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

  layouts->cameraBufferLayout =
      cameraSetDesc.buffers.at(cameraSetDesc.bindings.at(0).arrayIndex);
  layouts->objectBufferLayout =
      objectSetDesc.buffers.at(objectSetDesc.bindings.at(0).arrayIndex);
  layouts->sceneBufferLayout =
      sceneSetDesc.buffers.at(sceneSetDesc.bindings.at(0).arrayIndex);

  if (mShadowPass) {
    layouts->lightBufferLayout =
        lightSetDesc.buffers.at(lightSetDesc.bindings.at(0).arrayIndex);
    for (auto &binding : sceneSetDesc.bindings) {
      if (binding.second.name == "ShadowBuffer") {
        layouts->shadowBufferLayout =
            sceneSetDesc.buffers.at(binding.second.arrayIndex);
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

std::unordered_map<std::string, std::vector<ShaderPack::RenderTargetOperation>>
ShaderPack::generateTextureOperationTable() const {
  std::unordered_map<std::string,
                     std::vector<ShaderPack::RenderTargetOperation>>
      operationTable;

  // TODO texture format
  std::set<std::string> textureNames;
  std::vector<std::string> textureNamesOrdered;

  for (auto pass : mNonShadowPasses) {
    auto depthName = pass->getDepthRenderTargetName();
    if (depthName.has_value()) {
      if (!textureNames.contains(depthName.value())) {
        textureNames.insert(depthName.value());
        textureNamesOrdered.push_back(depthName.value());
      }
    }
    for (auto &elem : pass->getTextureOutputLayout()->elements) {
      std::string texName = getOutTextureName(elem.second.name);
      if (texName.ends_with("Depth")) {
        throw std::runtime_error(
            "Naming a texture \"*Depth\" is not allowed in a shader pack");
      }
      if (!textureNames.contains(texName)) {
        textureNames.insert(texName);
        textureNamesOrdered.push_back(texName);
      }
    }
  }

  // init op table
  auto passes = mNonShadowPasses;
  for (auto name : textureNames) {
    operationTable[name] = std::vector<ShaderPack::RenderTargetOperation>(
        passes.size(), ShaderPack::RenderTargetOperation::eNoOp);
  }

  for (uint32_t passIdx = 0; passIdx < passes.size(); ++passIdx) {
    auto pass = passes[passIdx];
    for (auto name : pass->getInputTextureNames()) {
      if (operationTable.find(name) != operationTable.end()) {
        operationTable.at(name)[passIdx] =
            ShaderPack::RenderTargetOperation::eRead;
      }
    }
    for (auto &name : pass->getColorRenderTargetNames()) {
      operationTable.at(name)[passIdx] =
          ShaderPack::RenderTargetOperation::eColorWrite;
    }
    auto name = pass->getDepthRenderTargetName();
    if (name.has_value()) {
      operationTable.at(name.value())[passIdx] =
          ShaderPack::RenderTargetOperation::eDepthWrite;
    }
  }

  std::stringstream ss;
  ss << "Operation Table" << std::endl;
  ss << std::setw(20) << ""
     << " ";
  for (auto pass : passes) {
    ss << std::setw(15) << pass->getName() << " ";
  }
  ss << std::endl;

  for (auto &tex : textureNamesOrdered) {
    auto ops = operationTable.at(tex);
    ss << std::setw(20) << tex << " ";
    for (auto &op : ops) {
      if (op == ShaderPack::RenderTargetOperation::eNoOp) {
        ss << std::setw(15) << " "
           << " ";
      } else if (op == ShaderPack::RenderTargetOperation::eRead) {
        ss << std::setw(15) << "R"
           << " ";
      } else if (op == ShaderPack::RenderTargetOperation::eColorWrite) {
        ss << std::setw(15) << "W"
           << " ";
      } else if (op == ShaderPack::RenderTargetOperation::eDepthWrite) {
        ss << std::setw(15) << "D"
           << " ";
      }
    }
    ss << std::endl;
  }
  // for (auto &[tex, ops] : operationTable) {
  // }
  logger::info(ss.str());

  return operationTable;
}

} // namespace shader
} // namespace svulkan2
