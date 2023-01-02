#include "svulkan2/shader/rt.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

static std::string summarizeResources(
    std::unordered_map<uint32_t, DescriptorSetDescription> const &resources) {
  std::stringstream ss;
  for (auto &[sid, set] : resources) {
    ss << "Set " << std::setw(2) << sid << "\n";
    for (auto &[bid, b] : set.bindings) {
      ss << "  Binding " << std::setw(2) << bid << std::setw(20) << b.name;
      switch (b.type) {
      case vk::DescriptorType::eUniformBuffer:
        ss << " UniformBuffer\n";
        ss << "    Dim  " << b.dim << "\n";
        if (b.dim > 0) {
          ss << "    Size " << b.arraySize << "\n";
        }
        break;
      case vk::DescriptorType::eStorageBuffer:
        ss << " StorageBuffer\n";
        ss << "    Dim  " << b.dim << "\n";
        if (b.dim > 0) {
          ss << "    Size " << b.arraySize << "\n";
        }
        ss << "    " << std::setw(10) << "Field" << std::setw(10) << "offset"
           << std::setw(10) << "size\n";
        for (auto &elem : set.buffers[b.arrayIndex]->getElementsSorted()) {
          ss << "    " << std::setw(10) << elem->name << std::setw(10)
             << elem->offset << std::setw(10) << elem->size << "\n";
        }
        break;
      case vk::DescriptorType::eCombinedImageSampler:
        ss << " CombinedImageSampler\n";
        break;
      case vk::DescriptorType::eStorageImage:
        ss << " StorageImage\n";
        break;
      case vk::DescriptorType::eAccelerationStructureKHR:
        ss << " AccelerationStructure\n";
        break;
      default:
        ss << " Unknown\n";
        break;
      }
    }
  }
  return ss.str();
}

std::future<void>
RayTracingStageParser::loadFileAsync(std::string const &filepath,
                                     vk::ShaderStageFlagBits stage) {
  return std::async(LAUNCH_ASYNC, [=, this]() {
    log::info("Compiling: " + filepath);
    mSPVCode = GLSLCompiler::compileGlslFileCached(stage, filepath);
    log::info("Compiled: " + filepath);
    reflectSPV();
  });
}

void RayTracingStageParser::reflectSPV() {
  mResources.clear();

  spirv_cross::Compiler compiler(mSPVCode);
  auto resources = compiler.get_shader_resources();

  for (auto &r : resources.uniform_buffers) {
    uint32_t setNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet);
    uint32_t bindingNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
    int dim = compiler.get_type(r.type_id).array.size();
    int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;

    if (mResources[setNumber].bindings.contains(bindingNumber)) {
      log::critical("duplicated set {} binding {}", setNumber, bindingNumber);
      throw std::runtime_error("shader compilation failed");
    }

    mResources[setNumber].buffers.push_back(parseBuffer(compiler, r));
    mResources[setNumber].bindings[bindingNumber] = {
        .name = r.name,
        .type = vk::DescriptorType::eUniformBuffer,
        .dim = dim,
        .arraySize = arraySize,
        .arrayIndex =
            static_cast<uint32_t>(mResources[setNumber].buffers.size() - 1)};
  }

  for (auto &r : resources.storage_buffers) {
    uint32_t setNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet);
    uint32_t bindingNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
    int dim = compiler.get_type(r.type_id).array.size();
    int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;

    if (mResources[setNumber].bindings.contains(bindingNumber)) {
      log::critical("duplicated set {} binding {}", setNumber, bindingNumber);
      throw std::runtime_error("shader compilation failed");
    }

    mResources[setNumber].buffers.push_back(parseBuffer(compiler, r));
    mResources[setNumber].bindings[bindingNumber] = {
        .name = r.name,
        .type = vk::DescriptorType::eStorageBuffer,
        .dim = dim,
        .arraySize = arraySize,
        .arrayIndex =
            static_cast<uint32_t>(mResources[setNumber].buffers.size() - 1)};
  }

  for (auto &r : resources.acceleration_structures) {
    uint32_t setNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet);
    uint32_t bindingNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
    int dim = compiler.get_type(r.type_id).array.size();
    int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;

    mResources[setNumber].bindings[bindingNumber] = {
        .name = r.name,
        .type = vk::DescriptorType::eAccelerationStructureKHR,
        .dim = dim,
        .arraySize = arraySize,
        .arrayIndex = 0};
  }

  for (auto &r : resources.storage_images) {
    uint32_t setNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet);
    uint32_t bindingNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
    vk::Format format =
        get_image_format(compiler.get_type(r.type_id).image.format);
    int dim = compiler.get_type(r.type_id).array.size();
    int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;

    mResources[setNumber].images.push_back(r.name);
    mResources[setNumber].bindings[bindingNumber] = {
        .name = r.name,
        .type = vk::DescriptorType::eStorageImage,
        .dim = dim,
        .arraySize = arraySize,
        .arrayIndex =
            static_cast<uint32_t>(mResources[setNumber].images.size() - 1),
        .format = format};
  }

  for (auto &r : resources.sampled_images) {
    uint32_t setNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationDescriptorSet);
    uint32_t bindingNumber =
        compiler.get_decoration(r.id, spv::Decoration::DecorationBinding);
    // log::info("name {} format {}", r.name,
    // compiler.get_type(r.type_id).image.format);
    int dim = compiler.get_type(r.type_id).array.size();
    int arraySize = dim == 1 ? compiler.get_type(r.type_id).array[0] : 0;

    mResources[setNumber].samplers.push_back(r.name);
    mResources[setNumber].bindings[bindingNumber] = {
        .name = r.name,
        .type = vk::DescriptorType::eCombinedImageSampler,
        .dim = dim,
        .arraySize = arraySize,
        .arrayIndex =
            static_cast<uint32_t>(mResources[setNumber].samplers.size() - 1)};
  }

  log::info("\n" + summary());

  // TODO: push constants
  // TODO: analyze rayPayload
}

std::string RayTracingStageParser::summary() const {
  return summarizeResources(mResources);
}

std::future<void> RayTracingParser::loadGLSLFilesAsync(
    const std::string &raygenFile, const std::vector<std::string> &missFiles,
    const std::vector<std::string> &anyhitFiles,
    const std::vector<std::string> &closesthitFiles) {

  return std::async(LAUNCH_ASYNC, [=, this]() {
    std::vector<std::future<void>> futures;
    mRaygenStageParser = std::make_unique<RayTracingStageParser>();
    futures.push_back(mRaygenStageParser->loadFileAsync(
        raygenFile, vk::ShaderStageFlagBits::eRaygenKHR));

    mMissStageParsers.clear();
    for (auto &path : missFiles) {
      mMissStageParsers.push_back(std::make_unique<RayTracingStageParser>());
      futures.push_back(mMissStageParsers.back()->loadFileAsync(
          path, vk::ShaderStageFlagBits::eMissKHR));
    }

    mAnyHitStageParsers.clear();
    for (auto &path : anyhitFiles) {
      mAnyHitStageParsers.push_back(std::make_unique<RayTracingStageParser>());
      futures.push_back(mAnyHitStageParsers.back()->loadFileAsync(
          path, vk::ShaderStageFlagBits::eAnyHitKHR));
    }

    mClosestHitStageParsers.clear();
    for (auto &path : closesthitFiles) {
      mClosestHitStageParsers.push_back(
          std::make_unique<RayTracingStageParser>());
      futures.push_back(mClosestHitStageParsers.back()->loadFileAsync(
          path, vk::ShaderStageFlagBits::eClosestHitKHR));
    }

    for (auto &f : futures) {
      f.get();
    }

    mResources = mRaygenStageParser->getResources();

    // TODO: clean up, hopefully one day view::concat is available
    for (auto &parser : mMissStageParsers) {
      for (auto &[sid, set] : parser->getResources()) {
        if (mResources.contains(sid)) {
          mResources[sid] = mResources.at(sid).merge(set);
        } else {
          mResources[sid] = set;
        }
      }
    }
    for (auto &parser : mAnyHitStageParsers) {
      for (auto &[sid, set] : parser->getResources()) {
        if (mResources.contains(sid)) {
          mResources[sid] = mResources.at(sid).merge(set);
        } else {
          mResources[sid] = set;
        }
      }
    }
    for (auto &parser : mClosestHitStageParsers) {
      for (auto &[sid, set] : parser->getResources()) {
        if (mResources.contains(sid)) {
          mResources[sid] = mResources.at(sid).merge(set);
        } else {
          mResources[sid] = set;
        }
      }
    }
  });
}

std::string RayTracingParser::summary() const {
  return summarizeResources(mResources);
}

} // namespace shader
} // namespace svulkan2
