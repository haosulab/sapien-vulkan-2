#include "svulkan2/shader/compute_module.h"
#include "../common/logger.h"
#include "reflect.h"
#include "svulkan2/core/command_buffer.h"
#include "svulkan2/core/context.h"
#include "svulkan2/core/device.h"
#include "svulkan2/core/physical_device.h"
#include <stdexcept>

namespace svulkan2 {
namespace shader {

static std::string summarizeResources(std::vector<DescriptorSetDescription> const &resources) {
  std::stringstream ss;
  for (uint32_t sid = 0; sid < resources.size(); ++sid) {
    auto &set = resources[sid];
    ss << "\nSet " << std::setw(2) << sid;
    if (set.type != UniformBindingType::eUnknown) {
      if (set.type == UniformBindingType::eRTCamera) {
        ss << "    Camera";
      } else if (set.type == UniformBindingType::eRTScene) {
        ss << "     Scene";
      } else if (set.type == UniformBindingType::eRTOutput) {
        ss << "    Output";
      }
    }
    ss << "\n";
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
        ss << "    " << std::setw(15) << "Field" << std::setw(15) << "offset" << std::setw(15)
           << "size" << std::setw(15) << "dim" << std::setw(15) << "type" << std::endl;
        for (auto &elem : set.buffers[b.arrayIndex]->getElementsSorted()) {
          ss << "    " << std::setw(15) << elem->name << std::setw(15) << elem->offset
             << std::setw(15) << elem->size << std::setw(15) << elem->arrayDim << std::setw(15)
             << DataTypeToString(elem->dtype) << std::endl;
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

static std::string summarizeConstant(SpecializationConstantLayout const &layout) {
  std::stringstream ss;
  for (auto elem : layout.getElementsSorted()) {
    ss << elem.id;
    ss << std::setw(15) << elem.name << std::setw(10) << DataTypeToString(elem.dtype);
    if (elem.dtype == DataType::eINT || elem.dtype == DataType::eUINT) {
      ss << std::setw(10) << elem.intValue << std::endl;
    } else {
      ss << std::setw(10) << elem.floatValue << std::endl;
    }
  }
  return ss.str();
}

ComputeModule::ComputeModule(std::string const &filename, int blockSizeX, int blockSizeY,
                             int blockSizeZ)
    : mBlockSize({blockSizeX, blockSizeY, blockSizeZ}) {
  mContext = core::Context::Get();
  mCode = GLSLCompiler::compileGlslFileCached(vk::ShaderStageFlagBits::eCompute, filename);
  reflect();
  compile();
}

void ComputeModule::reflect() {
  spirv_cross::Compiler compiler(mCode);
  auto resources = compiler.get_shader_resources();

  auto ids = getDescriptorSetIds(compiler);
  if (ids.size() == 0) {
    throw std::runtime_error("compute shader has no input");
  }

  if (ids.size() != 1) {
    throw std::runtime_error(
        "Only 1 single descriptor set is currently allowed"); // TODO: allow more
  }

  mDescriptorSetDescriptions.clear();
  for (uint32_t i = 0; i < ids.size(); ++i) {
    // ids should be consecutive integers
    if (i != ids.at(i)) {
      throw std::runtime_error("compute shader descriptor sets should be consecutive integers");
    }
    mDescriptorSetDescriptions.push_back(getDescriptorSetDescription(compiler, i));
  }

  mSpecializationConstantLayout = parseSpecializationConstant(compiler);

  mPushConstantLayout = nullptr;
  for (auto &r : resources.push_constant_buffers) {
    auto const &type = compiler.get_type(r.type_id);
    if (!type_is_struct(type)) {
      throw std::runtime_error("push constant buffer must be a struct");
    }
    mPushConstantLayout = parseBuffer(compiler, type);
  }

  // TODO: handle variable work group size
  spirv_cross::SpecializationConstant x, y, z;
  compiler.get_work_group_size_specialization_constants(x, y, z);

  if (x.id) {
    mSpecializationConstantLayout->elements["local_size_x_id"] = {
        .name = "local_size_x_id", .id = x.constant_id, .dtype = DataType::eUINT, .intValue = 0};
  }
  if (y.id) {
    mSpecializationConstantLayout->elements["local_size_y_id"] = {
        .name = "local_size_y_id", .id = y.constant_id, .dtype = DataType::eUINT, .intValue = 0};
  }
  if (z.id) {
    mSpecializationConstantLayout->elements["local_size_z_id"] = {
        .name = "local_size_z_id", .id = z.constant_id, .dtype = DataType::eUINT, .intValue = 0};
  }

  logger::info("{}", summarizeResources(mDescriptorSetDescriptions));
  logger::info("{}", summarizeConstant(*mSpecializationConstantLayout));
}

void ComputeModule::compile() {
  auto device = mContext->getDevice();

  std::vector<vk::DescriptorSetLayout> layouts;

  for (auto &set : mDescriptorSetDescriptions) {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (auto &[bid, binding] : set.bindings) {
      // TODO: handle storage image
      if (binding.type == vk::DescriptorType::eStorageBuffer) {
        // TODO: handle binding.dim != 0
        bindings.push_back(
            {bid, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
      }
    }
    mSetLayouts.push_back(device.createDescriptorSetLayoutUnique({{}, bindings}));
    layouts.push_back(mSetLayouts.back().get());
  }

  std::vector<vk::PushConstantRange> pushConstantRanges;
  if (mPushConstantLayout) {
    pushConstantRanges.push_back(
        vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, mPushConstantLayout->size));
    mConstantBuffer.resize(mPushConstantLayout->size);
  }

  mPipelineLayout = device.createPipelineLayoutUnique(
      vk::PipelineLayoutCreateInfo{{}, layouts, pushConstantRanges});

  auto elems = mSpecializationConstantLayout->getElementsSorted();
  std::vector<vk::SpecializationMapEntry> entries;
  std::vector<int> specializationData;
  if (elems.size()) {
    specializationData.resize(elems.size());
    for (uint32_t i = 0; i < elems.size(); ++i) {
      if (elems[i].name == "subgroupSize") {
        entries.emplace_back(elems[i].id, i * sizeof(int), sizeof(int));
        uint32_t v = core::Context::Get()->getDevice2()->getPhysicalDevice()->getSubgroupSize();
        // TODO: make simpler
        std::memcpy(specializationData.data() + i, &v, sizeof(int));
      } else if (elems[i].name == "local_size_x_id") {
        entries.emplace_back(elems[i].id, i * sizeof(int), sizeof(int));
        std::memcpy(specializationData.data() + i, &mBlockSize[0], sizeof(int));
      } else if (elems[i].name == "local_size_y_id") {
        entries.emplace_back(elems[i].id, i * sizeof(int), sizeof(int));
        std::memcpy(specializationData.data() + i, &mBlockSize[1], sizeof(int));
      } else if (elems[i].name == "local_size_z_id") {
        entries.emplace_back(elems[i].id, i * sizeof(int), sizeof(int));
        std::memcpy(specializationData.data() + i, &mBlockSize[2], sizeof(int));
      }
    }
  }
  auto specializationInfo =
      vk::SpecializationInfo(entries.size(), entries.data(),
                             specializationData.size() * sizeof(int), specializationData.data());

  auto shaderModule = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, mCode));
  auto shaderStageInfo =
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, shaderModule.get(),
                                        "main", elems.size() ? &specializationInfo : nullptr);
  vk::ComputePipelineCreateInfo computePipelineCreateInfo({}, shaderStageInfo,
                                                          mPipelineLayout.get());
  auto pipelinCache = device.createPipelineCacheUnique({});
  mPipeline =
      device.createComputePipelineUnique(pipelinCache.get(), computePipelineCreateInfo).value;

  mSets.clear();
  for (auto layout : layouts) {
    mSets.push_back(core::Context::Get()->getDescriptorPool().allocateSet(layout));
  }
}

void ComputeModule::bindBuffers(std::vector<core::Buffer *> buffers) {
  auto device = core::Context::Get()->getDevice();
  // TODO: make more general
  if (buffers.size() != mDescriptorSetDescriptions.at(0).bindings.size()) {
    throw std::runtime_error("incorrect number of buffers");
  }

  std::vector<vk::WriteDescriptorSet> writes;
  std::vector<vk::DescriptorBufferInfo> bufferInfo;
  for (uint32_t i = 0; i < buffers.size(); ++i) {
    bufferInfo.push_back(
        vk::DescriptorBufferInfo(buffers.at(i)->getVulkanBuffer(), 0, VK_WHOLE_SIZE));
  }
  for (uint32_t i = 0; i < buffers.size(); ++i) {
    vk::WriteDescriptorSet write(mSets.at(0).get(), i, 0, vk::DescriptorType::eStorageBuffer, {},
                                 bufferInfo.at(i));
    writes.push_back(write);
  }
  device.updateDescriptorSets(writes, nullptr);
}

void ComputeModule::bindConstantData(void *data, size_t size) {
  if (mPushConstantLayout->size != size) {
    throw std::runtime_error("push constant size does not match");
  }
  std::memcpy(mConstantBuffer.data(), data, size);
}

void ComputeModule::record(core::CommandBuffer &cb, int x, int y, int z) {
  cb.bindPipeline(vk::PipelineBindPoint::eCompute, mPipeline.get());
  if (mPushConstantLayout) {
    cb.pushConstants(mPipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
                     mConstantBuffer.size(), mConstantBuffer.data());
  }
  cb.bindDescriptorSet(vk::PipelineBindPoint::eCompute, mPipelineLayout.get(), 0,
                       mSets.at(0).get());
  cb.dispatch(x, y, z);
}

ComputeModule::~ComputeModule() {}

} // namespace shader
} // namespace svulkan2
