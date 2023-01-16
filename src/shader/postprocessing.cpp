#include "svulkan2/shader/postprocessing.h"
#include "svulkan2/common/launch_policy.h"
#include "svulkan2/common/log.h"

namespace svulkan2 {
namespace shader {

std::future<void>
PostprocessingShaderParser::loadFileAsync(std::string const &filepath) {
  return std::async(LAUNCH_ASYNC, [=, this]() {
    log::info("Compiling: " + filepath);
    mSPVCode = GLSLCompiler::compileGlslFileCached(
        vk::ShaderStageFlagBits::eCompute, filepath);
    log::info("Compiled: " + filepath);
    reflectSPV();
  });
}

void PostprocessingShaderParser::reflectSPV() {
  spirv_cross::Compiler compiler(mSPVCode);
  auto resources = compiler.get_shader_resources();
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

  // TODO: validate only 1 set is used and everything is storage image
}

} // namespace shader
} // namespace svulkan2
