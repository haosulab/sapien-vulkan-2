#pragma once
#include "svulkan2/core/image.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace core {
class Context;
}

namespace resource {

class SVRenderTarget {
  std::string mName;

  vk::Format mFormat;
  uint32_t mWidth{};
  uint32_t mHeight{};

  std::unique_ptr<core::Image> mImage{};
  vk::UniqueImageView mImageView;
  vk::UniqueSampler mSampler;

  bool mOnDevice{};

public:
  SVRenderTarget(std::string const &name, uint32_t width, uint32_t height,
                 vk::Format format);

  void createDeviceResources(core::Context &context);
  template <typename T> std::vector<T> download();
};

} // namespace resource
} // namespace svulkan2
