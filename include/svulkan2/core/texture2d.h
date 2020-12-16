#pragma once
#include "image.h"

namespace svulkan2 {
namespace core {

class Texture2D {
  std::shared_ptr<Image> mImage;
  vk::ImageView mImageView;
  vk::Sampler mSampler;

  Texture2D(std::shared_ptr<Image> image, vk::ImageView imageView,
            vk::Sampler sampler);
};

} // namespace core
} // namespace svulkan2
