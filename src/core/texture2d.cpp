#include "svulkan2/core/texture2d.h"

namespace svulkan2 {
namespace core {
Texture2D::Texture2D(std::shared_ptr<Image> image, vk::ImageView imageView,
                     vk::Sampler sampler)
    : mImage(image), mImageView(imageView), mSampler(sampler) {}

} // namespace core
} // namespace svulkan2
