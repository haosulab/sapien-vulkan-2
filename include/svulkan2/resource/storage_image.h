#pragma once
#include "svulkan2/core/image.h"
#include <memory>
#include <vector>

namespace svulkan2 {
namespace resource {

class SVStorageImage {
  std::shared_ptr<core::Context> mContext;
  std::string mName;

  vk::Format mFormat;
  uint32_t mWidth{};
  uint32_t mHeight{};

  std::shared_ptr<core::Image> mImage{};
  vk::UniqueImageView mImageView;

  bool mOnDevice{};

public:
  SVStorageImage(std::string const &name, uint32_t width, uint32_t height,
                 vk::Format format);

  void createDeviceResources();
  template <typename T> std::vector<T> download() {
    return mImage->download<T>();
  }

  inline uint32_t getWidth() const { return mWidth; }
  inline uint32_t getHeight() const { return mHeight; }

  inline core::Image &getImage() const { return *mImage; }
  inline vk::ImageView getImageView() const { return mImageView.get(); };
  inline vk::Format getFormat() { return mFormat; };
};

} // namespace resource
} // namespace svulkan2
