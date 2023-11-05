#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace svulkan2 {

/**
 * load image from file.
 * valid values for desiredChannels are {0, 1, 4}
 * When desiredChannels == 0, channels is detected from the file
 * */
std::vector<uint8_t> loadImage(std::string const &filename, int &width, int &height, int &channels,
                               int desiredChannels);

std::vector<uint8_t> loadImageFromMemory(unsigned char *buffer, int len, int &width, int &height,
                                         int &channels, int desiredChannels);

std::vector<uint8_t> loadKTXImage(std::string const &filename, int &width, int &height,
                                  int &levels, int &faces, int &layers, vk::Format &format);
std::vector<uint8_t> loadKTXImageFromMemory(unsigned char *buffer, size_t size, int &width,
                                            int &height, int &levels, int &faces, int &layers,
                                            vk::Format &format);

// the loaded image is always rgba float16
std::vector<std::byte> loadExrImage(std::string const &filename, int &width, int &height);

template <typename T> std::vector<char> toRawBytes(std::vector<T> const &data) {
  std::vector<char> raw(data.size() * sizeof(T));
  std::memcpy(raw.data(), data.data(), raw.size());
  return raw;
}

uint32_t computeMipLevelSize(vk::Extent3D extent, uint32_t level);
vk::Extent3D computeMipLevelExtent(vk::Extent3D extent, uint32_t level);

} // namespace svulkan2
