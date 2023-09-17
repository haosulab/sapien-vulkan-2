#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace svulkan2 {
std::vector<uint8_t> loadImage(std::string const &filename, int &width, int &height,
                               int &channels);

std::vector<uint8_t> loadImageFromMemory(unsigned char *buffer, int len, int &width, int &height,
                                         int &channels);

std::vector<uint8_t> loadKTXImage(std::string const &filename, int &width, int &height,
                                  int &levels, int &faces, int &layers, vk::Format &format);

template <typename T> std::vector<char> toRawBytes(std::vector<T> const &data) {
  std::vector<char> raw(data.size() * sizeof(T));
  std::memcpy(raw.data(), data.data(), raw.size());
  return raw;
}

uint32_t computeMipLevelSize(vk::Extent3D extent, uint32_t level);
vk::Extent3D computeMipLevelExtent(vk::Extent3D extent, uint32_t level);

} // namespace svulkan2
