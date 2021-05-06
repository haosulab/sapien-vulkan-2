#include "svulkan2/common/image.h"
#include <stdexcept>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <stb_image.h>
#pragma GCC diagnostic pop

namespace svulkan2 {
std::vector<uint8_t> loadImage(std::string const &filename, int &width,
                               int &height) {
  int nrChannels;
  unsigned char *data =
      stbi_load(filename.c_str(), &width, &height, &nrChannels, STBI_rgb_alpha);
  if (!data) {
    throw std::runtime_error("failed to load image: " + filename);
  }
  std::vector<uint8_t> dataVector(data, data + width * height * 4);
  stbi_image_free(data);
  return dataVector;
}

std::vector<uint8_t> loadImageFromMemory(unsigned char *buffer, int len,
                                         int &width, int &height) {
  int nrChannels;
  unsigned char *data = stbi_load_from_memory(buffer, len, &width, &height,
                                              &nrChannels, STBI_rgb_alpha);
  if (!data) {
    throw std::runtime_error("failed to load image from memory");
  }
  std::vector<uint8_t> dataVector(data, data + width * height * 4);
  stbi_image_free(data);
  return dataVector;
}

} // namespace svulkan2
