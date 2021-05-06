#include <string>
#include <vector>

namespace svulkan2 {
std::vector<uint8_t> loadImage(std::string const &filename, int &width,
                               int &height);

std::vector<uint8_t> loadImageFromMemory(unsigned char *buffer, int len,
                                         int &width, int &height);
} // namespace svulkan2
