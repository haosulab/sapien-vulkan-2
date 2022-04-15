#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace svulkan2 {

void exportTriangleMesh(std::string const &filename,
                        std::vector<float> const &vertices,
                        std::vector<uint32_t> const &indices,
                        std::vector<float> const &normals,
                        std::vector<float> const &uvs);
}; // namespace svulkan2
