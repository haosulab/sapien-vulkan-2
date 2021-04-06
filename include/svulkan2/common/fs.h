#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace svulkan2 {

std::vector<char> readFile(std::filesystem::path const &filename);

} // namespace svulkan2
