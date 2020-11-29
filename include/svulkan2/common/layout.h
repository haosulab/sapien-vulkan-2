#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace svulkan2 {
struct DataLayoutElement {
  std::string name;
  std::string typeName;
  uint32_t size;
  uint32_t location;
};

struct DataLayout {
  std::unordered_map<std::string, DataLayoutElement> elements;
  uint32_t size;

  inline std::vector<DataLayoutElement> getElementsSorted() const {
    std::vector<DataLayoutElement> result;
    std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                   [](auto const &kv) { return kv.second; });
    std::sort(result.begin(), result.end(),
              [](auto const &elem1, auto const &elem2) {
                return elem1.location < elem2.location;
              });
    return result;
  }

  inline std::string summarize() const {
    auto elements = getElementsSorted();
    std::stringstream ss;
    for (auto &e : elements) {
      ss << e.name << "[" << e.typeName << "]"
         << "at: " << e.location << ", size: " << e.size << "\n";
    }
    return ss.str();
  }
};

}; // namespace svulkan2
