#pragma once

#include <algorithm>
#include <unordered_map>
#include <string>
#include <vector>

namespace svulkan2 {
struct DataLayoutElement {
  std::string name;
  uint32_t size;
  uint32_t offset;
};

struct DataLayout {
  std::unordered_map<std::string, DataLayoutElement> elements;
  uint32_t size;

  inline std::vector<DataLayoutElement> getElementsSorted() {
    std::vector<DataLayoutElement> result;
    std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                   [](auto &kv) { return kv.second; });
    std::sort(result.begin(), result.end(),
              [](auto const &elem1, auto const &elem2) {
                return elem1.offset < elem2.offset;
              });
    return result;
  }
};

}; // namespace svulkan2
