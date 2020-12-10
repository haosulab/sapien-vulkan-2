#include "svulkan2/common/layout.h"

namespace svulkan2 {

std::vector<CombinedSamplerLayout::Element>
CombinedSamplerLayout::getElementsSorted() const {
  std::vector<CombinedSamplerLayout::Element> result;
  std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                 [](auto const &kv) { return kv.second; });
  std::sort(result.begin(), result.end(),
            [](auto const &elem1, auto const &elem2) {
              return elem1.location < elem2.location;
            });
  return result;
}

std::vector<InputDataLayout::Element>
InputDataLayout::getElementsSorted() const {
  std::vector<InputDataLayout::Element> result;
  std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                 [](auto const &kv) { return kv.second; });
  std::sort(result.begin(), result.end(),
            [](auto const &elem1, auto const &elem2) {
              return elem1.location < elem2.location;
            });
  return result;
}

uint32_t InputDataLayout::getSize() const {
  uint32_t sum = 0;
  for (auto &e : elements) {
    sum += e.second.getSize();
  }
  return sum;
}

std::vector<OutputDataLayout::Element>
OutputDataLayout::getElementsSorted() const {
  std::vector<OutputDataLayout::Element> result;
  std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                 [](auto const &kv) { return kv.second; });
  std::sort(result.begin(), result.end(),
            [](auto const &elem1, auto const &elem2) {
              return elem1.location < elem2.location;
            });
  return result;
}

inline std::vector<StructDataLayout::Element>
StructDataLayout::getElementsSorted() const {
  std::vector<StructDataLayout::Element> result;
  std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                 [](auto const &kv) { return kv.second; });
  std::sort(result.begin(), result.end(),
            [](auto const &elem1, auto const &elem2) {
              return elem1.offset < elem2.offset;
            });
  return result;
}

}; // namespace svulkan2
