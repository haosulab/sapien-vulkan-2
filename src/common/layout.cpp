#include "svulkan2/common/layout.h"

namespace svulkan2 {

std::vector<SpecializationConstantLayout::Element>
SpecializationConstantLayout::getElementsSorted() const {
  std::vector<SpecializationConstantLayout::Element> result;
  std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                 [](auto const &kv) { return kv.second; });
  std::sort(
      result.begin(), result.end(),
      [](auto const &elem1, auto const &elem2) { return elem1.id < elem2.id; });
  return result;
}

std::vector<CombinedSamplerLayout::Element>
CombinedSamplerLayout::getElementsSorted() const {
  std::vector<CombinedSamplerLayout::Element> result;
  std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                 [](auto const &kv) { return kv.second; });
  std::sort(result.begin(), result.end(),
            [](auto const &elem1, auto const &elem2) {
              return elem1.binding < elem2.binding;
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

std::vector<vk::VertexInputBindingDescription>
InputDataLayout::computeVertexInputBindingDescriptions() const {
  return {vk::VertexInputBindingDescription(0, getSize())};
}

std::vector<vk::VertexInputAttributeDescription>
InputDataLayout::computeVertexInputAttributesDescriptions() const {
  std::vector<vk::VertexInputAttributeDescription>
      vertexInputAttributeDescriptions;
  uint32_t offset = 0;
  uint32_t count = 0;
  auto elements = getElementsSorted();
  for (auto &elem : elements) {
    if (elem.location != count) {
      throw std::runtime_error("vertex layout locations must be "
                               "consecutive integers starting from 0");
    }
    if (elem.dtype == eFLOAT) {
      vertexInputAttributeDescriptions.push_back(
          {elem.location, 0, vk::Format::eR32Sfloat, offset});
    } else if (elem.dtype == eFLOAT2) {
      vertexInputAttributeDescriptions.push_back(
          {elem.location, 0, vk::Format::eR32G32Sfloat, offset});
    } else if (elem.dtype == eFLOAT3) {
      vertexInputAttributeDescriptions.push_back(
          {elem.location, 0, vk::Format::eR32G32B32Sfloat, offset});
    } else if (elem.dtype == eFLOAT4) {
      vertexInputAttributeDescriptions.push_back(
          {elem.location, 0, vk::Format::eR32G32B32A32Sfloat, offset});
    } else {
      throw std::runtime_error("vertex attributes only supports float, "
                               "float2, float3, float4 formats");
    }

    offset += elem.getSize();
    count += 1;
  }
  return vertexInputAttributeDescriptions;
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

std::vector<StructDataLayout::Element const *>
StructDataLayout::getElementsSorted() const {
  std::vector<StructDataLayout::Element const *> result;
  std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                 [](auto const &kv) { return &kv.second; });
  std::sort(result.begin(), result.end(),
            [](auto const &elem1, auto const &elem2) {
              return elem1->offset < elem2->offset;
            });
  return result;
}
}; // namespace svulkan2
