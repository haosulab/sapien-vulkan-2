#pragma once
#include "svulkan2/common/vk.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace svulkan2 {

enum class DataType {
  eSTRUCT,
  eINT,
  eINT2,
  eINT3,
  eINT4,
  eINT44,
  eUINT,
  eUINT2,
  eUINT3,
  eUINT4,
  eUINT44,
  eFLOAT,
  eFLOAT2,
  eFLOAT3,
  eFLOAT4,
  eFLOAT44,
  eUNKNOWN
};

inline std::string DataTypeToString(DataType type) {
  switch (type) {
  case DataType::eSTRUCT:
    return "struct";
  case DataType::eINT:
    return "int";
  case DataType::eINT2:
    return "int2";
  case DataType::eINT3:
    return "int3";
  case DataType::eINT4:
    return "int4";
  case DataType::eINT44:
    return "int44";
  case DataType::eUINT:
    return "uint";
  case DataType::eUINT2:
    return "uint2";
  case DataType::eUINT3:
    return "uint3";
  case DataType::eUINT4:
    return "uint4";
  case DataType::eUINT44:
    return "uint44";
  case DataType::eFLOAT:
    return "float";
  case DataType::eFLOAT2:
    return "float2";
  case DataType::eFLOAT3:
    return "float3";
  case DataType::eFLOAT4:
    return "float4";
  case DataType::eFLOAT44:
    return "float44";
  case DataType::eUNKNOWN:
    return "unknown";
  }
  throw std::runtime_error("invalid data type");
}

inline uint32_t GetDataTypeSize(DataType type) {
  switch (type) {
  case DataType::eSTRUCT:
    return 0;
  case DataType::eINT:
    return 4;
  case DataType::eINT2:
    return 8;
  case DataType::eINT3:
    return 12;
  case DataType::eINT4:
    return 16;
  case DataType::eINT44:
    return 64;
  case DataType::eUINT:
    return 4;
  case DataType::eUINT2:
    return 8;
  case DataType::eUINT3:
    return 12;
  case DataType::eUINT4:
    return 16;
  case DataType::eUINT44:
    return 64;
  case DataType::eFLOAT:
    return 4;
  case DataType::eFLOAT2:
    return 8;
  case DataType::eFLOAT3:
    return 12;
  case DataType::eFLOAT4:
    return 16;
  case DataType::eFLOAT44:
    return 64;
  case DataType::eUNKNOWN:
    return 0;
  }
  throw std::runtime_error("invalid data type");
}

struct SpecializationConstantLayout {
  struct Element {
    std::string name{};
    uint32_t id{0};
    DataType dtype{};
    union {
      int intValue;
      float floatValue;
    };
  };
  std::unordered_map<std::string, SpecializationConstantLayout::Element>
      elements;
  std::vector<SpecializationConstantLayout::Element> getElementsSorted() const;
};

struct CombinedSamplerLayout {
  struct Element {
    std::string name{};
    uint32_t binding{0};
    uint32_t set{0};
  };
  std::unordered_map<std::string, CombinedSamplerLayout::Element> elements;
  std::vector<CombinedSamplerLayout::Element> getElementsSorted() const;
};

struct InputDataLayout {
  struct Element {
    std::string name{};
    uint32_t location{0};
    DataType dtype{DataType::eFLOAT};
    inline uint32_t getSize() const { return GetDataTypeSize(dtype); }
    inline bool operator==(Element const &other) const {
      return name == other.name && location == other.location &&
             dtype == other.dtype;
    }
  };

  std::unordered_map<std::string, InputDataLayout::Element> elements;

  std::vector<InputDataLayout::Element> getElementsSorted() const;
  uint32_t getSize() const;

  std::vector<vk::VertexInputBindingDescription>
  computeVertexInputBindingDescriptions() const;
  std::vector<vk::VertexInputAttributeDescription>
  computeVertexInputAttributesDescriptions() const;

  bool operator==(InputDataLayout const &other) const;
  bool operator!=(InputDataLayout const &other) const;
};

struct OutputDataLayout {
  struct Element {
    std::string name{};
    uint32_t location{0};
    DataType dtype{DataType::eFLOAT};
  };

  std::unordered_map<std::string, OutputDataLayout::Element> elements;

  std::vector<OutputDataLayout::Element> getElementsSorted() const;
};

struct StructDataLayout {
  struct Element {
    std::string name{};
    uint32_t size{0};
    uint32_t offset{0};
    uint32_t arrayDim{0};
    DataType dtype{DataType::eFLOAT};
    std::shared_ptr<StructDataLayout> member;
    bool operator==(Element const &other) const;
  };

  uint32_t size;
  std::unordered_map<std::string, StructDataLayout::Element> elements;
  std::vector<StructDataLayout::Element const *> getElementsSorted() const;

  bool operator==(StructDataLayout const &other) const;
  bool operator!=(StructDataLayout const &other) const;
  uint32_t getAlignedSize(uint32_t alignment) const;
};

inline void strided_memcpy(void *target, void *source, size_t chunk_size,
                           size_t chunks, size_t stride) {
  char *target_ = reinterpret_cast<char *>(target);
  char *source_ = reinterpret_cast<char *>(source);

  for (size_t i = 0; i < chunks; ++i) {
    std::memcpy(target_, source_, chunk_size);
    target_ += stride;
    source_ += chunk_size;
  }
}

}; // namespace svulkan2
