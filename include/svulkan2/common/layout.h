#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace svulkan2 {

enum DataType {
  eUNKNOWN,
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
};

inline std::string DataTypeToString(DataType type) {
  switch (type) {
  case eUNKNOWN:
    return "unknown";
  case eINT:
    return "int";
  case eINT2:
    return "int2";
  case eINT3:
    return "int3";
  case eINT4:
    return "int4";
  case eINT44:
    return "int44";
  case eUINT:
    return "uint";
  case eUINT2:
    return "uint2";
  case eUINT3:
    return "uint3";
  case eUINT4:
    return "uint4";
  case eUINT44:
    return "uint44";
  case eFLOAT:
    return "float";
  case eFLOAT2:
    return "float2";
  case eFLOAT3:
    return "float3";
  case eFLOAT4:
    return "float4";
  case eFLOAT44:
    return "float44";
  }
  throw std::runtime_error("invalid data type");
}

inline uint32_t GetDataTypeSize(DataType type) {
  switch (type) {
  case eUNKNOWN:
    return 0;
  case eINT:
    return 4;
  case eINT2:
    return 8;
  case eINT3:
    return 12;
  case eINT4:
    return 16;
  case eINT44:
    return 64;
  case eUINT:
    return 4;
  case eUINT2:
    return 8;
  case eUINT3:
    return 12;
  case eUINT4:
    return 16;
  case eUINT44:
    return 64;
  case eFLOAT:
    return 4;
  case eFLOAT2:
    return 8;
  case eFLOAT3:
    return 12;
  case eFLOAT4:
    return 16;
  case eFLOAT44:
    return 64;
  }
  throw std::runtime_error("invalid data type");
}

struct StructDataLayoutElement {
  std::string name{};
  DataType type{};
  uint32_t size{0};
  uint32_t offset{0};
};

struct StructDataLayout {
  std::unordered_map<std::string, StructDataLayoutElement> elements;
  uint32_t size;

  inline std::vector<StructDataLayoutElement> getElementsSorted() const {
    std::vector<StructDataLayoutElement> result;
    std::transform(elements.begin(), elements.end(), std::back_inserter(result),
                   [](auto const &kv) { return kv.second; });
    std::sort(result.begin(), result.end(),
              [](auto const &elem1, auto const &elem2) {
                return elem1.offset < elem2.offset;
              });
    return result;
  }

  inline std::string summarize() const {
    auto elements = getElementsSorted();
    std::stringstream ss;
    ss << "total size: " << size << "; ";
    for (auto &e : elements) {
      ss << e.name << "[" << DataTypeToString(e.type) << "]"
         << "at: " << e.offset << ", size: " << e.size << "; ";
    }
    return ss.str();
  }
};

struct InOutDataLayoutElement {
  std::string name{};
  DataType type{};
  uint32_t size{0};
  uint32_t location{0};
};

struct InOutDataLayout {
  std::unordered_map<std::string, InOutDataLayoutElement> elements;

  inline std::vector<InOutDataLayoutElement> getElementsSorted() const {
    std::vector<InOutDataLayoutElement> result;
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
      ss << e.name << "[" << DataTypeToString(e.type) << "]"
         << "at: " << e.location << ", size: " << e.size << "; ";
    }
    return ss.str();
  }
};

}; // namespace svulkan2
