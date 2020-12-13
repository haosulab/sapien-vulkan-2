#pragma once

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace svulkan2 {

enum DataType {
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
  case eSTRUCT:
    return "struct";
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
  case eUNKNOWN:
    return "unknown";
  }
  throw std::runtime_error("invalid data type");
}

inline uint32_t GetDataTypeSize(DataType type) {
  switch (type) {
  case eSTRUCT:
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
  case eUNKNOWN:
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
    DataType dtype{eFLOAT};
    inline uint32_t getSize() const { return GetDataTypeSize(dtype); }
  };

  std::unordered_map<std::string, InputDataLayout::Element> elements;

  std::vector<InputDataLayout::Element> getElementsSorted() const;
  uint32_t getSize() const;
};

struct OutputDataLayout {
  struct Element {
    std::string name{};
    uint32_t location{0};
    DataType dtype{eFLOAT};
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
    DataType dtype{eFLOAT};
    std::unique_ptr<StructDataLayout> member;
  };

  uint32_t size;
  std::unordered_map<std::string, StructDataLayout::Element> elements;

  std::vector<StructDataLayout::Element const *> getElementsSorted() const;
};

// struct StructDataLayoutElement {
//   std::string name{};
//   DataType type{};
//   uint32_t size{0};
//   uint32_t offset{0};
// };

// struct StructDataLayout {
//   std::unordered_map<std::string, StructDataLayoutElement> elements;
//   uint32_t size;

//   inline std::vector<StructDataLayoutElement> getElementsSorted() const {
//     std::vector<StructDataLayoutElement> result;
//     std::transform(elements.begin(), elements.end(),
//     std::back_inserter(result),
//                    [](auto const &kv) { return kv.second; });
//     std::sort(result.begin(), result.end(),
//               [](auto const &elem1, auto const &elem2) {
//                 return elem1.offset < elem2.offset;
//               });
//     return result;
//   }

//   inline std::string summarize() const {
//     auto elements = getElementsSorted();
//     std::stringstream ss;
//     ss << "total size: " << size << "; ";
//     for (auto &e : elements) {
//       ss << e.name << "[" << DataTypeToString(e.type) << "]"
//          << "at: " << e.offset << ", size: " << e.size << "; ";
//     }
//     return ss.str();
//   }
// };

// struct InOutDataLayoutElement {
//   std::string name{};
//   DataType type{};
//   uint32_t size{0};
//   uint32_t location{0};
// };

// struct InOutDataLayout {
//   std::unordered_map<std::string, InOutDataLayoutElement> elements;

//   inline std::vector<InOutDataLayoutElement> getElementsSorted() const {
//     std::vector<InOutDataLayoutElement> result;
//     std::transform(elements.begin(), elements.end(),
//     std::back_inserter(result),
//                    [](auto const &kv) { return kv.second; });
//     std::sort(result.begin(), result.end(),
//               [](auto const &elem1, auto const &elem2) {
//                 return elem1.location < elem2.location;
//               });
//     return result;
//   }

//   inline std::string summarize() const {
//     auto elements = getElementsSorted();
//     std::stringstream ss;
//     for (auto &e : elements) {
//       ss << e.name << "[" << DataTypeToString(e.type) << "]"
//          << "at: " << e.location << ", size: " << e.size << "; ";
//     }
//     return ss.str();
//   }
// };

}; // namespace svulkan2
