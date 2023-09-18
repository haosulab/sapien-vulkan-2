#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

namespace svulkan2 {

enum class ComponentFormat { eSnorm, eUnorm, eSint, eUint, eSfloat, eOther };

struct VulkanFormatInfo {
  uint32_t size;
  uint32_t channels;
  uint32_t elemSize;
  ComponentFormat format;
  bool srgb;
  vk::ImageAspectFlags aspect;
};

uint32_t getFormatSize(vk::Format format);
uint32_t getFormatChannels(vk::Format format);
uint32_t getFormatElementSize(vk::Format format);
ComponentFormat getFormatComponentFormat(vk::Format format);
bool getFormatSupportSrgb(vk::Format format);
vk::ImageAspectFlags getFormatAspectFlags(vk::Format format);
template <typename T> bool isFormatCompatible(vk::Format format);

struct TypeKind {
  static constexpr char eInt = 'i';
  static constexpr char eUint = 'u';
  static constexpr char eFloat = 'f';
  static constexpr char eRaw = 'V';
  static constexpr char eStruct = 'O';
};

struct DataType {
  uint32_t shape;
  char kind;
  uint32_t bytes;

  bool operator==(DataType const &other) const = default;

  static const DataType eUINT;
  static const DataType eUINT2;
  static const DataType eUINT3;
  static const DataType eUINT4;

  static const DataType eINT;
  static const DataType eINT2;
  static const DataType eINT3;
  static const DataType eINT4;

  static const DataType eFLOAT;
  static const DataType eFLOAT2;
  static const DataType eFLOAT3;
  static const DataType eFLOAT4;

  static const DataType eINT44;
  static const DataType eFLOAT44;

  static const DataType eSTRUCT;

  uint32_t size() const { return shape * bytes; }
  std::string typestr() const {
    return std::to_string(shape) + std::to_string(kind) + std::to_string(bytes);
  }
};

constexpr const DataType DataType::eUINT = {1, TypeKind::eUint, 4};
constexpr const DataType DataType::eUINT2 = {2, TypeKind::eUint, 4};
constexpr const DataType DataType::eUINT3 = {3, TypeKind::eUint, 4};
constexpr const DataType DataType::eUINT4 = {4, TypeKind::eUint, 4};

constexpr const DataType DataType::eINT = {1, TypeKind::eInt, 4};
constexpr const DataType DataType::eINT2 = {2, TypeKind::eInt, 4};
constexpr const DataType DataType::eINT3 = {3, TypeKind::eInt, 4};
constexpr const DataType DataType::eINT4 = {4, TypeKind::eInt, 4};

constexpr const DataType DataType::eFLOAT = {1, TypeKind::eFloat, 4};
constexpr const DataType DataType::eFLOAT2 = {2, TypeKind::eFloat, 4};
constexpr const DataType DataType::eFLOAT3 = {3, TypeKind::eFloat, 4};
constexpr const DataType DataType::eFLOAT4 = {4, TypeKind::eFloat, 4};

constexpr const DataType DataType::eINT44 = {16, TypeKind::eInt, 4};
constexpr const DataType DataType::eFLOAT44 = {16, TypeKind::eFloat, 4};

constexpr const DataType DataType::eSTRUCT = {0, TypeKind::eStruct, 0};

template <typename T> struct DataTypeFor {};

template <> struct DataTypeFor<int> {
  static constexpr DataType value = DataType::eINT;
};
template <> struct DataTypeFor<glm::ivec4> {
  static constexpr DataType value = DataType::eINT4;
};

template <> struct DataTypeFor<uint32_t> {
  static constexpr DataType value = DataType::eUINT;
};

template <> struct DataTypeFor<glm::uvec4> {
  static constexpr DataType value = DataType::eUINT4;
};

template <> struct DataTypeFor<float> {
  static constexpr DataType value = DataType::eFLOAT;
};

template <> struct DataTypeFor<glm::vec4> {
  static constexpr DataType value = DataType::eFLOAT4;
};

template <> struct DataTypeFor<glm::vec3> {
  static constexpr DataType value = DataType::eFLOAT3;
};

template <> struct DataTypeFor<glm::mat4> {
  static constexpr DataType value = DataType::eFLOAT44;
};

} // namespace svulkan2
