#pragma once
#include "svulkan2/common/glm.h"
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(DisplayText) {
  UI_ATTRIBUTE(DisplayText, std::string, Text);

  void build() override;
};

UI_CLASS(InputText) {
  UI_ATTRIBUTE(InputText, std::string, Label);
  UI_ATTRIBUTE(InputText, bool, ReadOnly);
  UI_ATTRIBUTE(InputText, std::function<void(std::shared_ptr<InputText>)>,
               Callback);

protected:
  std::vector<char> mBuffer{std::vector<char>(100, 0)};

public:
  inline std::shared_ptr<InputText> Size(uint32_t size) {
    mBuffer.resize(size);
    return std::static_pointer_cast<InputText>(shared_from_this());
  };

  inline std::shared_ptr<InputText> Value(std::string const &value) {
    if (mBuffer.size() <= value.length()) {
      mBuffer.resize(value.size() + 1);
    }
    std::copy(value.begin(), value.end(), mBuffer.data());
    return std::static_pointer_cast<InputText>(shared_from_this());
  };

  inline std::string get() const {
    return std::string(mBuffer.begin(), mBuffer.end());
  }
  void build() override;
};

UI_CLASS(InputFloat) {
  UI_ATTRIBUTE(InputFloat, std::string, Label);
  UI_ATTRIBUTE(InputFloat, float, Value);
  UI_ATTRIBUTE(InputFloat, bool, ReadOnly);
  UI_ATTRIBUTE(InputFloat, std::function<void(std::shared_ptr<InputFloat>)>,
               Callback);

public:
  inline float get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputFloat2) {
  UI_ATTRIBUTE(InputFloat2, std::string, Label);
  UI_ATTRIBUTE(InputFloat2, glm::vec2, Value);
  UI_ATTRIBUTE(InputFloat2, bool, ReadOnly);
  UI_ATTRIBUTE(InputFloat2, std::function<void(std::shared_ptr<InputFloat2>)>,
               Callback);

public:
  inline glm::vec2 get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputFloat3) {
  UI_ATTRIBUTE(InputFloat3, std::string, Label);
  UI_ATTRIBUTE(InputFloat3, glm::vec3, Value);
  UI_ATTRIBUTE(InputFloat3, bool, ReadOnly);
  UI_ATTRIBUTE(InputFloat3, std::function<void(std::shared_ptr<InputFloat3>)>,
               Callback);

public:
  inline glm::vec3 get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputFloat4) {
  UI_ATTRIBUTE(InputFloat4, std::string, Label);
  UI_ATTRIBUTE(InputFloat4, glm::vec4, Value);
  UI_ATTRIBUTE(InputFloat4, bool, ReadOnly);
  UI_ATTRIBUTE(InputFloat4, std::function<void(std::shared_ptr<InputFloat4>)>,
               Callback);

public:
  inline glm::vec4 get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputInt) {
  UI_ATTRIBUTE(InputInt, std::string, Label);
  UI_ATTRIBUTE(InputInt, int, Value);
  UI_ATTRIBUTE(InputInt, bool, ReadOnly);
  UI_ATTRIBUTE(InputInt, std::function<void(std::shared_ptr<InputInt>)>,
               Callback);

public:
  inline int get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputInt2) {
  UI_ATTRIBUTE(InputInt2, std::string, Label);
  UI_ATTRIBUTE(InputInt2, glm::ivec2, Value);
  UI_ATTRIBUTE(InputInt2, bool, ReadOnly);
  UI_ATTRIBUTE(InputInt2, std::function<void(std::shared_ptr<InputInt2>)>,
               Callback);

public:
  inline glm::ivec2 get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputInt3) {
  UI_ATTRIBUTE(InputInt3, std::string, Label);
  UI_ATTRIBUTE(InputInt3, glm::ivec3, Value);
  UI_ATTRIBUTE(InputInt3, bool, ReadOnly);
  UI_ATTRIBUTE(InputInt3, std::function<void(std::shared_ptr<InputInt3>)>,
               Callback);

public:
  inline glm::ivec3 get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputInt4) {
  UI_ATTRIBUTE(InputInt4, std::string, Label);
  UI_ATTRIBUTE(InputInt4, glm::ivec4, Value);
  UI_ATTRIBUTE(InputInt4, bool, ReadOnly);
  UI_ATTRIBUTE(InputInt4, std::function<void(std::shared_ptr<InputInt4>)>,
               Callback);

public:
  inline glm::ivec4 get() const { return mValue; }
  void build() override;
};

} // namespace ui
} // namespace svulkan2
