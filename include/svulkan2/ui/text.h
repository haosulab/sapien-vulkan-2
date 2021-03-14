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
  inline std::string get() const {
    return std::string(mBuffer.begin(), mBuffer.end());
  }
  void build() override;
};

UI_CLASS(InputFloat) {
  UI_ATTRIBUTE(InputFloat, std::string, Label);
  UI_ATTRIBUTE(InputFloat, float, Value);
  UI_ATTRIBUTE(InputText, bool, ReadOnly);
  UI_ATTRIBUTE(InputText, std::function<void(std::shared_ptr<InputFloat>)>,
               Callback);

public:
  inline float get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputFloat2) {
  UI_ATTRIBUTE(InputFloat2, std::string, Label);
  UI_ATTRIBUTE(InputFloat, glm::vec2, Value);
  UI_ATTRIBUTE(InputText, bool, ReadOnly);
  UI_ATTRIBUTE(InputText, std::function<void(std::shared_ptr<InputFloat2>)>,
               Callback);

public:
  inline glm::vec2 get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputFloat3) {
  UI_ATTRIBUTE(InputFloat3, std::string, Label);
  UI_ATTRIBUTE(InputFloat, glm::vec3, Value);
  UI_ATTRIBUTE(InputText, bool, ReadOnly);
  UI_ATTRIBUTE(InputText, std::function<void(std::shared_ptr<InputFloat3>)>,
               Callback);

public:
  inline glm::vec3 get() const { return mValue; }
  void build() override;
};

UI_CLASS(InputFloat4) {
  UI_ATTRIBUTE(InputFloat4, std::string, Label);
  UI_ATTRIBUTE(InputFloat, glm::vec4, Value);
  UI_ATTRIBUTE(InputText, bool, ReadOnly);
  UI_ATTRIBUTE(InputText, std::function<void(std::shared_ptr<InputFloat4>)>,
               Callback);

public:
  inline glm::vec4 get() const { return mValue; }
  void build() override;
};

} // namespace ui
} // namespace svulkan2
