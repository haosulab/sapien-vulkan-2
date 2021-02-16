#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

class DisplayText : public Widget {
protected:
  std::string mText;

public:
  void build() override;
  inline std::shared_ptr<DisplayText> Text(std::string const &text) {
    mText = text;
    return std::static_pointer_cast<DisplayText>(shared_from_this());
  };
};

class InputText : public Widget {
protected:
  std::string mLabel;
  std::vector<char> mBuffer{std::vector<char>(100, 0)};

public:
  void build() override;
  inline std::shared_ptr<InputText> Label(std::string const &label) {
    mLabel = label;
    return std::static_pointer_cast<InputText>(shared_from_this());
  };
  inline std::shared_ptr<InputText> Size(uint32_t size) {
    mBuffer.resize(size);
    return std::static_pointer_cast<InputText>(shared_from_this());
  };
  inline std::string get() const {
    return std::string(mBuffer.begin(), mBuffer.end());
  }
};

class InputFloat : public Widget {
protected:
  std::string mLabel;
  float mValue;

public:
  void build() override;
  inline std::shared_ptr<InputText> Label(std::string const &label) {
    mLabel = label;
    return std::static_pointer_cast<InputText>(shared_from_this());
  };
  inline float get() const { return mValue; }
};

class InputFloat2 : public Widget {
protected:
  std::string mLabel;
  std::array<float, 2> mValue{0, 0};

public:
  void build() override;
  inline std::shared_ptr<InputText> Label(std::string const &label) {
    mLabel = label;
    return std::static_pointer_cast<InputText>(shared_from_this());
  };
  inline std::array<float, 2> get() const { return mValue; }
};

class InputFloat3 : public Widget {
protected:
  std::string mLabel;
  std::array<float, 3> mValue{0, 0};

public:
  void build() override;
  inline std::shared_ptr<InputText> Label(std::string const &label) {
    mLabel = label;
    return std::static_pointer_cast<InputText>(shared_from_this());
  };
  inline std::array<float, 3> get() const { return mValue; }
};

class InputFloat4 : public Widget {
protected:
  std::string mLabel;
  std::array<float, 4> mValue{0, 0};

public:
  void build() override;
  inline std::shared_ptr<InputText> Label(std::string const &label) {
    mLabel = label;
    return std::static_pointer_cast<InputText>(shared_from_this());
  };
  inline std::array<float, 4> get() const { return mValue; }
};

} // namespace ui
} // namespace svulkan2
