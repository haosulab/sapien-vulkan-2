#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(RadioButtonGroup) {
  int mIndex{};
  UI_ATTRIBUTE(RadioButtonGroup, std::vector<std::string>, Labels);
  UI_ATTRIBUTE(RadioButtonGroup,
               std::function<void(std::shared_ptr<RadioButtonGroup>)>,
               Callback);

public:
  inline int getIndex() const { return mIndex; }
  inline std::string get() const { return mLabels[mIndex]; }

  void build() override;
};

} // namespace ui
} // namespace svulkan2
