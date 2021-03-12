#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(RadioButtonGroup) {
  UI_ATTRIBUTE(RadioButtonGroup, int, Index);
  UI_ATTRIBUTE(RadioButtonGroup, std::vector<std::string>, Labels);
  UI_ATTRIBUTE(RadioButtonGroup,
               std::function<void(std::shared_ptr<RadioButtonGroup>)>,
               Callback);

public:
  int getIndex();
  std::string get();

  void build() override;
};

} // namespace ui
} // namespace svulkan2
