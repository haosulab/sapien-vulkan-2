#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Button) {
  UI_ATTRIBUTE(Button, std::string, Label);
  UI_ATTRIBUTE(Button, std::function<void(std::shared_ptr<Button>)>, Callback);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
