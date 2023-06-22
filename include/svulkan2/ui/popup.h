#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Popup) {
  UI_DECLARE_APPEND(Popup);
  UI_DECLARE_LABEL(Popup);

  UI_ATTRIBUTE(Popup, std::function<void()>, EscCallback);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
