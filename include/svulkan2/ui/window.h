#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(Window) {
  UI_DECLARE_APPEND(Window);
  UI_ATTRIBUTE(Window, std::string, Label);
  UI_ATTRIBUTE(Window, ImVec2, Pos);
  UI_ATTRIBUTE(Window, ImVec2, Size);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
