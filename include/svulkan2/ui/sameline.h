#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(SameLine) {
  UI_DECLARE_APPEND(SameLine);
  UI_ATTRIBUTE(SameLine, float, Spacing);
  UI_ATTRIBUTE(SameLine, float, Offset);

public:
  SameLine();
  void build() override;
};

} // namespace ui
} // namespace svulkan2
