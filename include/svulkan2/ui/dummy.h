#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(Dummy) {
  UI_ATTRIBUTE(Dummy, float, Width);
  UI_ATTRIBUTE(Dummy, float, Height);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
