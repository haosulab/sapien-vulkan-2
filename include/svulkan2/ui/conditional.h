#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(Conditional) {
  UI_DECLARE_APPEND(Conditional);
  UI_BINDING_READONLY(Conditional, bool, Condition);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
