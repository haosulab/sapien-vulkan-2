#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(Section) {
  UI_DECLARE_APPEND(Section);
  UI_ATTRIBUTE(Section, bool, Expanded);
  UI_ATTRIBUTE(Section, std::string, Label);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
