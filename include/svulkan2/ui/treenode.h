#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(TreeNode) {
  UI_DECLARE_APPEND(TreeNode);
  UI_ATTRIBUTE(TreeNode, std::string, Label);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
