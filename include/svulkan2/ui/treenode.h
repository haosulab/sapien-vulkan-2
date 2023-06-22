#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(TreeNode) {
  UI_DECLARE_APPEND(TreeNode);
  UI_DECLARE_LABEL(TreeNode);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
