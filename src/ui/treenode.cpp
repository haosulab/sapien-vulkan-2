#include "svulkan2/ui/treenode.h"

namespace svulkan2 {
namespace ui {

void TreeNode::build() {
  if (ImGui::TreeNode(mLabel.c_str())) {
    for (auto c : mChildren) {
      c->build();
    }
    ImGui::TreePop();
  }
}
} // namespace ui
} // namespace svulkan2
