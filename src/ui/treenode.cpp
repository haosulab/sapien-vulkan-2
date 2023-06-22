#include "svulkan2/ui/treenode.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void TreeNode::build() {
  if (ImGui::TreeNode(getLabelId().c_str())) {
    for (auto c : mChildren) {
      c->build();
    }
    ImGui::TreePop();
  }
}
} // namespace ui
} // namespace svulkan2
