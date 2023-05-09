#include "svulkan2/ui/sameline.h"
#include <imgui.h>
#include <stdexcept>

namespace svulkan2 {
namespace ui {

SameLine::SameLine() : mSpacing(-1.f) {}

void SameLine::build() {
  if (mChildren.size() == 0) {
    throw std::runtime_error("failed to build SameLine: no children.");
  }
  for (uint32_t i = 0; i < mChildren.size() - 1; ++i) {
    mChildren[i]->build();
    ImGui::SameLine(mOffset, mSpacing);
  }
  mChildren.back()->build();
}
} // namespace ui
} // namespace svulkan2
