#include "svulkan2/ui/button.h"

namespace svulkan2 {
namespace ui {

void Button::build() {
  if (ImGui::Button(mLabel.c_str()) && mCallback) {
    mCallback(std::static_pointer_cast<Button>(shared_from_this()));
  }
}
} // namespace ui
} // namespace svulkan2
