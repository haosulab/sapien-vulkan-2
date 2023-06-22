#include "svulkan2/ui/checkbox.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void Checkbox::build() {
  if (mCheckedGetter) {
    mChecked = mCheckedGetter();
  }
  if (ImGui::Checkbox(getLabelId().c_str(), &mChecked)) {
    if (mCheckedSetter) {
      mCheckedSetter(mChecked);
    }
    if (mCallback) {
      mCallback(std::static_pointer_cast<Checkbox>(shared_from_this()));
    }
  }
}

} // namespace ui
} // namespace svulkan2
