#include "svulkan2/ui/radio_button_group.h"

namespace svulkan2 {
namespace ui {

void RadioButtonGroup::build() {
  for (uint32_t i = 0; i < mLabels.size(); ++i) {
    if (ImGui::RadioButton(mLabels[i].c_str(), &mIndex, i)) {
      if (mCallback) {
        mCallback(
            std::static_pointer_cast<RadioButtonGroup>(shared_from_this()));
      }
    }
  }
}

} // namespace ui
} // namespace svulkan2
