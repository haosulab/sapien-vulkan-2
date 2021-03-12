#include "svulkan2/ui/radio_button_group.h"

namespace svulkan2 {
namespace ui {

int RadioButtonGroup::getIndex() {
  if (mIndex < 0 || static_cast<uint32_t>(mIndex) >= mLabels.size()) {
    mIndex = 0;
  }
  return mIndex;
}
std::string RadioButtonGroup::get() {
  if (mIndex < 0 || static_cast<uint32_t>(mIndex) >= mLabels.size()) {
    mIndex = 0;
  }
  return mLabels[mIndex];
}

void RadioButtonGroup::build() {
  if (mIndex < 0 || static_cast<uint32_t>(mIndex) >= mLabels.size()) {
    mIndex = 0;
  }
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
