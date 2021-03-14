#include "svulkan2/ui/options.h"

namespace svulkan2 {
namespace ui {

std::string Options::get() {
  if (mIndex < 0 || static_cast<uint32_t>(mIndex) >= mItems.size()) {
    mIndex = 0;
  }
  return mItems[mIndex];
}

void Options::build() {
  if (mIndex < 0 || static_cast<uint32_t>(mIndex) >= mItems.size()) {
    mIndex = 0;
  }
  if (mStyle == "radio") {
    for (uint32_t i = 0; i < mItems.size(); ++i) {
      if (ImGui::RadioButton(mItems[i].c_str(), &mIndex, i)) {
        if (mCallback) {
          mCallback(std::static_pointer_cast<Options>(shared_from_this()));
        }
      }
    }
  } else if (mStyle == "select") {
    if (ImGui::BeginCombo(mLabel.c_str(), get().c_str())) {
      for (uint32_t i = 0; i < mItems.size(); ++i) {
        if (ImGui::Selectable(mItems[i].c_str(),
                              mIndex == static_cast<int>(i))) {
          mIndex = i;
          if (mCallback) {
            mCallback(std::static_pointer_cast<Options>(shared_from_this()));
          }
        }
      }
      ImGui::EndCombo();
    }
  } else {
    throw std::runtime_error(
        "\"radio\" or \"select\" style must be specified for UI options.");
  }
}

} // namespace ui
} // namespace svulkan2
