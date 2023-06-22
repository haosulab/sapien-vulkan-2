#include "svulkan2/ui/options.h"
#include <imgui.h>
#include <stdexcept>

namespace svulkan2 {
namespace ui {

std::string Options::get() {
  if (mIndex < 0 || static_cast<uint32_t>(mIndex) >= mItems.size()) {
    mIndex = 0;
  }
  return mItems[mIndex];
}

void Options::build() {
  if (mItemsGetter) {
    mItems = mItemsGetter();
  }
  if (mIndexGetter) {
    mIndex = mIndexGetter();
  }

  if (mIndex < 0 || static_cast<uint32_t>(mIndex) >= mItems.size()) {
    mIndex = 0;
  }
  if (mStyle == "radio") {
    for (uint32_t i = 0; i < mItems.size(); ++i) {
      if (ImGui::RadioButton(mItems[i].c_str(), &mIndex, i)) {
        if (mCallback) {
          mCallback(std::static_pointer_cast<Options>(shared_from_this()));
        }
        if (mIndexSetter) {
          mIndexSetter(mIndex);
        }
      }
    }
  } else if (mStyle == "select") {
    if (ImGui::BeginCombo(getLabelId().c_str(), get().c_str())) {
      for (uint32_t i = 0; i < mItems.size(); ++i) {
        ImGui::PushID(i);
        if (ImGui::Selectable(mItems[i].c_str(), mIndex == static_cast<int>(i))) {
          mIndex = i;
          if (mCallback) {
            mCallback(std::static_pointer_cast<Options>(shared_from_this()));
          }
          if (mIndexSetter) {
            mIndexSetter(mIndex);
          }
        }
        ImGui::PopID();
      }
      ImGui::EndCombo();
    }
  } else {
    throw std::runtime_error("\"radio\" or \"select\" style must be specified for UI options.");
  }
}

} // namespace ui
} // namespace svulkan2
