#include "svulkan2/ui/slider.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void SliderFloat::build() {
  if (mWidthRatio > 0) {
    ImGui::SetNextItemWidth(mWidthRatio * ImGui::GetWindowContentRegionWidth());
  }
  if (mValueGetter) {
    mValue = mValueGetter();
  }
  if (ImGui::SliderFloat(getLabelId().c_str(), &mValue, mMin, mMax)) {
    if (mValueSetter) {
      mValueSetter(mValue);
    }
    if (mCallback)
      mCallback(std::static_pointer_cast<SliderFloat>(shared_from_this()));
  }
}

void SliderAngle::build() {
  if (mWidthRatio > 0) {
    ImGui::SetNextItemWidth(mWidthRatio * ImGui::GetWindowContentRegionWidth());
  }
  if (mValueGetter) {
    mValue = mValueGetter();
  }
  if (ImGui::SliderAngle(getLabelId().c_str(), &mValue, mMin, mMax)) {
    if (mValueSetter) {
      mValueSetter(mValue);
    }
    if (mCallback) {
      mCallback(std::static_pointer_cast<SliderAngle>(shared_from_this()));
    }
  }
}

} // namespace ui
} // namespace svulkan2
