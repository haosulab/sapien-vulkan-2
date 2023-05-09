#include "svulkan2/ui/slider.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void SliderFloat::build() {
  if (mWidth > 0) {
    ImGui::PushItemWidth(mWidth);
  }
  if (ImGui::SliderFloat(mLabel.c_str(), &mValue, mMin, mMax) && mCallback) {
    mCallback(std::static_pointer_cast<SliderFloat>(shared_from_this()));
  }
  if (mWidth > 0) {
    ImGui::PopItemWidth();
  }
}

void SliderAngle::build() {
  if (mWidth > 0) {
    ImGui::PushItemWidth(mWidth);
  }
  if (ImGui::SliderAngle(mLabel.c_str(), &mValue, mMin, mMax) && mCallback) {
    mCallback(std::static_pointer_cast<SliderAngle>(shared_from_this()));
  }
  if (mWidth > 0) {
    ImGui::PopItemWidth();
  }
}

} // namespace ui
} // namespace svulkan2
