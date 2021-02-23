#include "svulkan2/ui/slider.h"

namespace svulkan2 {
namespace ui {

void SliderFloat::build() {
  if (ImGui::SliderFloat(mLabel.c_str(), &mValue, mMin, mMax) && mCallback) {
    mCallback(std::static_pointer_cast<SliderFloat>(shared_from_this()));
  }
}

void SliderAngle::build() {
  if (ImGui::SliderAngle(mLabel.c_str(), &mValue, mMin, mMax) && mCallback) {
    mCallback(std::static_pointer_cast<SliderAngle>(shared_from_this()));
  }
}

} // namespace ui
} // namespace svulkan2
