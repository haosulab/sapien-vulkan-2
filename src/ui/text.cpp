#include "svulkan2/ui/text.h"

namespace svulkan2 {
namespace ui {

void DisplayText::build() { ImGui::Text("%s", mText.c_str()); }
void InputText::build() {
  ImGui::InputText(mLabel.c_str(), mBuffer.data(), mBuffer.size());
}

void InputFloat::build() { ImGui::InputFloat(mLabel.c_str(), &mValue); }
void InputFloat2::build() { ImGui::InputFloat2(mLabel.c_str(), mValue.data()); }
void InputFloat3::build() { ImGui::InputFloat3(mLabel.c_str(), mValue.data()); }
void InputFloat4::build() { ImGui::InputFloat4(mLabel.c_str(), mValue.data()); }

} // namespace ui
} // namespace svulkan2
