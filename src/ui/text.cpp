#include "svulkan2/ui/text.h"

namespace svulkan2 {
namespace ui {

void DisplayText::build() { ImGui::Text("%s", mText.c_str()); }
void InputText::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  flags |= ImGuiInputTextFlags_EnterReturnsTrue;
  if (ImGui::InputText(mLabel.c_str(), mBuffer.data(), mBuffer.size(), flags) &&
      mCallback) {
    mCallback(std::static_pointer_cast<InputText>(shared_from_this()));
  }
}
void InputFloat::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  flags |= ImGuiInputTextFlags_EnterReturnsTrue;
  if (ImGui::InputFloat(mLabel.c_str(), &mValue, 0.f, 0.f, "%.3f", flags) &&
      mCallback) {
    mCallback(std::static_pointer_cast<InputFloat>(shared_from_this()));
  }
}
void InputFloat2::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  flags |= ImGuiInputTextFlags_EnterReturnsTrue;
  if (ImGui::InputFloat2(mLabel.c_str(), &mValue[0], "%.3f", flags) &&
      mCallback) {
    mCallback(std::static_pointer_cast<InputFloat2>(shared_from_this()));
  }
}
void InputFloat3::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  flags |= ImGuiInputTextFlags_EnterReturnsTrue;
  if (ImGui::InputFloat3(mLabel.c_str(), &mValue[0], "%.3f", flags) &&
      mCallback) {
    mCallback(std::static_pointer_cast<InputFloat3>(shared_from_this()));
  }
}
void InputFloat4::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  flags |= ImGuiInputTextFlags_EnterReturnsTrue;
  if (ImGui::InputFloat4(mLabel.c_str(), &mValue[0], "%.3f", flags) &&
      mCallback) {
    mCallback(std::static_pointer_cast<InputFloat4>(shared_from_this()));
  }
}

} // namespace ui
} // namespace svulkan2
