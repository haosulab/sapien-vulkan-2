#include "svulkan2/ui/text.h"

namespace svulkan2 {
namespace ui {

void DisplayText::build() { ImGui::Text("%s", mText.c_str()); }
void InputText::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  ImGui::InputText(mLabel.c_str(), mBuffer.data(), mBuffer.size(), flags);
}
void InputFloat::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  ImGui::InputFloat(mLabel.c_str(), &mValue, 0.f, 0.f, "%.3f", flags);
}
void InputFloat2::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  ImGui::InputFloat2(mLabel.c_str(), &mValue[0], "%.3f", flags); }
void InputFloat3::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  ImGui::InputFloat3(mLabel.c_str(), &mValue[0], "%.3f", flags); }
void InputFloat4::build() {
  ImGuiInputTextFlags flags = mReadOnly ? ImGuiInputTextFlags_ReadOnly : 0;
  ImGui::InputFloat4(mLabel.c_str(), &mValue[0], "%.3f", flags); }

} // namespace ui
} // namespace svulkan2
