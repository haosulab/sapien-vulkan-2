#include "svulkan2/ui/dummy.h"
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void Dummy::build() { ImGui::Dummy({mWidth, mHeight}); }
} // namespace ui
} // namespace svulkan2
