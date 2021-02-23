#include "svulkan2/ui/widget.h"

namespace svulkan2 {
namespace ui {

void Widget::setParent(std::weak_ptr<Widget> parent) { mParent = parent; }

} // namespace ui
} // namespace svulkan2
