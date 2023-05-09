#include "svulkan2/ui/widget.h"
#include <algorithm>
#include <imgui.h>

namespace svulkan2 {
namespace ui {

void Widget::setParent(std::weak_ptr<Widget> parent) { mParent = parent; }

void Widget::remove() {
  if (!mParent.expired()) {
    auto p = mParent.lock();
    auto it =
        std::find(p->mChildren.begin(), p->mChildren.end(), shared_from_this());
    p->mChildren.erase(it);
  }
}

void Widget::removeChildren() { mChildren.clear(); }

} // namespace ui
} // namespace svulkan2
