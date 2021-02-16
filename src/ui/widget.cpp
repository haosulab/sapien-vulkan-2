#include "svulkan2/ui/widget.h"

namespace svulkan2 {
namespace ui {

std::shared_ptr<Widget> Widget::Name(std::string const &name) {
  mName = name;
  return shared_from_this();
}

void Widget::setParent(std::shared_ptr<Widget> parent) { mParent = parent; }

std::shared_ptr<Widget> Widget::append(std::shared_ptr<Widget> child) {
  mChildren.push_back(child);
  return shared_from_this();
};

} // namespace ui
} // namespace svulkan2
