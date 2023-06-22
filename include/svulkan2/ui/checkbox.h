#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Checkbox) {
  UI_DECLARE_LABEL(Checkbox);
  UI_ATTRIBUTE(Checkbox, bool, Checked);
  UI_ATTRIBUTE(Checkbox, std::function<void(std::shared_ptr<Checkbox>)>, Callback);

  UI_BINDING(Checkbox, bool, Checked);

public:
  inline bool get() const { return mChecked; }

  void build() override;
};

} // namespace ui
} // namespace svulkan2
