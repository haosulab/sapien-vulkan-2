#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Checkbox) {
  UI_ATTRIBUTE(Checkbox, bool, Checked);
  UI_ATTRIBUTE(Checkbox, std::string, Label);
  UI_ATTRIBUTE(Checkbox, std::function<void(std::shared_ptr<Checkbox>)>,
               Callback);

public:
  inline bool get() const { return mChecked; }

  void build() override;
};

} // namespace ui
} // namespace svulkan2
