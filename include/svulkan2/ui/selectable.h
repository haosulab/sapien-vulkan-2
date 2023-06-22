#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Selectable) {
  UI_DECLARE_LABEL(Selectable);
  UI_ATTRIBUTE(Selectable, bool, Selected);
  UI_ATTRIBUTE(Selectable, std::function<void(std::shared_ptr<Selectable>)>,
               Callback);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2
