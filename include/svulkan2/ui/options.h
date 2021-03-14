#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(Options) {
  UI_ATTRIBUTE(Options, int, Index);
  UI_ATTRIBUTE(Options, std::string, Label);
  UI_ATTRIBUTE(Options, std::vector<std::string>, Items);
  UI_ATTRIBUTE(Options, std::function<void(std::shared_ptr<Options>)>,
               Callback);
  UI_ATTRIBUTE(Options, std::string, Style);

public:
  std::string get();

  void build() override;
};

} // namespace ui
} // namespace svulkan2
