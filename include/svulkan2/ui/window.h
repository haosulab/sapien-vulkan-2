#pragma once
#include "widget.h"

namespace svulkan2 {
namespace ui {

class Window : public Widget {
protected:
  ImVec2 mPos{0, 0};
  ImVec2 mSize{100, 100};

public:
  void build() override;

  inline std::shared_ptr<Window> Pos(float x, float y) {
    mPos = {x, y};
    return std::static_pointer_cast<Window>(shared_from_this());
  };
  inline std::shared_ptr<Window> Size(float x, float y) {
    mSize = {x, y};
    return std::static_pointer_cast<Window>(shared_from_this());
  };
};
} // namespace ui
} // namespace svulkan2
