#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(SliderFloat) {
  UI_ATTRIBUTE(SliderFloat, float, WidthRatio);
  UI_ATTRIBUTE(SliderFloat, float, Value);
  UI_DECLARE_LABEL(SliderFloat);
  UI_ATTRIBUTE(SliderFloat, float, Min);
  UI_ATTRIBUTE(SliderFloat, float, Max);
  UI_ATTRIBUTE(SliderFloat, std::function<void(std::shared_ptr<SliderFloat>)>, Callback);

  UI_BINDING(SliderFloat, float, Value);

public:
  inline float get() const { return mValue; }

  void build() override;
};

UI_CLASS(SliderAngle) {
  UI_ATTRIBUTE(SliderAngle, float, WidthRatio);
  UI_ATTRIBUTE(SliderAngle, float, Value);
  UI_DECLARE_LABEL(SliderAngle);
  UI_ATTRIBUTE(SliderAngle, float, Min);
  UI_ATTRIBUTE(SliderAngle, float, Max);
  UI_ATTRIBUTE(SliderAngle, std::function<void(std::shared_ptr<SliderAngle>)>, Callback);

  UI_BINDING(SliderAngle, float, Value);

public:
  inline float get() const { return mValue; }

  void build() override;
};

} // namespace ui
} // namespace svulkan2
