#pragma once
#include "widget.h"
#include <functional>
#include <imgui_internal.h>
#include <set>

namespace svulkan2 {
namespace ui {

UI_CLASS(KeyFrameEditor) {
  UI_ATTRIBUTE(KeyFrameEditor,
               std::function<void(std::shared_ptr<KeyFrameEditor>)>, Callback);

public:
  KeyFrameEditor(float contentScale_);
  void build() override;

private:
  // Timeline
  int currentFrame{0};
  int frameRange[2]{0, 128};
  int stride{8};
  int minIntervals{16}; // maxStride = frameRange / minIntervals
  int selectedMaxFrame{0};
  int prevSelectedMaxFrame;
  std::set<int> keyFrames;

  // Visual
  float contentScale;

  float pan[2]{0.0f, 0.0f}; // Deviation of {timeline, lister} in pixels
  float initialPan[2];

  float zoom[2]{10.0f,
                10.0f}; // Distance between each {frame, object} in pixels
  float horizZoomRange[2];
  bool resetHorizZoom;

  // Theme
  struct CrossTheme_ {
    float borderWidth{1.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.114f)};
    ImVec4 border{ImColor::HSV(0.0f, 0.0f, 0.2f)};
  } CrossTheme;

  struct ListerTheme_ {
    float width{100.0f};
    float minWidth{100.0f};
    float handleWidth{10.0f};
    float initialWidth;

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.114f)};
  } ListerTheme;

  struct TimelineTheme_ {
    float height{40.0f};
    float indicatorSize{10.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.114f)};
    ImVec4 text{ImColor::HSV(0.0f, 0.0f, 0.7f)};
    ImVec4 dark{ImColor::HSV(0.0f, 0.0f, 0.075f)};
    ImVec4 currentFrame{ImColor::HSV(0.6f, 0.6f, 0.95f)};
    ImVec4 keyFrame{ImColor::HSV(0.12f, 0.8f, 0.95f)};
  } TimelineTheme;

  struct EditorTheme_ {
    float horizScrollbarPadding{20.0f};
    float horizScrollbarHeight{10.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.188f)};
    ImVec4 dark{ImColor::HSV(0.0f, 0.0f, 0.075f)};
    ImVec4 mid{ImColor::HSV(0.0f, 0.0f, 0.15f)};
    ImVec4 horizScrollbar{ImColor::HSV(0.0f, 0.0f, 0.33f)};
  } EditorTheme;
};

} // namespace ui
} // namespace svulkan2
