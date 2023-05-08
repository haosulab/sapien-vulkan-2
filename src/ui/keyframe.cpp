#include "svulkan2/ui/keyframe.h"
#include <algorithm>
#include <imgui.h>
#include <string>

namespace svulkan2 {
namespace ui {

#ifndef IMGUI_DEFINE_MATH_OPERATORS
inline ImVec2 operator*(const ImVec2 &lhs, const float rhs) {
  return ImVec2(lhs.x * rhs, lhs.y * rhs);
}
inline ImVec2 operator/(const ImVec2 &lhs, const float rhs) {
  return ImVec2(lhs.x / rhs, lhs.y / rhs);
}
inline ImVec2 operator+(const ImVec2 &lhs, const ImVec2 &rhs) {
  return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y);
}
inline ImVec2 operator-(const ImVec2 &lhs, const ImVec2 &rhs) {
  return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y);
}
inline ImVec2 operator*(const ImVec2 &lhs, const ImVec2 &rhs) {
  return ImVec2(lhs.x * rhs.x, lhs.y * rhs.y);
}
inline ImVec2 operator/(const ImVec2 &lhs, const ImVec2 &rhs) {
  return ImVec2(lhs.x / rhs.x, lhs.y / rhs.y);
}
inline ImVec2 &operator*=(ImVec2 &lhs, const float rhs) {
  lhs.x *= rhs;
  lhs.y *= rhs;
  return lhs;
}
inline ImVec2 &operator/=(ImVec2 &lhs, const float rhs) {
  lhs.x /= rhs;
  lhs.y /= rhs;
  return lhs;
}
inline ImVec2 &operator+=(ImVec2 &lhs, const ImVec2 &rhs) {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  return lhs;
}
inline ImVec2 &operator-=(ImVec2 &lhs, const ImVec2 &rhs) {
  lhs.x -= rhs.x;
  lhs.y -= rhs.y;
  return lhs;
}
inline ImVec2 &operator*=(ImVec2 &lhs, const ImVec2 &rhs) {
  lhs.x *= rhs.x;
  lhs.y *= rhs.y;
  return lhs;
}
inline ImVec2 &operator/=(ImVec2 &lhs, const ImVec2 &rhs) {
  lhs.x /= rhs.x;
  lhs.y /= rhs.y;
  return lhs;
}
inline ImVec4 operator+(const ImVec4 &lhs, const ImVec4 &rhs) {
  return ImVec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}
inline ImVec4 operator-(const ImVec4 &lhs, const ImVec4 &rhs) {
  return ImVec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}
inline ImVec4 operator*(const ImVec4 &lhs, const ImVec4 &rhs) {
  return ImVec4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}
#endif

KeyFrameEditor::KeyFrameEditor(float contentScale_) {
  prevSelectedMaxFrame = selectedMaxFrame;

  if (contentScale_ < 0.1f) {
    contentScale = 1.0f;
  } else {
    contentScale = contentScale_;
  }

  zoom[1] *= contentScale;
  resetHorizZoom = true;

  CrossTheme.borderWidth *= contentScale;

  ListerTheme.width *= contentScale;
  ListerTheme.minWidth *= contentScale;
  ListerTheme.handleWidth *= contentScale;

  TimelineTheme.height *= contentScale;
  TimelineTheme.indicatorSize *= contentScale;

  EditorTheme.horizScrollbarPadding *= contentScale;
  EditorTheme.horizScrollbarHeight *= contentScale;
}

void KeyFrameEditor::build() {
  // Control panel
  ImGui::PushItemWidth(50.0f * contentScale);
  ImGui::DragInt("Current Frame", &currentFrame, 1.0f, frameRange[0],
                 frameRange[1], "%d", ImGuiSliderFlags_AlwaysClamp);
  ImGui::PopItemWidth();

  ImGui::SameLine();

  ImGui::PushItemWidth(100.0f * contentScale);
  const char *maxFrames[] = {"128", "256", "512", "1024"};
  if (ImGui::BeginCombo("Max Frame", maxFrames[selectedMaxFrame])) {
    for (int i = 0; i < IM_ARRAYSIZE(maxFrames); i++) {
      bool isSelected = (selectedMaxFrame == i);
      if (ImGui::Selectable(maxFrames[i], isSelected)) {
        selectedMaxFrame = i;
      }

      if (isSelected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::PopItemWidth();

  if (selectedMaxFrame != prevSelectedMaxFrame) {
    currentFrame = 0;
    frameRange[1] = std::stoi(maxFrames[selectedMaxFrame]);
    resetHorizZoom = true;
    prevSelectedMaxFrame = selectedMaxFrame;
  }

  ImGui::SameLine();

  // Key frame buttons
  auto it = std::find_if(keyFrames.begin(), keyFrames.end(),
                         [&](auto &kf) { return kf->frame == currentFrame; });
  if (it == keyFrames.end()) { // Not a key frame
    if (ImGui::Button("Insert Key Frame") && mInsertKeyFrameCallback) {
      keyFrames.push_back(
          std::unique_ptr<UIKeyFrame>(new UIKeyFrame{currentFrame}));
      std::sort(keyFrames.begin(), keyFrames.end(),
                [](auto &a, auto &b) { return a->frame < b->frame; });
      mInsertKeyFrameCallback(
          std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
    }
  } else {
    if (ImGui::Button("Load Key Frame") && mLoadKeyFrameCallback) {
      mLoadKeyFrameCallback(
          std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
    }

    ImGui::SameLine();

    if (ImGui::Button("Update Key Frame") && mUpdateKeyFrameCallback) {
      mUpdateKeyFrameCallback(
          std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
    }

    ImGui::SameLine();

    if (ImGui::Button("Delete Key Frame") && mDeleteKeyFrameCallback) {
      keyFrames.erase(it);
      mDeleteKeyFrameCallback(
          std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
    }
  }

  const ImGuiIO &io = ImGui::GetIO();
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
  const ImVec2 canvasSize = ImGui::GetContentRegionAvail();

  if (canvasSize.x <= 0 || canvasSize.y <= 0) {
    return;
  }

  // Update max stride and horizontal zoom range
  int frameCount = frameRange[1] - frameRange[0];
  int maxStride = frameCount / minIntervals;
  float minTimelineLength =
      std::max(canvasSize.x - ListerTheme.width, 1024.0f * contentScale);
  horizZoomRange[0] = minTimelineLength / frameCount;
  horizZoomRange[1] = horizZoomRange[0] * maxStride;

  // Clamp zoom
  if (resetHorizZoom) {
    zoom[0] = horizZoomRange[0];
    pan[0] = 0.0f;
    resetHorizZoom = false;
  } else {
    zoom[0] = ImClamp(zoom[0], horizZoomRange[0], horizZoomRange[1]);
  }

  // Update stride
  stride = 1;
  float minHorizZoom = horizZoomRange[1] / 2; // For stride == 1
  while (zoom[0] <= minHorizZoom) {
    stride *= 2;
    minHorizZoom /= 2;
  }

  /**
   * Background
   *
   *          _________________________________________________
   *         |       |                                         |
   *   Cross |   X   |                  B                      | Timeline
   *         |_______|_________________________________________|
   *         |       |                                         |
   *         |       |                                         |
   *         |       |                                         |
   *  Lister |   A   |                  C                      | Editor
   *         |       |                                         |
   *         |       |                                         |
   *         |_______|_________________________________________|
   *
   */
  ImVec2 X = canvasPos;
  ImVec2 A = canvasPos + ImVec2{0.0f, TimelineTheme.height};
  ImVec2 B = canvasPos + ImVec2{ListerTheme.width, 0.0f};
  ImVec2 C = canvasPos + ImVec2{ListerTheme.width, TimelineTheme.height};

  auto CrossBackground = [&]() {
    drawList->AddRectFilled(X,
                            X + ImVec2{ListerTheme.width, TimelineTheme.height},
                            ImColor(CrossTheme.background));

    drawList->AddLine(
        X + ImVec2{ListerTheme.width - 1.0f, 0.0f},
        X + ImVec2{ListerTheme.width - 1.0f, TimelineTheme.height},
        ImColor(CrossTheme.border), CrossTheme.borderWidth);

    drawList->AddLine(
        X + ImVec2{0.0f, TimelineTheme.height - 1.0f},
        X + ImVec2{ListerTheme.width, TimelineTheme.height - 1.0f},
        ImColor(CrossTheme.border), CrossTheme.borderWidth);
  };

  auto ListerBackground = [&]() {
    drawList->AddRectFilled(
        A, A + ImVec2{ListerTheme.width, canvasSize.y - TimelineTheme.height},
        ImColor(ListerTheme.background));
  };

  auto TimelineBackground = [&]() {
    drawList->AddRectFilled(
        B, B + ImVec2{canvasSize.x - ListerTheme.width, TimelineTheme.height},
        ImColor(TimelineTheme.background));
  };

  auto EditorBackground = [&]() {
    drawList->AddRectFilled(
        C, C + canvasSize - ImVec2{ListerTheme.width, TimelineTheme.height},
        ImColor(EditorTheme.background));
  };

  auto Timeline = [&]() {
    float interspace = zoom[0] * stride; // Distance between each base frame
    int start =
        ceil(-pan[0] / interspace) * stride; // Only draw in visible area
    start = ImClamp(start, frameRange[0], frameRange[1]);
    float yMin = B.y;
    float yMax = B.y + TimelineTheme.height;

    for (int frame = start; frame <= frameRange[1]; frame += stride) {
      int i = frame / stride; // ith line

      float x = B.x + i * interspace + pan[0];
      if (x > canvasPos.x + canvasSize.x) { // Only draw in visible area
        break;
      }

      drawList->AddLine(ImVec2(x, yMin), ImVec2(x, yMax),
                        ImColor(TimelineTheme.dark), 1.0f * contentScale);

      ImVec2 textSize =
          ImGui::CalcTextSize(std::to_string(frame).c_str()) * 0.85f;
      if (frame != frameRange[1] &&
          x + 5.0f * contentScale + textSize.x <=
              canvasPos.x + canvasSize.x) { // Only draw in visible area
        drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 0.85f,
                          ImVec2{x + 5.0f * contentScale, yMin},
                          ImColor(TimelineTheme.text),
                          std::to_string(frame).c_str());
      }
    }
  };

  auto FrameIndicator = [&](int frame, ImVec4 color, bool larger,
                            bool hasLine) {
    float x = B.x + frame * zoom[0] + pan[0];
    if (x < B.x || x > canvasPos.x + canvasSize.x) {
      return;
    }

    float innerSize = TimelineTheme.indicatorSize;
    float outerSize = innerSize * 2;
    ImVec2 center = ImVec2{x, C.y - outerSize / 2};

    // Calculate the inner diamond's corner points
    ImVec2 innerTop(center.x, center.y - innerSize / 2);
    ImVec2 innerRight(center.x + innerSize / 2, center.y);
    ImVec2 innerBottom(center.x, center.y + innerSize / 2);
    ImVec2 innerLeft(center.x - innerSize / 2, center.y);

    // Color and thickness for contour
    ImColor contourColor(0, 0, 0, 255);
    float contourThickness = 1.0f * contentScale;

    if (!larger) {
      drawList->AddQuadFilled(innerTop, innerRight, innerBottom, innerLeft,
                              ImColor(color));
      drawList->AddQuad(innerTop, innerRight, innerBottom, innerLeft,
                        contourColor, contourThickness);
    } else {
      // Calculate the outer diamond's corner points
      ImVec2 outerTop(center.x, center.y - outerSize / 2);
      ImVec2 outerRight(center.x + outerSize / 2, center.y);
      ImVec2 outerBottom(center.x, center.y + outerSize / 2);
      ImVec2 outerLeft(center.x - outerSize / 2, center.y);

      drawList->AddQuadFilled(outerTop, outerRight, outerBottom, outerLeft,
                              ImColor(color));
      drawList->AddQuad(outerTop, outerRight, outerBottom, outerLeft,
                        contourColor, contourThickness);
      drawList->AddQuad(innerTop, innerRight, innerBottom, innerLeft,
                        contourColor, contourThickness);
    }

    if (hasLine &&
        canvasSize.y - TimelineTheme.height > 0) { // Editor area exists
      float yMin = C.y;
      float yMax = C.y + canvasSize.y - TimelineTheme.height;
      float lineWidth = outerSize / 10;
      drawList->AddLine(ImVec2{x, yMin}, ImVec2{x, yMax}, ImColor(color),
                        lineWidth);
      drawList->AddLine(ImVec2{x - lineWidth / 2, yMin},
                        ImVec2{x - lineWidth / 2, yMax}, contourColor, 1.0f);
      drawList->AddLine(ImVec2{x + lineWidth / 2, yMin},
                        ImVec2{x + lineWidth / 2, yMax}, contourColor, 1.0f);
    }
  };

  auto VerticalGrid = [&]() {
    float interspace = zoom[0] * stride; // Distance between each base frame
    int start =
        ceil(-pan[0] / interspace) * stride; // Only draw in visible area
    start = ImClamp(start, frameRange[0], frameRange[1]);
    float xMidStart =
        C.x + start / stride * interspace + pan[0] - interspace / 2;
    float yMin = C.y;
    float yMax = C.y + canvasSize.y - TimelineTheme.height;

    if (stride > 1 && xMidStart > C.x) { // First mid line
      drawList->AddLine(ImVec2(xMidStart, yMin), ImVec2(xMidStart, yMax),
                        ImColor(EditorTheme.mid), 1.0f * contentScale);
    }

    for (int frame = start; frame <= frameRange[1]; frame += stride) {
      int i = frame / stride; // ith line

      float x = C.x + i * interspace + pan[0];
      if (x > canvasPos.x + canvasSize.x) { // Only draw in visible area
        break;
      }

      drawList->AddLine(ImVec2(x, yMin), ImVec2(x, yMax),
                        ImColor(EditorTheme.dark), 1.0f * contentScale);

      float xMid = x + interspace / 2;
      if (frame != frameRange[1] && stride > 1 &&
          xMid <
              canvasPos.x + canvasSize.x) { // Draw when stride > 1 and visible
        drawList->AddLine(ImVec2(xMid, yMin), ImVec2(xMid, yMax),
                          ImColor(EditorTheme.mid), 1.0f * contentScale);
      }
    }
  };

  ImGui::BeginGroup();

  // Editor elements
  if (canvasSize.y - TimelineTheme.height > 0 &&
      canvasSize.x - ListerTheme.width > 0) {
    EditorBackground();
    VerticalGrid();
  }

  // Timeline elements
  if (canvasSize.x - ListerTheme.width > 0) {
    TimelineBackground();
    Timeline();

    for (auto &keyFrame : keyFrames) { // Ascending order
      FrameIndicator(keyFrame->frame, TimelineTheme.keyFrame, false, false);
    }

    auto it = std::find_if(keyFrames.begin(), keyFrames.end(),
                           [&](auto &kf) { return kf->frame == currentFrame; });
    if (it == keyFrames.end()) {
      FrameIndicator(currentFrame, TimelineTheme.currentFrame, true, true);
    } else {
      FrameIndicator(currentFrame, TimelineTheme.keyFrame, true, true);
    }

    // Select/drag key frame
    for (int i = static_cast<int>(keyFrames.size()) - 1; i >= 0;
         i--) { // Desending order
      float x = B.x + (keyFrames[i])->frame * zoom[0] + pan[0];
      if (x < B.x || x > canvasPos.x + canvasSize.x) {
        continue;
      }
      float size = TimelineTheme.indicatorSize;
      ImVec2 topLeft = ImVec2{x - size / 2, C.y - size * 3 / 2};

      ImGui::PushID(i);
      ImGui::SetCursorPos(topLeft - ImGui::GetWindowPos());
      ImGui::InvisibleButton("##KeyFrame", ImVec2{size, size});

      if (ImGui::IsItemActive()) {
        ImVec4 color = ImVec4{
            TimelineTheme.keyFrame.x * 1.2f, TimelineTheme.keyFrame.y * 1.2f,
            TimelineTheme.keyFrame.z * 1.2f, TimelineTheme.keyFrame.w};
        FrameIndicator((keyFrames[i])->frame, color, false, false);

        int frameTemp = static_cast<int>(
            std::round((io.MousePos.x - B.x - pan[0]) / zoom[0]));
        (keyFrames[i])->frame =
            ImClamp(frameTemp, frameRange[0], frameRange[1]);
        currentFrame = (keyFrames[i])->frame;
      }

      if (ImGui::IsItemDeactivated()) {
        draggedIndex = i;
        draggedNewVal = (keyFrames[i])->frame;
        if (mDragKeyFrameCallback) {
          mDragKeyFrameCallback(
              std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
        }

        if (std::count_if(keyFrames.begin(), keyFrames.end(), [&](auto &kf) {
              return kf->frame == (keyFrames[i])->frame;
            }) == 2) { // Duplicate key frame
          keyFrames.erase(keyFrames.begin() + i);
        } else {
          std::sort(keyFrames.begin(), keyFrames.end(),
                    [](auto &a, auto &b) { return a->frame < b->frame; });
        }
      }
      ImGui::PopID();
    }
  }

  // Lister elements
  if (ListerTheme.width > 0 && canvasSize.y - TimelineTheme.height > 0) {
    ListerBackground();
  }

  // Cross elements
  if (ListerTheme.width > 0) {
    CrossBackground();
  }

  // Horizontal scrollbar
  if ((canvasSize.x - ListerTheme.width) / zoom[0] <
      frameRange[1] - frameRange[0]) { // Visible frame less than total
    float barPadding = EditorTheme.horizScrollbarPadding;
    float areaWidth = canvasSize.x - ListerTheme.width - 2 * barPadding;
    float height = EditorTheme.horizScrollbarHeight;
    float heightMin = canvasPos.y + canvasSize.y - height - 4.0f * contentScale;

    if (areaWidth > 0 && heightMin > C.y) {
      float offsetRatio =
          -pan[0] / (zoom[0] * ((frameRange[1] - frameRange[0])));
      float offsetWidth = areaWidth * offsetRatio;
      float visibleFrameCount = (canvasSize.x - ListerTheme.width) / zoom[0];
      float barWidthRatio = visibleFrameCount / (frameRange[1] - frameRange[0]);
      float barWidth = areaWidth * barWidthRatio;
      float barWidthVisual = (barWidth < barPadding - 1.0f * contentScale)
                                 ? barPadding - 1.0f * contentScale
                                 : barWidth; // Ensure grabbable

      ImVec2 barMin = ImVec2{C.x + barPadding + offsetWidth, heightMin};
      ImVec2 barMax = ImVec2{C.x + barPadding + offsetWidth + barWidthVisual,
                             heightMin + height};
      ImVec4 color = EditorTheme.horizScrollbar;

      ImGui::SetCursorPos(barMin - ImGui::GetWindowPos());
      ImGui::InvisibleButton("##HorizScrollbar",
                             ImVec2{barWidthVisual, height});

      if (ImGui::IsItemActivated()) {
        initialPan[0] = pan[0];
      }

      if (ImGui::IsItemActive()) {
        color = ImVec4{color.x * 1.2f, color.y * 1.2f, color.z * 1.2f, color.w};

        float minPan = -((frameRange[1] - frameRange[0]) * zoom[0] -
                         (canvasSize.x - ListerTheme.width));
        minPan = std::min(minPan, 0.0f);
        float deltaPan =
            (ImGui::GetMouseDragDelta().x / (areaWidth - barWidth)) * minPan;
        float panTemp = initialPan[0] + deltaPan;
        pan[0] = ImClamp(panTemp, minPan, 0.0f);
      }

      drawList->AddRectFilled(barMin, barMax, ImColor(color),
                              8.0f * contentScale);
    }
  }

  // Lister width handle
  float buttonLeftX = std::max(B.x - ListerTheme.handleWidth / 2, canvasPos.x);
  float buttonRightX =
      std::min(B.x + ListerTheme.handleWidth / 2, canvasPos.x + canvasSize.x);
  float adaptedHandleWidth = buttonRightX - buttonLeftX;
  if (adaptedHandleWidth > 0) {
    ImGui::SetCursorPos(ImVec2{buttonLeftX, B.y} - ImGui::GetWindowPos());
    ImGui::InvisibleButton("##ListerWidthHandle",
                           ImVec2{adaptedHandleWidth, canvasSize.y});

    if (ImGui::IsItemHovered()) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    if (ImGui::IsItemActivated()) {
      ListerTheme.initialWidth = ListerTheme.width;
    }

    if (ImGui::IsItemActive()) {
      float deltaWidth = ImGui::GetMouseDragDelta().x;
      ListerTheme.width = ImClamp(ListerTheme.initialWidth + deltaWidth,
                                  ListerTheme.minWidth, canvasSize.x);
    }
  }

  // Timeline background input
  if (canvasSize.x - ListerTheme.width > 0) {
    ImGui::SetCursorPos(ImVec2{B.x, B.y} - ImGui::GetWindowPos());
    ImGui::InvisibleButton(
        "##Timeline",
        ImVec2{canvasSize.x - ListerTheme.width, TimelineTheme.height});

    if (ImGui::IsItemHovered()) {
      // Zooming
      ImGui::SetItemUsingMouseWheel();
      float initialZoom = zoom[0];
      zoom[0] = ImClamp(zoom[0] + io.MouseWheel, horizZoomRange[0],
                        horizZoomRange[1]);

      // Change pan according to zoom
      if (zoom[0] != initialZoom) {
        float minHorizPan = -((frameRange[1] - frameRange[0]) * zoom[0] -
                              (canvasSize.x - ListerTheme.width));
        minHorizPan = std::min(minHorizPan, 0.0f);
        float panTemp = pan[0] / initialZoom * zoom[0];
        pan[0] = ImClamp(panTemp, minHorizPan, 0.0f);
      }
    }

    if (ImGui::IsItemActive()) {
      int frameTemp = static_cast<int>(
          std::round((io.MousePos.x - B.x - pan[0]) / zoom[0]));
      currentFrame = ImClamp(frameTemp, frameRange[0], frameRange[1]);
    }
  }

  // Editor background input
  if (canvasSize.x - ListerTheme.width > 0 &&
      canvasSize.y - TimelineTheme.height > 0) {
    ImGui::SetCursorPos(ImVec2{C.x, C.y} - ImGui::GetWindowPos());
    ImGui::InvisibleButton("##Editor",
                           ImVec2{canvasSize.x - ListerTheme.width,
                                  canvasSize.y - TimelineTheme.height});

    if (ImGui::IsItemHovered()) {
      // Horizontal scrolling
      ImGui::SetItemUsingMouseWheel();
      float minHorizPan = -((frameRange[1] - frameRange[0]) * zoom[0] -
                            (canvasSize.x - ListerTheme.width));
      minHorizPan = std::min(minHorizPan, 0.0f);
      pan[0] = ImClamp(pan[0] + io.MouseWheelH * zoom[0], minHorizPan, 0.0f);
    }
  }

  // Middle button panning
  if (ImGui::IsWindowFocused() && io.MouseDown[2]) {
    // Horizontal
    float minHorizPan = -((frameRange[1] - frameRange[0]) * zoom[0] -
                          (canvasSize.x - ListerTheme.width));
    minHorizPan = std::min(minHorizPan, 0.0f);
    float deltaHorizPan = io.MouseDelta.x;
    float horizPanTemp = pan[0] + deltaHorizPan;
    pan[0] = ImClamp(horizPanTemp, minHorizPan, 0.0f);
  }

  ImGui::EndGroup();
}

} // namespace ui
} // namespace svulkan2
