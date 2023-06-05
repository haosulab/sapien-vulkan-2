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
  // Control panel
  addingReward = false;

  // Timeline
  currentFrame = 0;
  totalFrame = 100;

  // Key frame
  keyFrameIdGenerator = IdGenerator();
  keyFramesInUsedUpdated = false;

  // Reward
  rewardIdGenerator = IdGenerator();
  selectedReward = -1;
  defaultRewardDefinition = "reward = 0";

  // Visual
  if (contentScale_ < 0.1f) {
    contentScale = 1.0f;
  } else {
    contentScale = contentScale_;
  }
  pan[0] = 0.0f;
  pan[1] = 0.0f;
  zoom[0] = 25.0f * contentScale;
  horizZoomRange[1] = 100.0f * contentScale;

  // Theme
  CrossTheme.borderWidth *= contentScale;
  ListerTheme.width *= contentScale;
  ListerTheme.handleWidth *= contentScale;
  TimelineTheme.height *= contentScale;
  TimelineTheme.keyFrameIndicatorSize *= contentScale;
  TimelineTheme.currentFrameIndicatorSize *= contentScale;
  EditorTheme.scrollbarPadding *= contentScale;
  EditorTheme.scrollbarSize *= contentScale;
  EditorTheme.rewardHeight *= contentScale;
}

void KeyFrameEditor::build() {
  auto ControlPanel = [&]() {
    if (addingReward) { // Reward adding mode
      if (ImGui::Button("Exit")) {
        keyFramesForNewReward.clear();
        addingReward = false;
      }

      if (keyFramesForNewReward.size() == 0) {
        ImGui::SameLine();
        ImGui::Text("Select first key frame for the new reward:");
      } else if (keyFramesForNewReward.size() == 1) {
        ImGui::SameLine();
        ImGui::Text("Select second key frame for the new reward:");
      }
    } else { // Normal mode
      // Current frame setter
      ImGui::PushItemWidth(50.0f * contentScale);
      ImGui::DragInt("Current Frame", &currentFrame, 1.0f, 0, totalFrame - 1,
                     "%d", ImGuiSliderFlags_AlwaysClamp);
      ImGui::PopItemWidth();

      // Total frame setter
      int minTotalFrame = totalFrameRange[0];
      if (keyFramesInUsed.size() != 0) {
        auto lastKf = keyFramesInUsed.back();
        minTotalFrame = std::max(minTotalFrame, lastKf->frame + 1);
      }
      ImGui::SameLine();
      ImGui::PushItemWidth(50.0f * contentScale);
      ImGui::DragInt("Total Frame", &totalFrame, 1.0f, minTotalFrame,
                     totalFrameRange[1], "%d", ImGuiSliderFlags_AlwaysClamp);
      ImGui::PopItemWidth();
      currentFrame =
          ImClamp(currentFrame, 0, totalFrame - 1); // Clamp currentFrame

      // Key frame control
      auto it =
          std::find_if(keyFramesInUsed.begin(), keyFramesInUsed.end(),
                       [&](auto &kf) { return kf->frame == currentFrame; });
      if (it == keyFramesInUsed.end()) { // Not a key frame
        ImGui::SameLine();
        if (ImGui::Button("Insert Key Frame") && mInsertKeyFrameCallback) {
          mInsertKeyFrameCallback(
              std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
          auto kf = std::make_shared<KeyFrame>(keyFrameIdGenerator.next(),
                                               currentFrame);
          keyFrames.push_back(kf);
          keyFramesInUsed.push_back(kf);
          std::sort(keyFramesInUsed.begin(), keyFramesInUsed.end(),
                    [](auto &a, auto &b) { return a->frame < b->frame; });
        }
      } else {
        keyFrameToModify = (*it)->getId();

        ImGui::SameLine();
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
          // Delete all rewards depending on the key frame
          for (auto it = rewardsInUsed.begin(); it != rewardsInUsed.end();) {
            auto reward = *it;
            if (reward->kf1Id == keyFrameToModify ||
                reward->kf2Id == keyFrameToModify) {
              rewards[reward->getId()] = nullptr;
              it = rewardsInUsed.erase(it);
            } else {
              it++;
            }
          }

          // Delete key frame
          mDeleteKeyFrameCallback(
              std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
          keyFrames[keyFrameToModify] = nullptr;
          keyFramesInUsed.erase(it);
        }
      }

      // Reward control
      if (keyFramesInUsed.size() >= 2) { // Exists at least two key frames
        ImGui::SameLine();
        if (ImGui::Button("Add Reward")) {
          addingReward = true;
        }
      }

      // Export & import current work
      ImGui::SameLine();
      if (ImGui::Button("Export") && mExportCallback) {
        mExportCallback(
            std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
      }

      ImGui::SameLine();
      if (ImGui::Button("Import") && mImportCallback) {
        mImportCallback(
            std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
      }
    }
  };

  ControlPanel();

  const ImGuiIO &io = ImGui::GetIO();
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  const ImVec2 canvasPos =
      ImVec2{ImGui::GetWindowPos().x, ImGui::GetCursorScreenPos().y};
  const ImVec2 canvasSize =
      ImVec2{ImGui::GetWindowSize().x, ImGui::GetContentRegionAvail().y};

  if (canvasSize.x <= 0 || canvasSize.y <= 0) {
    return;
  }

  // Clamp lister width
  ListerTheme.width = ImClamp(ListerTheme.width, 0.0f, canvasSize.x);

  // Update minimum horizontal zoom range
  float minTimelineLength = std::max(
      canvasSize.x - ListerTheme.width - 0.01f,
      1024.0f * contentScale); // 0.01f is to make sure no numerical error will
                               // cause horizZoomRange[0] * totalFrame >
                               // canvasSize.x - ListerTheme.width
  horizZoomRange[0] = minTimelineLength / totalFrame;

  // Clamp zoom
  float initialZoom = zoom[0];
  zoom[0] = ImClamp(zoom[0], horizZoomRange[0], horizZoomRange[1]);

  // Change pan according to zoom
  float minHorizPan =
      -(totalFrame * zoom[0] - (canvasSize.x - ListerTheme.width));
  minHorizPan = std::min(minHorizPan, 0.0f);
  float panTemp = pan[0] / initialZoom * zoom[0];
  pan[0] = ImClamp(panTemp, minHorizPan, 0.0f);

  // Update stride
  stride = 1;
  float minHorizZoom = horizZoomRange[1] / 2; // For stride == 1
  while (zoom[0] <= minHorizZoom) {
    stride *= 2;
    minHorizZoom /= 2;
  }

  // Compute vertical zoom
  ImVec2 textSize = ImGui::CalcTextSize("test");
  zoom[1] = std::max(textSize.y, EditorTheme.rewardHeight) * 1.5;

  // Clampe vertical pan
  float minVertPan =
      -(static_cast<int>(rewardsInUsed.size()) * zoom[1] + 5.0f * contentScale -
        (canvasSize.y - TimelineTheme.height));
  minVertPan = std::min(minVertPan, 0.0f);
  pan[1] = ImClamp(pan[1], minVertPan, 0.0f);

  /**
   * Division
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

  auto EditorBackground = [&]() {
    drawList->AddRectFilled(
        C, C + canvasSize - ImVec2{ListerTheme.width, TimelineTheme.height},
        ImColor(EditorTheme.background));

    // Mouse event
    ImGui::SetCursorPos(ImVec2{C.x, C.y} - ImGui::GetWindowPos());
    ImGui::InvisibleButton("##EditorBackground",
                           ImVec2{canvasSize.x - ListerTheme.width,
                                  canvasSize.y - TimelineTheme.height});
    ImGui::SetItemAllowOverlap();

    if (ImGui::IsItemHovered()) {
      ImGui::SetItemUsingMouseWheel();

      // Horizontal scrolling
      float minHorizPan =
          -(totalFrame * zoom[0] - (canvasSize.x - ListerTheme.width));
      minHorizPan = std::min(minHorizPan, 0.0f);
      pan[0] = ImClamp(pan[0] + io.MouseWheelH * zoom[0], minHorizPan, 0.0f);

      // Vertical scrolling
      float minVertPan =
          -(static_cast<int>(rewardsInUsed.size()) * zoom[1] +
            5.0f * contentScale - (canvasSize.y - TimelineTheme.height));
      minVertPan = std::min(minVertPan, 0.0f);
      pan[1] = ImClamp(pan[1] + io.MouseWheel * zoom[1], minVertPan, 0.0f);
    }
  };

  auto VerticalGrid = [&]() {
    float interspace = zoom[0] * stride; // Distance between each base frame
    int frameStart =
        ceil(-pan[0] / interspace) * stride; // Only draw in visible area
    frameStart = ImClamp(frameStart, 0, totalFrame - 1);
    float xMidStart =
        C.x + frameStart / stride * interspace + pan[0] - interspace / 2;
    float yMin = C.y;
    float yMax = C.y + canvasSize.y - TimelineTheme.height;

    if (stride > 1 && xMidStart > C.x) { // First mid line
      drawList->AddLine(ImVec2(xMidStart, yMin), ImVec2(xMidStart, yMax),
                        ImColor(EditorTheme.mid), 1.0f * contentScale);
    }

    for (int frame = frameStart; frame <= totalFrame - 1; frame += stride) {
      int i = frame / stride; // ith line

      float x = C.x + i * interspace + pan[0];
      if (x > canvasPos.x + canvasSize.x) { // Only draw in visible area
        break;
      }

      drawList->AddLine(ImVec2(x, yMin), ImVec2(x, yMax),
                        ImColor(EditorTheme.dark), 1.0f * contentScale);

      float xMid = x + interspace / 2;
      if (stride > 1 &&
          xMid <
              canvasPos.x + canvasSize.x) { // Draw when stride > 1 and visible
        drawList->AddLine(ImVec2(xMid, yMin), ImVec2(xMid, yMax),
                          ImColor(EditorTheme.mid), 1.0f * contentScale);
      }
    }
  };

  auto Editor = [&]() {
    for (int i = 0; i < static_cast<int>(rewardsInUsed.size()); i++) {
      float y = std::round(A.y + 5.0f * contentScale + zoom[1] * i +
                           pan[1]); // Avoid sub-pixel rendering
      if (y < A.y) {                // Start drawing from top of lister
        continue;
      }
      ImVec2 textSize = ImGui::CalcTextSize("test");
      float itemHeight = std::max(textSize.y, EditorTheme.rewardHeight);
      if (y + itemHeight >
          canvasPos.y + canvasSize.y) { // End drawing when exceeds bottom
        break;
      }

      auto reward = rewardsInUsed[i];

      int frameA = (keyFrames[reward->kf1Id])->frame;
      int frameB = (keyFrames[reward->kf2Id])->frame;
      int frameStart = (frameA < frameB) ? frameA : frameB;
      int frameEnd = (frameA < frameB) ? frameB : frameA;
      float xStart = B.x + frameStart * zoom[0] + pan[0];
      float xEnd = B.x + frameEnd * zoom[0] + pan[0];
      if (xEnd <= B.x || xStart >= canvasPos.x + canvasSize.x) {
        continue;
      }

      float btnXStart = (xStart < B.x) ? B.x : xStart;
      float btnXEnd = (xEnd > canvasPos.x + canvasSize.x)
                          ? canvasPos.x + canvasSize.x
                          : xEnd;

      ImVec4 color = EditorTheme.reward;

      // Mouse event
      ImGui::PushID(reward->getId());
      ImGui::SetCursorPos(ImVec2{btnXStart, y} - ImGui::GetWindowPos());
      ImGui::InvisibleButton(
          "##Reward", ImVec2{btnXEnd - btnXStart, EditorTheme.rewardHeight});
      ImGui::SetItemAllowOverlap();
      ImGui::PopID();

      if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
      }

      if (ImGui::IsItemClicked()) {
        selectedReward = reward->getId();
        initRewardDetails = true;
      }

      if (reward->getId() == selectedReward) { // Selected
        color = ImVec4{color.x * 1.2f, color.y * 1.2f, color.z * 1.2f, color.w};

        ImGui::OpenPopup("Reward Details");
        bool pOpen = true;
        if (ImGui::BeginPopupModal("Reward Details", &pOpen)) {
          if (initRewardDetails) {
            if (reward->name.size() < sizeof(nameBuffer)) {
              strcpy(nameBuffer, reward->name.c_str());
            } else {
              strncpy(nameBuffer, reward->name.c_str(), sizeof(nameBuffer) - 1);
              nameBuffer[sizeof(nameBuffer) - 1] = '\0';
            }

            if (reward->definition.size() < sizeof(definitionBuffer)) {
              strcpy(definitionBuffer, reward->definition.c_str());
            } else {
              strncpy(definitionBuffer, reward->definition.c_str(),
                      sizeof(definitionBuffer) - 1);
              definitionBuffer[sizeof(nameBuffer) - 1] = '\0';
            }

            initRewardDetails = false;
          }

          // Name
          ImGui::InputText("Name", nameBuffer, sizeof(nameBuffer));

          // Key frames information
          auto kf1 = keyFrames[reward->kf1Id];
          auto kf2 = keyFrames[reward->kf2Id];
          ImGui::Text("kf1 (First Key Frame): %d", kf1->frame);
          ImGui::SameLine();
          ImGui::Text("kf2 (Second Key Frame): %d", kf2->frame);

          // Definition
          ImVec2 availableSpace = ImGui::GetContentRegionAvail();
          availableSpace.x -= ImGui::CalcTextSize("Python Definition").x;
          availableSpace.y -= ImGui::GetFrameHeightWithSpacing();
          ImGui::InputTextMultiline(
              "Python Definition", definitionBuffer,
              sizeof(definitionBuffer) / 4, availableSpace,
              ImGuiInputTextFlags_AllowTabInput); // Divide by 4 so that
                                                  // interpreting tab as 4
                                                  // spaces won't exceed buffer
                                                  // limit

          // Save changes (tab will be interpreted as 4 spaces)
          if (ImGui::Button("Save Changes")) {
            reward->name = std::string(nameBuffer);
            reward->definition = std::string(definitionBuffer);
            size_t pos = 0;
            while ((pos = reward->definition.find("\t", pos)) !=
                   std::string::npos) {
              reward->definition.replace(pos, 1, "    ");
              pos += 4;
            }
            initRewardDetails = true; // Update buffer
          }

          // Delete Reward
          ImGui::SameLine();
          if (ImGui::Button("Delete Reward")) {
            ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
            selectedReward = -1;
            rewards[reward->getId()] = nullptr;
            auto it =
                std::find(rewardsInUsed.begin(), rewardsInUsed.end(), reward);
            rewardsInUsed.erase(it);
            break; // rewardsInUsed updated, stop iterating
          }
          ImGui::EndPopup();
        } else {
          selectedReward = -1; // Popup closed by clicking X
        }
      }

      ImVec2 rectMin(xStart, y);
      ImVec2 rectMax(xEnd, y + EditorTheme.rewardHeight);
      ImColor contourColor(0, 0, 0, 255);
      float contourThickness = 1.2f * contentScale;
      drawList->AddRectFilled(rectMin, rectMax, ImColor(color),
                              2.0f * contentScale);
      drawList->AddRect(rectMin, rectMax, contourColor, 2.0f * contentScale, 0,
                        contourThickness);
    }
  };

  auto ListerBackground = [&]() {
    drawList->AddRectFilled(
        A, A + ImVec2{ListerTheme.width, canvasSize.y - TimelineTheme.height},
        ImColor(ListerTheme.background));
  };

  auto Lister = [&]() {
    float x = A.x + 10.0f * contentScale;
    for (int i = 0; i < static_cast<int>(rewardsInUsed.size()); i++) {
      float y = std::round(A.y + 5.0f * contentScale + zoom[1] * i +
                           pan[1]); // Avoid sub-pixel rendering
      if (y < A.y) {                // Start drawing from top of lister
        continue;
      }
      ImVec2 textSize = ImGui::CalcTextSize("test");
      float itemHeight = std::max(textSize.y, EditorTheme.rewardHeight);
      if (y + itemHeight >
          canvasPos.y + canvasSize.y) { // End drawing when exceeds bottom
        break;
      }

      auto reward = rewardsInUsed[i];
      std::string name = reward->name;
      std::string visibleName;
      for (int i = name.size(); i > 0; i--) { // Find visible substring
        visibleName = name.substr(0, i);
        ImVec2 textSize = ImGui::CalcTextSize(visibleName.c_str());
        if (x + textSize.x <= C.x) {
          break;
        }
      }

      if (visibleName.size() > 0) {
        // Mouse event
        ImGui::PushID(reward->getId());
        ImGui::SetCursorPos(ImVec2{x, y} - ImGui::GetWindowPos());
        ImVec2 textSize = ImGui::CalcTextSize(visibleName.c_str());
        ImGui::InvisibleButton("##RewardName", textSize);
        ImGui::SetItemAllowOverlap();
        ImGui::PopID();

        if (ImGui::IsItemHovered()) {
          ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }

        if (ImGui::BeginDragDropSource()) {
          int sourceIndex = i;
          ImGui::SetDragDropPayload("DragReward", &sourceIndex, sizeof(int));
          ImGui::Text("%s", visibleName.c_str());
          ImGui::EndDragDropSource();
        }

        if (ImGui::BeginDragDropTarget()) {
          if (const ImGuiPayload *payload =
                  ImGui::AcceptDragDropPayload("DragReward")) {
            IM_ASSERT(payload->DataSize == sizeof(int));
            int sourceIndex = *(const int *)payload->Data;
            int targetIndex = i;
            auto item = rewardsInUsed[sourceIndex];
            rewardsInUsed.erase(rewardsInUsed.begin() + sourceIndex);
            rewardsInUsed.insert(rewardsInUsed.begin() + targetIndex, item);
            ImGui::EndDragDropTarget();
            break; // RewardsInUsed updated, stop iterating
          } else {
            ImGui::EndDragDropTarget();
          }
        }

        drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2{x, y},
                          ImColor(ListerTheme.text), visibleName.c_str());
      }
    }
  };

  auto TimelineBackground = [&]() {
    drawList->AddRectFilled(
        B, B + ImVec2{canvasSize.x - ListerTheme.width, TimelineTheme.height},
        ImColor(TimelineTheme.background));

    // Mouse event
    ImGui::SetCursorPos(ImVec2{B.x, B.y} - ImGui::GetWindowPos());
    ImGui::InvisibleButton(
        "##TimelineBackground",
        ImVec2{canvasSize.x - ListerTheme.width, TimelineTheme.height});
    ImGui::SetItemAllowOverlap();

    if (ImGui::IsItemHovered()) {
      ImGui::SetItemUsingMouseWheel();

      // Zooming
      float initialZoom = zoom[0];
      zoom[0] = ImClamp(zoom[0] + io.MouseWheel, horizZoomRange[0],
                        horizZoomRange[1]);

      // Change pan according to zoom
      if (zoom[0] != initialZoom) {
        float minHorizPan =
            -(totalFrame * zoom[0] - (canvasSize.x - ListerTheme.width));
        minHorizPan = std::min(minHorizPan, 0.0f);
        float panTemp = pan[0] / initialZoom * zoom[0];
        pan[0] = ImClamp(panTemp, minHorizPan, 0.0f);
      }
    }

    if (!addingReward && ImGui::IsItemActive()) {
      int frameTemp = static_cast<int>(
          std::round((io.MousePos.x - B.x - pan[0]) / zoom[0]));
      currentFrame = ImClamp(frameTemp, 0, totalFrame - 1);
    }
  };

  auto Timeline = [&]() {
    float interspace = zoom[0] * stride; // Distance between each base frame
    int frameStart =
        ceil(-pan[0] / interspace) * stride; // Only draw in visible area
    frameStart = ImClamp(frameStart, 0, totalFrame - 1);
    float yMin = B.y;
    float yMax = B.y + TimelineTheme.height;

    for (int frame = frameStart; frame <= totalFrame - 1; frame += stride) {
      int i = frame / stride; // ith line

      float x = B.x + i * interspace + pan[0];
      if (x > canvasPos.x + canvasSize.x) { // Only draw in visible area
        break;
      }

      drawList->AddLine(ImVec2(x, yMin), ImVec2(x, yMax),
                        ImColor(TimelineTheme.dark), 1.0f * contentScale);

      ImVec2 textSize =
          ImGui::CalcTextSize(std::to_string(frame).c_str()) * 0.85f;
      if (x + 5.0f * contentScale + textSize.x <=
          canvasPos.x + canvasSize.x) { // Only draw in visible area
        drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize() * 0.85f,
                          ImVec2{x + 5.0f * contentScale, yMin},
                          ImColor(TimelineTheme.text),
                          std::to_string(frame).c_str());
      }
    }
  };

  auto KeyFrameIndicator = [&](std::shared_ptr<KeyFrame> kf, ImVec4 color) {
    float x = B.x + kf->frame * zoom[0] + pan[0];
    if (x < B.x ||
        x > canvasPos.x + canvasSize.x) { // Only draw in visible area
      return;
    }

    float size = TimelineTheme.keyFrameIndicatorSize;
    ImVec2 center = ImVec2{x, C.y - size};

    // Calculate the diamond's corner points
    ImVec2 top(center.x, center.y - size / 2);
    ImVec2 right(center.x + size / 2, center.y);
    ImVec2 bottom(center.x, center.y + size / 2);
    ImVec2 left(center.x - size / 2, center.y);

    // Mouse event
    ImGui::PushID(kf->getId());
    ImGui::SetCursorPos(ImVec2{left.x, top.y} - ImGui::GetWindowPos());
    ImGui::InvisibleButton("##KeyFrame", ImVec2{size, size});
    ImGui::SetItemAllowOverlap();
    ImGui::PopID();

    if (ImGui::IsItemHovered()) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }

    if (addingReward) { // Select first and second key frames for new reward
      if (keyFramesForNewReward.size() == 0) {
        if (ImGui::IsItemClicked()) {
          keyFramesForNewReward.push_back(kf);
        }
      } else if (keyFramesForNewReward.size() == 1) {
        if (ImGui::IsItemClicked()) {
          auto it = std::find(keyFramesForNewReward.begin(),
                              keyFramesForNewReward.end(), kf);
          if (it != keyFramesForNewReward.end()) { // Deselect key frame
            keyFramesForNewReward.erase(it);
          } else {
            // Add reward
            int rewardId = rewardIdGenerator.next();
            std::string name = "Reward " + std::to_string(rewardId);
            int kf1Id = (keyFramesForNewReward[0])->getId();
            int kf2Id = kf->getId();
            auto reward = std::make_shared<Reward>(rewardId, kf1Id, kf2Id, name,
                                                   defaultRewardDefinition);
            rewards.push_back(reward);
            rewardsInUsed.push_back(reward);

            // Exit reward adding mode
            keyFramesForNewReward.clear();
            addingReward = false;
          }
        }
      }
    } else { // Drag key frame
      if (ImGui::IsItemActive()) {
        color = ImVec4{color.x * 1.2f, color.y * 1.2f, color.z * 1.2f, color.w};

        int frameTemp = static_cast<int>(
            std::round((io.MousePos.x - B.x - pan[0]) / zoom[0]));
        kf->frame = ImClamp(frameTemp, 0, totalFrame - 1);
        currentFrame = kf->frame;
      }

      if (ImGui::IsItemDeactivated()) {
        auto it = std::find_if(
            keyFramesInUsed.begin(), keyFramesInUsed.end(),
            [&](auto &kf2) { return kf->frame == kf2->frame && kf != kf2; });

        if (it != keyFramesInUsed.end()) { // Has duplicate
          keyFrameToModify = (*it)->getId();

          // Delete all rewards depending on the key frame
          for (auto it = rewardsInUsed.begin(); it != rewardsInUsed.end();) {
            auto reward = *it;
            if (reward->kf1Id == keyFrameToModify ||
                reward->kf2Id == keyFrameToModify) {
              rewards[reward->getId()] = nullptr;
              it = rewardsInUsed.erase(it);
            } else {
              it++;
            }
          }

          // Delete key frame
          if (mDeleteKeyFrameCallback) {
            mDeleteKeyFrameCallback(
                std::static_pointer_cast<KeyFrameEditor>(shared_from_this()));
          }
          keyFrames[keyFrameToModify] = nullptr;
          keyFramesInUsed.erase(it);
          keyFramesInUsedUpdated = true;
        }

        std::sort(keyFramesInUsed.begin(), keyFramesInUsed.end(),
                  [](auto &a, auto &b) { return a->frame < b->frame; });
      }
    }

    // Color and thickness for contour
    ImColor contourColor(0, 0, 0, 255);
    float contourThickness = 1.0f * contentScale;

    drawList->AddQuadFilled(top, right, bottom, left, ImColor(color));
    drawList->AddQuad(top, right, bottom, left, contourColor, contourThickness);
  };

  auto CurrentFrameIndicator = [&](int frame, ImVec4 color) {
    float x = B.x + frame * zoom[0] + pan[0];
    if (x < B.x ||
        x > canvasPos.x + canvasSize.x) { // Only draw in visible area
      return;
    }

    float innerSize = TimelineTheme.keyFrameIndicatorSize;
    float outerSize = TimelineTheme.currentFrameIndicatorSize;
    ImVec2 center = ImVec2{x, C.y - innerSize};

    // Calculate the inner diamond's corner points
    ImVec2 innerTop(center.x, center.y - innerSize / 2);
    ImVec2 innerRight(center.x + innerSize / 2, center.y);
    ImVec2 innerBottom(center.x, center.y + innerSize / 2);
    ImVec2 innerLeft(center.x - innerSize / 2, center.y);

    // Calculate the outer diamond's corner points
    ImVec2 outerTop(center.x, center.y - outerSize / 2);
    ImVec2 outerRight(center.x + outerSize / 2, center.y);
    ImVec2 outerBottom(center.x, center.y + outerSize / 2);
    ImVec2 outerLeft(center.x - outerSize / 2, center.y);

    // Color and thickness for contour
    ImColor contourColor(0, 0, 0, 255);
    float contourThickness = 1.0f * contentScale;

    drawList->AddQuadFilled(outerTop, outerRight, outerBottom, outerLeft,
                            ImColor(color));
    drawList->AddQuad(outerTop, outerRight, outerBottom, outerLeft,
                      contourColor, contourThickness);
    drawList->AddQuad(innerTop, innerRight, innerBottom, innerLeft,
                      contourColor, contourThickness);

    if (canvasSize.y - TimelineTheme.height > 0) { // Editor area exists
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

  auto HorizScrollbar = [&]() {
    float barPadding = EditorTheme.scrollbarPadding;
    float areaWidth = canvasSize.x - ListerTheme.width - 2 * barPadding;
    float height = EditorTheme.scrollbarSize;
    float heightMin = canvasPos.y + canvasSize.y - height - 4.0f * contentScale;

    if (areaWidth > 0 && heightMin > C.y) {
      float offsetRatio = -pan[0] / (zoom[0] * totalFrame);
      float offsetWidth = areaWidth * offsetRatio;
      float barWidthRatio =
          (canvasSize.x - ListerTheme.width) / (totalFrame * zoom[0]);
      float rawBarWidth = areaWidth * barWidthRatio;
      float barWidth = (rawBarWidth < barPadding - 1.0f * contentScale)
                           ? barPadding - 1.0f * contentScale
                           : rawBarWidth; // Ensure grabbable

      ImVec2 barMin = ImVec2{C.x + barPadding + offsetWidth, heightMin};
      ImVec2 barMax =
          ImVec2{C.x + barPadding + offsetWidth + barWidth, heightMin + height};
      ImVec4 color = EditorTheme.scrollbar;

      ImGui::SetCursorPos(barMin - ImGui::GetWindowPos());
      ImGui::InvisibleButton("##HorizScrollbar", ImVec2{barWidth, height});
      ImGui::SetItemAllowOverlap();

      if (ImGui::IsItemActivated()) {
        initialPan[0] = pan[0];
      }

      if (ImGui::IsItemActive()) {
        color = ImVec4{color.x * 1.2f, color.y * 1.2f, color.z * 1.2f, color.w};

        float minPan =
            -(totalFrame * zoom[0] - (canvasSize.x - ListerTheme.width));
        minPan = std::min(minPan, 0.0f);
        float deltaPan =
            (ImGui::GetMouseDragDelta().x / (areaWidth - rawBarWidth)) * minPan;
        float panTemp = initialPan[0] + deltaPan;
        pan[0] = ImClamp(panTemp, minPan, 0.0f);
      }

      drawList->AddRectFilled(barMin, barMax, ImColor(color),
                              8.0f * contentScale);
    }
  };

  auto VertScrollbar = [&]() {
    float barPadding = EditorTheme.scrollbarPadding;
    float areaHeight = canvasSize.y - TimelineTheme.height - 2 * barPadding;
    float width = EditorTheme.scrollbarSize;
    float widthMin = canvasPos.x + canvasSize.x - width - 8.0f * contentScale;

    if (areaHeight > 0 && widthMin > C.x) {
      float offsetRatio =
          -pan[1] / (zoom[1] * static_cast<int>(rewardsInUsed.size()));
      float offsetHeight = areaHeight * offsetRatio;
      float barHeightRatio = (canvasSize.y - TimelineTheme.height) /
                             (static_cast<int>(rewardsInUsed.size()) * zoom[1] +
                              10.0f * contentScale);
      float rawBarHeight = areaHeight * barHeightRatio;
      float barHeight = (rawBarHeight < barPadding - 1.0f * contentScale)
                            ? barPadding - 1.0f * contentScale
                            : rawBarHeight; // Ensure grabbable

      ImVec2 barMin = ImVec2{widthMin, C.y + barPadding + offsetHeight};
      ImVec2 barMax =
          ImVec2{widthMin + width, C.y + barPadding + offsetHeight + barHeight};
      ImVec4 color = EditorTheme.scrollbar;

      ImGui::SetCursorPos(barMin - ImGui::GetWindowPos());
      ImGui::InvisibleButton("##VertScrollbar", ImVec2{width, barHeight});
      ImGui::SetItemAllowOverlap();

      if (ImGui::IsItemActivated()) {
        initialPan[1] = pan[1];
      }

      if (ImGui::IsItemActive()) {
        color = ImVec4{color.x * 1.2f, color.y * 1.2f, color.z * 1.2f, color.w};

        float minPan =
            -(static_cast<int>(rewardsInUsed.size()) * zoom[1] +
              5.0f * contentScale - (canvasSize.y - TimelineTheme.height));
        minPan = std::min(minPan, 0.0f);
        float deltaPan =
            (ImGui::GetMouseDragDelta().y / (areaHeight - rawBarHeight)) *
            minPan;
        float panTemp = initialPan[1] + deltaPan;
        pan[1] = ImClamp(panTemp, minPan, 0.0f);
      }

      drawList->AddRectFilled(barMin, barMax, ImColor(color),
                              8.0f * contentScale);
    }
  };

  auto ListerWidthHandle = [&]() {
    float buttonLeftX =
        std::max(B.x - ListerTheme.handleWidth / 2, canvasPos.x);
    float buttonRightX =
        std::min(B.x + ListerTheme.handleWidth / 2, canvasPos.x + canvasSize.x);
    float adaptedHandleWidth = buttonRightX - buttonLeftX;
    if (adaptedHandleWidth > 0) {
      ImGui::SetCursorPos(ImVec2{buttonLeftX, A.y} - ImGui::GetWindowPos());
      ImGui::InvisibleButton(
          "##ListerWidthHandle",
          ImVec2{adaptedHandleWidth, canvasSize.y - TimelineTheme.height});
      ImGui::SetItemAllowOverlap();

      if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
      }

      if (ImGui::IsItemActivated()) {
        listerInitialWidth = ListerTheme.width;
      }

      if (ImGui::IsItemActive()) {
        float deltaWidth = ImGui::GetMouseDragDelta().x;
        ListerTheme.width =
            ImClamp(listerInitialWidth + deltaWidth, 0.0f, canvasSize.x);
      }
    }
  };

  auto MiddleButtonPanning = [&]() {
    // Horizontal
    float minHorizPan =
        -(totalFrame * zoom[0] - (canvasSize.x - ListerTheme.width));
    minHorizPan = std::min(minHorizPan, 0.0f);
    float deltaHorizPan = io.MouseDelta.x;
    float horizPanTemp = pan[0] + deltaHorizPan;
    pan[0] = ImClamp(horizPanTemp, minHorizPan, 0.0f);

    // Vertical
    float minVertPan =
        -(static_cast<int>(rewardsInUsed.size()) * zoom[1] +
          5.0f * contentScale - (canvasSize.y - TimelineTheme.height));
    minVertPan = std::min(minVertPan, 0.0f);
    float deltaVertPan = io.MouseDelta.y;
    float vertPanTemp = pan[1] + deltaVertPan;
    pan[1] = ImClamp(vertPanTemp, minVertPan, 0.0f);
  };

  ImGui::BeginGroup();

  // Cross elements
  if (ListerTheme.width > 0) {
    CrossBackground();
  }

  // Editor elements
  if (canvasSize.y - TimelineTheme.height > 0 &&
      canvasSize.x - ListerTheme.width > 0) {
    EditorBackground();
    VerticalGrid();
    Editor();
  }

  // Lister elements
  if (ListerTheme.width > 0 && canvasSize.y - TimelineTheme.height > 0) {
    ListerBackground();
    Lister();
  }

  // Timeline elements
  if (canvasSize.x - ListerTheme.width > 0) {
    TimelineBackground();
    Timeline();

    if (!addingReward) {
      auto it =
          std::find_if(keyFramesInUsed.begin(), keyFramesInUsed.end(),
                       [&](auto &kf) { return kf->frame == currentFrame; });
      ImVec4 color = (it == keyFramesInUsed.end()) ? TimelineTheme.currentFrame
                                                   : TimelineTheme.keyFrame;
      CurrentFrameIndicator(currentFrame, color);
    }

    for (auto &kf : keyFramesInUsed) { // Ascending order
      if (addingReward) {
        auto it = std::find(keyFramesForNewReward.begin(),
                            keyFramesForNewReward.end(), kf);
        ImVec4 color = (it == keyFramesForNewReward.end())
                           ? TimelineTheme.keyFrame
                           : TimelineTheme.selectedKeyFrame;
        KeyFrameIndicator(kf, color);
      } else {
        KeyFrameIndicator(kf, TimelineTheme.keyFrame);
        if (keyFramesInUsedUpdated) {
          keyFramesInUsedUpdated = false;
          break;
        }
      }
    }
  }

  // Horizontal scrollbar
  if (totalFrame * zoom[0] > canvasSize.x - ListerTheme.width) {
    HorizScrollbar();
  }

  // Vertical scrollbar
  if (static_cast<int>(rewardsInUsed.size()) * zoom[1] + 10.0f * contentScale >
      canvasSize.y - TimelineTheme.height) {
    VertScrollbar();
  }

  // Lister width handle
  ListerWidthHandle();

  // Middle button panning
  if (ImGui::IsWindowHovered() && io.MouseDown[2]) {
    MiddleButtonPanning();
  }

  ImGui::EndGroup();
}

std::vector<KeyFrame *> KeyFrameEditor::getKeyFramesInUsed() const {
  std::vector<KeyFrame *> output;
  for (auto &kf : keyFramesInUsed) {
    output.push_back(kf.get());
  }
  return output;
}

std::vector<Reward *> KeyFrameEditor::getRewardsInUsed() const {
  std::vector<Reward *> output;
  for (auto &reward : rewardsInUsed) {
    output.push_back(reward.get());
  }
  return output;
}

void KeyFrameEditor::clear() {
  keyFrames.clear();
  keyFramesInUsed.clear();
  rewards.clear();
  rewardsInUsed.clear();
}

void KeyFrameEditor::addKeyFrame(int id, int frame) {
  auto kf = std::make_shared<KeyFrame>(id, frame);
  if (id >= static_cast<int>(keyFrames.size())) {
    keyFrames.resize(id + 1, nullptr);
  }
  keyFrames[id] = kf;
  keyFramesInUsed.push_back(kf);
  std::sort(keyFramesInUsed.begin(), keyFramesInUsed.end(),
            [](auto &a, auto &b) { return a->frame < b->frame; });
}

void KeyFrameEditor::addReward(int id, int kf1Id, int kf2Id, std::string name,
                               std::string definition) {
  auto reward = std::make_shared<Reward>(id, kf1Id, kf2Id, name, definition);
  if (id >= static_cast<int>(rewards.size())) {
    rewards.resize(id + 1, nullptr);
  }
  rewards[id] = reward;
  rewardsInUsed.push_back(reward);
}

} // namespace ui
} // namespace svulkan2
