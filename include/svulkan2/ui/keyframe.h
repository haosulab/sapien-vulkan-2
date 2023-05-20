#pragma once
#include "widget.h"
#include <functional>
#include <imgui_internal.h>
#include <vector>

namespace svulkan2 {
namespace ui {

class IdGenerator {
public:
  inline int next() { return id++; }
  inline IdGenerator() : id(0) {}

private:
  int id;
};

class KeyFrame {
private:
  int id;

public:
  int frame;
  std::vector<int> itemIds; // Items depending on this key frame

  KeyFrame(int id, int frame) : id(id), frame(frame) {}
  int getId() const { return id; };
};

class Item {
private:
  int id;

public:
  int kfaId;
  int kfbId;
  std::string name;
  std::string content;

  Item(int id, int kfaId, int kfbId, std::string name, std::string content)
      : id(id), kfaId(kfaId), kfbId(kfbId), name(name), content(content) {}
  int getId() const { return id; };
};

UI_CLASS(KeyFrameEditor) {
  UI_ATTRIBUTE(KeyFrameEditor,
               std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               InsertKeyFrameCallback);
  UI_ATTRIBUTE(KeyFrameEditor,
               std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               LoadKeyFrameCallback);
  UI_ATTRIBUTE(KeyFrameEditor,
               std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               UpdateKeyFrameCallback);
  UI_ATTRIBUTE(KeyFrameEditor,
               std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               DeleteKeyFrameCallback);

public:
  KeyFrameEditor(float contentScale_);
  void build() override;
  int getCurrentFrame() const { return currentFrame; };
  std::vector<KeyFrame *> getKeyFramesInUsed() const;
  int getKeyFrameToModify() const {
    return keyFrameToModify;
  }; // Notify middleware

private:
  // Timeline
  int currentFrame{0};
  int frameRange[2]{0, 128};
  int stride{8};
  int minIntervals{16}; // maxStride = frameRange / minIntervals
  int selectedMaxFrame{0};
  int prevSelectedMaxFrame{0};

  // Key frame container
  IdGenerator keyFrameIdGenerator;
  std::vector<std::shared_ptr<KeyFrame>> keyFrames;
  std::vector<std::shared_ptr<KeyFrame>> keyFramesInUsed;
  int keyFrameToModify; // Id of key frame that need to be modified

  // Item container
  IdGenerator itemIdGenerator;
  std::vector<std::shared_ptr<Item>> items;
  std::vector<std::shared_ptr<Item>> itemsInUsed;

  // Visual
  float contentScale;

  float pan[2]{0.0f, 0.0f}; // Deviation of {timeline, lister} in pixels
  float initialPan[2];

  float zoom[2]; // Distance between each {frame, item} in pixels
  float horizZoomRange[2];
  bool resetHorizZoom{true};

  // Theme
  struct CrossTheme_ {
    float borderWidth{1.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.114f)};
    ImVec4 border{ImColor::HSV(0.0f, 0.0f, 0.2f)};
  } CrossTheme;

  struct ListerTheme_ {
    float width{100.0f};
    float handleWidth{15.0f};
    float initialWidth;

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.114f)};
    ImVec4 text{ImColor::HSV(0.0f, 0.0f, 0.7f)};
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
    float scrollbarPadding{20.0f};
    float scrollbarSize{10.0f};
    float itemSize{15.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.188f)};
    ImVec4 dark{ImColor::HSV(0.0f, 0.0f, 0.075f)};
    ImVec4 mid{ImColor::HSV(0.0f, 0.0f, 0.15f)};
    ImVec4 scrollbar{ImColor::HSV(0.0f, 0.0f, 0.33f)};
    ImVec4 item{ImColor::HSV(0.12f, 0.8f, 0.95f)};
  } EditorTheme;
};

} // namespace ui
} // namespace svulkan2
