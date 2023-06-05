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

  // Serialization
  inline int getState() const { return id; }
  inline void setState(const int &state) { id = state; }

private:
  int id;
};

class KeyFrame {
private:
  int id;

public:
  int frame;

  KeyFrame(int id, int frame) : id(id), frame(frame) {}
  int getId() const { return id; };
};

class Reward {
private:
  int id;

public:
  int kf1Id;
  int kf2Id;
  std::string name;
  std::string definition;

  Reward(int id, int kf1Id, int kf2Id, std::string name, std::string definition)
      : id(id), kf1Id(kf1Id), kf2Id(kf2Id), name(name), definition(definition) {
  }
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
  UI_ATTRIBUTE(KeyFrameEditor,
               std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               ExportCallback);
  UI_ATTRIBUTE(KeyFrameEditor,
               std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               ImportCallback);

public:
  KeyFrameEditor(float contentScale_);
  void build() override;
  int getCurrentFrame() const { return currentFrame; };
  int getKeyFrameToModify() const { return keyFrameToModify; };
  IdGenerator *getKeyFrameIdGenerator() { return &keyFrameIdGenerator; };
  IdGenerator *getRewardIdGenerator() { return &rewardIdGenerator; };
  std::vector<KeyFrame *> getKeyFramesInUsed() const;
  std::vector<Reward *> getRewardsInUsed() const;

  // Import helper
  void setKeyFrameIdGeneratorState(int id) {
    keyFrameIdGenerator.setState(id);
  };
  void setRewardIdGeneratorState(int id) { rewardIdGenerator.setState(id); };
  void clear(); // Clear key frames and rewards
  void addKeyFrame(int id, int frame);
  void addReward(int id, int kf1Id, int kf2Id, std::string name,
                 std::string definition);

private:
  // Control panel
  bool addingReward;
  std::vector<std::shared_ptr<KeyFrame>> keyFramesForNewReward;

  // Timeline
  int currentFrame;
  int totalFrame;
  int totalFrameRange[2]{32, 2048};
  int stride;

  // Key frame
  IdGenerator keyFrameIdGenerator;
  std::vector<std::shared_ptr<KeyFrame>> keyFrames;
  std::vector<std::shared_ptr<KeyFrame>> keyFramesInUsed;
  int keyFrameToModify;        // Id of key frame that need to be modified
  bool keyFramesInUsedUpdated; // Use this to break the iteration through
                               // keyFramesInUsed when update occurs

  // Reward
  IdGenerator rewardIdGenerator;
  std::vector<std::shared_ptr<Reward>> rewards;
  std::vector<std::shared_ptr<Reward>> rewardsInUsed;
  int selectedReward; // Id of reward that is being selected
  bool initRewardDetails;
  char nameBuffer[256];
  char definitionBuffer[65536];
  std::string defaultRewardDefinition;

  // Visual
  float contentScale;

  float pan[2]; // Deviation of {timeline, lister} in pixels
  float initialPan[2];

  float zoom[2]; // Distance between each {frame, reward} in pixels
  float horizZoomRange[2];

  float listerInitialWidth;

  // Theme
  struct CrossTheme_ {
    float borderWidth{1.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.114f)};
    ImVec4 border{ImColor::HSV(0.0f, 0.0f, 0.2f)};
  } CrossTheme;

  struct ListerTheme_ {
    float width{100.0f};
    float handleWidth{15.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.114f)};
    ImVec4 text{ImColor::HSV(0.0f, 0.0f, 0.7f)};
  } ListerTheme;

  struct TimelineTheme_ {
    float height{40.0f};
    float keyFrameIndicatorSize{12.0f};
    float currentFrameIndicatorSize{22.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.114f)};
    ImVec4 text{ImColor::HSV(0.0f, 0.0f, 0.7f)};
    ImVec4 dark{ImColor::HSV(0.0f, 0.0f, 0.075f)};
    ImVec4 currentFrame{ImColor::HSV(0.6f, 0.6f, 0.95f)};
    ImVec4 keyFrame{ImColor::HSV(0.12f, 0.8f, 0.95f)};
    ImVec4 selectedKeyFrame{ImColor::HSV(0.75f, 0.25f, 0.95f)};
  } TimelineTheme;

  struct EditorTheme_ {
    float scrollbarPadding{20.0f};
    float scrollbarSize{10.0f};
    float rewardHeight{15.0f};

    ImVec4 background{ImColor::HSV(0.0f, 0.0f, 0.188f)};
    ImVec4 dark{ImColor::HSV(0.0f, 0.0f, 0.075f)};
    ImVec4 mid{ImColor::HSV(0.0f, 0.0f, 0.15f)};
    ImVec4 scrollbar{ImColor::HSV(0.0f, 0.0f, 0.33f)};
    ImVec4 reward{ImColor::HSV(0.12f, 0.8f, 0.9f)};
  } EditorTheme;
};

} // namespace ui
} // namespace svulkan2
