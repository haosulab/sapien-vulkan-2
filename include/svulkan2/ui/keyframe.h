#pragma once
#include "widget.h"
#include <functional>
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

class Duration {
private:
  int id;

public:
  int kf1Id;
  int kf2Id;
  std::string name;
  std::string definition;

  Duration(int id, int kf1Id, int kf2Id, std::string name, std::string definition)
      : id(id), kf1Id(kf1Id), kf2Id(kf2Id), name(name), definition(definition) {}
  int getId() const { return id; };
};

UI_CLASS(KeyFrameEditor) {
  UI_ATTRIBUTE(KeyFrameEditor, std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               InsertKeyFrameCallback);
  UI_ATTRIBUTE(KeyFrameEditor, std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               LoadKeyFrameCallback);
  UI_ATTRIBUTE(KeyFrameEditor, std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               UpdateKeyFrameCallback);
  UI_ATTRIBUTE(KeyFrameEditor, std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               DeleteKeyFrameCallback);
  UI_ATTRIBUTE(KeyFrameEditor, std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               ExportCallback);
  UI_ATTRIBUTE(KeyFrameEditor, std::function<void(std::shared_ptr<KeyFrameEditor>)>,
               ImportCallback);

public:
  KeyFrameEditor(float contentScale_);
  void build() override;
  int getCurrentFrame() const { return currentFrame; };
  int getKeyFrameToModify() const { return keyFrameToModify; };
  IdGenerator *getKeyFrameIdGenerator() { return &keyFrameIdGenerator; };
  IdGenerator *getDurationIdGenerator() { return &durationIdGenerator; };
  std::vector<KeyFrame *> getKeyFramesInUsed() const;
  std::vector<Duration *> getDurationsInUsed() const;

  // Import helper
  void setKeyFrameIdGeneratorState(int id) { keyFrameIdGenerator.setState(id); };
  void setDurationIdGeneratorState(int id) { durationIdGenerator.setState(id); };
  void clear(); // Clear key frames and durations
  void addKeyFrame(int id, int frame);
  void addDuration(int id, int kf1Id, int kf2Id, std::string name, std::string definition);
  int getTotalFrame() { return totalFrame; };
  void setTotalFrame(int totalFrame_) { totalFrame = totalFrame_; };

private:
  // Control panel
  bool addingDuration;
  std::vector<std::shared_ptr<KeyFrame>> keyFramesForNewDuration;

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

  // Duration
  IdGenerator durationIdGenerator;
  std::vector<std::shared_ptr<Duration>> durations;
  std::vector<std::shared_ptr<Duration>> durationsInUsed;
  int selectedDuration; // Id of duration that is being selected
  bool initDurationDetails;
  char nameBuffer[256];
  char definitionBuffer[65536];
  std::string defaultDurationDefinition;

  // Visual
  float contentScale;

  float pan[2]; // Deviation of {timeline, lister} in pixels
  float initialPan[2];

  float zoom[2]; // Distance between each {frame, duration} in pixels
  float horizZoomRange[2];

  float listerInitialWidth;
};

} // namespace ui
} // namespace svulkan2
