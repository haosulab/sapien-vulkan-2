#pragma once
#include "widget.h"
#include <functional>
#include <vector>
#include <array>

namespace svulkan2 {
namespace ui {

class Keyframe : public std::enable_shared_from_this<Keyframe> {
public:
  virtual int frame() = 0;
  virtual ~Keyframe() = default;
};

class Duration : public std::enable_shared_from_this<Duration> {
public:
  virtual std::shared_ptr<Keyframe> keyframe0() = 0;
  virtual std::shared_ptr<Keyframe> keyframe1() = 0;
  virtual std::string name() = 0;
  virtual ~Duration() = default;
};

UI_CLASS(KeyframeEditor) {
  UI_DECLARE_APPEND(KeyframeEditor);

  // indicate UI wants to add a key frame
  UI_ATTRIBUTE(KeyframeEditor,
               std::function<void(std::shared_ptr<Keyframe>, std::shared_ptr<Keyframe>)>,
               AddDurationCallback);

  // indicate UI wants to add a duration
  UI_ATTRIBUTE(KeyframeEditor, std::function<void(int)>, AddKeyframeCallback);
  // indicate UI wants to move a keyframe
  UI_ATTRIBUTE(KeyframeEditor, std::function<void(std::shared_ptr<Keyframe>, int)>,
               MoveKeyframeCallback);

  // called when double clicked on a keyframe
  UI_ATTRIBUTE(KeyframeEditor, std::function<void(std::shared_ptr<Keyframe>)>,
               DoubleClickKeyframeCallback);

  // called when clicked on a duration bar
  UI_ATTRIBUTE(KeyframeEditor, std::function<void(std::shared_ptr<Duration>)>,
               DoubleClickDurationCallback)

  UI_BINDING(KeyframeEditor, int, CurrentFrame);
  UI_BINDING(KeyframeEditor, int, TotalFrames);

public:
  virtual void addKeyframe(std::shared_ptr<Keyframe> frame) = 0;
  virtual void removeKeyframe(std::shared_ptr<Keyframe> frame) = 0;

  virtual void addDuration(std::shared_ptr<Duration> duration) = 0;
  virtual void removeDuration(std::shared_ptr<Duration> duration) = 0;

  virtual void setState(std::vector<std::shared_ptr<Keyframe>>,
                        std::vector<std::shared_ptr<Duration>>) = 0;
  virtual std::vector<std::shared_ptr<Keyframe>> getKeyframes() = 0;
  virtual std::vector<std::shared_ptr<Duration>> getDurations() = 0;

  static std::shared_ptr<KeyframeEditor> Create(float contentScale);
};

} // namespace ui
} // namespace svulkan2
