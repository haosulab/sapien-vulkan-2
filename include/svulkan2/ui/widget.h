#pragma once
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

namespace svulkan2 {
namespace ui {

#define UI_CLASS(CLASS) class CLASS : public Widget

#define UI_DECLARE_APPEND(CLASS)                                               \
public:                                                                        \
  inline std::shared_ptr<CLASS> append(std::shared_ptr<Widget> child) {        \
    mChildren.push_back(child);                                                \
    child->setParent(shared_from_this());                                      \
    return std::static_pointer_cast<CLASS>(shared_from_this());                \
  }

#define UI_ATTRIBUTE(CLASS, TYPE, NAME)                                        \
private:                                                                       \
  TYPE m##NAME{};                                                              \
                                                                               \
public:                                                                        \
  inline std::shared_ptr<CLASS> NAME(TYPE value) {                             \
    m##NAME = value;                                                           \
    return std::static_pointer_cast<CLASS>(shared_from_this());                \
  }\
  inline TYPE get##NAME() { return m##NAME; }

class Widget : public std::enable_shared_from_this<Widget> {
protected:
  std::weak_ptr<Widget> mParent;
  std::vector<std::shared_ptr<Widget>> mChildren;

public:
  template <typename T> static std::shared_ptr<T> Create() {
    static_assert(std::is_convertible<T *, Widget *>(),
                  "Only widgets can be created.");
    return std::make_shared<T>();
  }

  void setParent(std::weak_ptr<Widget> parent);

  void remove();
  void removeChildren();
  inline std::vector<std::shared_ptr<Widget>> getChildren() const {
    return mChildren;
  };

  /** build imgui */
  virtual void build() = 0;

  virtual ~Widget() = default;
};

} // namespace ui
} // namespace svulkan2
