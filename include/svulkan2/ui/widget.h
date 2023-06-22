#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace svulkan2 {
namespace ui {

#define UI_DECLARE_APPEND(CLASS)                                                                  \
public:                                                                                           \
  inline std::shared_ptr<CLASS> append(std::shared_ptr<Widget> child) {                           \
    mChildren.push_back(child);                                                                   \
    child->setParent(shared_from_this());                                                         \
    return std::static_pointer_cast<CLASS>(shared_from_this());                                   \
  }

#define UI_ATTRIBUTE(CLASS, TYPE, NAME)                                                           \
protected:                                                                                        \
  TYPE m##NAME{};                                                                                 \
                                                                                                  \
public:                                                                                           \
  inline std::shared_ptr<CLASS> NAME(TYPE value) {                                                \
    m##NAME = value;                                                                              \
    return std::static_pointer_cast<CLASS>(shared_from_this());                                   \
  }                                                                                               \
  inline TYPE get##NAME() { return m##NAME; }

#define UI_BINDING(CLASS, TYPE, NAME)                                                             \
protected:                                                                                        \
  std::function<void(TYPE)> m##NAME##Setter;                                                      \
  std::function<TYPE()> m##NAME##Getter;                                                          \
                                                                                                  \
public:                                                                                           \
  inline std::shared_ptr<CLASS> Bind##NAME(std::function<TYPE()> getter,                          \
                                           std::function<void(TYPE)> setter) {                    \
    m##NAME##Getter = getter;                                                                     \
    m##NAME##Setter = setter;                                                                     \
    return std::static_pointer_cast<CLASS>(shared_from_this());                                   \
  }

#define UI_BINDING_READONLY(CLASS, TYPE, NAME)                                                    \
protected:                                                                                        \
  std::function<TYPE()> m##NAME##Getter;                                                          \
                                                                                                  \
public:                                                                                           \
  inline std::shared_ptr<CLASS> Bind##NAME(std::function<TYPE()> getter) {                        \
    m##NAME##Getter = getter;                                                                     \
    return std::static_pointer_cast<CLASS>(shared_from_this());                                   \
  }

#define UI_BINDING_WRITEONLY(CLASS, TYPE, NAME)                                                   \
protected:                                                                                        \
  std::function<void(TYPE)> m##NAME##Setter;                                                      \
                                                                                                  \
public:                                                                                           \
  inline std::shared_ptr<CLASS> Bind##NAME(std::function<void(TYPE)> setter) {                    \
    m##NAME##Setter = setter;                                                                     \
    return std::static_pointer_cast<CLASS>(shared_from_this());                                   \
  }

#define UI_CLASS(CLASS) class CLASS : public Widget

#define UI_DECLARE_LABEL(CLASS)                                                                   \
protected:                                                                                        \
  std::string mLabel{};                                                                           \
  std::string mId{};                                                                              \
                                                                                                  \
public:                                                                                           \
  inline std::shared_ptr<CLASS> Label(std::string value) {                                        \
    mLabel = value;                                                                               \
    return std::static_pointer_cast<CLASS>(shared_from_this());                                   \
  }                                                                                               \
  inline std::string getLabel() { return mLabel; }                                                \
  inline std::shared_ptr<CLASS> Id(std::string value) {                                           \
    mId = value;                                                                                  \
    return std::static_pointer_cast<CLASS>(shared_from_this());                                   \
  }                                                                                               \
  inline std::string getId() { return mId; }                                                      \
  inline std::string getLabelId() { return mLabel + "##" + mId; }

class Widget : public std::enable_shared_from_this<Widget> {

public:
  template <typename T> static std::shared_ptr<T> Create() {
    static_assert(std::is_convertible<T *, Widget *>(), "Only widgets can be created.");
    return std::make_shared<T>();
  }

  void setParent(std::weak_ptr<Widget> parent);

  void remove();
  void removeChildren();
  inline std::vector<std::shared_ptr<Widget>> getChildren() const { return mChildren; };

  /** build imgui */
  virtual void build() = 0;

  virtual ~Widget() = default;

protected:
  std::weak_ptr<Widget> mParent;
  std::vector<std::shared_ptr<Widget>> mChildren;
};

} // namespace ui
} // namespace svulkan2
