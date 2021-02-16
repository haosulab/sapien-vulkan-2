#pragma once
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

namespace svulkan2 {
namespace ui {

class Widget : public std::enable_shared_from_this<Widget> {
protected:
  std::string mName;
  std::weak_ptr<Widget> mParent;
  std::vector<std::shared_ptr<Widget>> mChildren;

public:
  template <typename T> static std::shared_ptr<T> Create() {
    static_assert(std::is_convertible<T*, Widget*>(),
                  "Only widgets can be created.");
    return std::make_shared<T>();
  }

  std::shared_ptr<Widget> Name(std::string const &name);
  void setParent(std::shared_ptr<Widget> parent);

  /** add child */
  // virtual std::shared_ptr<Widget> operator+(std::shared_ptr<Widget> child);
  virtual std::shared_ptr<Widget> append(std::shared_ptr<Widget> child);

  /** build imgui */
  virtual void build() = 0;

  virtual ~Widget() = default;
};

} // namespace ui
} // namespace svulkan2
