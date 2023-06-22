#pragma once
#include "widget.h"
#include <functional>

namespace svulkan2 {
namespace ui {

UI_CLASS(FileChooser) {
  UI_DECLARE_LABEL(FileChooser);
  UI_ATTRIBUTE(FileChooser, std::string, Title);
  UI_ATTRIBUTE(FileChooser, std::string, Filter);
  UI_ATTRIBUTE(FileChooser, std::string, Path);

  UI_ATTRIBUTE(
      FileChooser,
      std::function<void(std::shared_ptr<FileChooser>, std::string name, std::string path)>,
      Callback);

public:
  FileChooser();

  void build() override;
  void open();
  void close();
};

} // namespace ui
} // namespace svulkan2
