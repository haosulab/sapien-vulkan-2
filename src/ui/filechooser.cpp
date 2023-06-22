#include "svulkan2/ui/filechooser.h"
#include <ImGuiFileDialog.h>

namespace svulkan2 {
namespace ui {

FileChooser::FileChooser() : mTitle("Choose File"), mPath(".") {}

void FileChooser::build() {
  if (ImGuiFileDialog::Instance()->Display(mLabel)) {
    if (ImGuiFileDialog::Instance()->IsOk()) {
      std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
      std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
      if (mCallback) {
        mCallback(std::static_pointer_cast<FileChooser>(shared_from_this()), filePathName,
                  filePath);
      }
    }
  }
}

void FileChooser::open() {
  ImGuiFileDialog::Instance()->OpenDialog(mLabel, mTitle, mFilter.c_str(), mPath);
}

void FileChooser::close() { ImGuiFileDialog::Instance()->Close(); }

} // namespace ui
} // namespace svulkan2
