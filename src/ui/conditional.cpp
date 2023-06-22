#include "svulkan2/ui/conditional.h"

namespace svulkan2 {
namespace ui {

void Conditional::build() {
  if (mConditionGetter && mConditionGetter()) {
    for (uint32_t i = 0; i < mChildren.size(); ++i) {
      mChildren[i]->build();
    }
  }
}

} // namespace ui
} // namespace svulkan2
