/*
 * Copyright 2025 Hillbot Inc.
 * Copyright 2020-2024 UCSD SU Lab
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "svulkan2/common/glm.h"
#include "widget.h"

namespace svulkan2 {
namespace ui {

UI_CLASS(Window) {
  UI_DECLARE_APPEND(Window);
  UI_DECLARE_LABEL(Window);
  UI_ATTRIBUTE(Window, glm::vec2, Pos);
  UI_ATTRIBUTE(Window, glm::vec2, Size);

public:
  void build() override;
};

} // namespace ui
} // namespace svulkan2