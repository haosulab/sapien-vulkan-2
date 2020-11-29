#include "svulkan2/common/log.h"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace svulkan2 {
namespace log {
std::shared_ptr<spdlog::logger> getLogger() {
  static auto logger = spdlog::stderr_color_mt("svulkan2");
  return logger;
}
} // namespace log
} // namespace svulkan2
