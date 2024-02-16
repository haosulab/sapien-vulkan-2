#include "logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace svulkan2 {
namespace logger {

std::shared_ptr<spdlog::logger> getLogger() {
  static std::shared_ptr<spdlog::logger> logger;
  if (!logger) {
    logger = spdlog::stderr_color_mt("svulkan2");
    logger->set_level(spdlog::level::warn);
  }
  return logger;
}

void setLogLevel(std::string_view level) {
  if (level == "off") {
    getLogger()->set_level(spdlog::level::off);
  } else if (level == "trace") {
    getLogger()->set_level(spdlog::level::trace);
  } else if (level == "debug") {
    getLogger()->set_level(spdlog::level::debug);
  } else if (level == "info") {
    getLogger()->set_level(spdlog::level::info);
  } else if (level == "warn" || level == "warning") {
    getLogger()->set_level(spdlog::level::warn);
  } else if (level == "error" || level == "err") {
    getLogger()->set_level(spdlog::level::err);
  } else if (level == "critical") {
    getLogger()->set_level(spdlog::level::critical);
  } else {
    throw std::runtime_error("unknown log level " + std::string(level));
  }
}

std::string getLogLevel() {
  switch (getLogger()->level()) {
  case spdlog::level::off:
    return "off";
  case spdlog::level::trace:
    return "trace";
  case spdlog::level::debug:
    return "debug";
  case spdlog::level::info:
    return "info";
  case spdlog::level::warn:
    return "warn";
  case spdlog::level::err:
    return "err";
  case spdlog::level::critical:
    return "critical";
  default:
    return "";
  }
}

} // namespace logger
} // namespace svulkan2
