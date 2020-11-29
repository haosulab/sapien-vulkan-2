#pragma once

#include <exception>
#include <spdlog/spdlog.h>
#include <string_view>

namespace svulkan2 {
namespace log {

std::shared_ptr<spdlog::logger> getLogger();

template <typename... Args>
inline void debug(spdlog::string_view_t fmt, const Args &... args) {
  getLogger()->debug(fmt, args...);
};

template <typename... Args>
inline void info(spdlog::string_view_t fmt, const Args &... args) {
  getLogger()->info(fmt, args...);
};

template <typename... Args>
inline void warn(spdlog::string_view_t fmt, const Args &... args) {
  getLogger()->warn(fmt, args...);
};

template <typename... Args>
inline void error(spdlog::string_view_t fmt, const Args &... args) {
  getLogger()->error(fmt, args...);
};

template <typename... Args>
inline void critical(spdlog::string_view_t fmt, const Args &... args) {
  getLogger()->critical(fmt, args...);
};

} // namespace log
} // namespace svulkan
