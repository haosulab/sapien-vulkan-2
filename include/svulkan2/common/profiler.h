#pragma once

namespace svulkan2 {

void ProfilerEvent(char const *name);
void ProfilerBlockBegin(char const *name);
void ProfilerBlockEnd();

class ProfilerBlock {
public:
  ProfilerBlock(char const *name) { ProfilerBlockBegin(name); }
  ~ProfilerBlock() { ProfilerBlockEnd(); }
};

}

#define SVULKAN2_PROFILE_CONCAT_(prefix, suffix) prefix##suffix
#define SVULKAN2_PROFILE_CONCAT(prefix, suffix) SVULKAN2_PROFILE_CONCAT_(prefix, suffix)
#define SVULKAN2_PROFILE_FUNCTION ::svulkan2::ProfilerBlock SVULKAN2_PROFILE_CONCAT(svulkan2_profiler_block_, __LINE__)(__func__);
#define SVULKAN2_PROFILE_BLOCK(name) ::svulkan2::ProfilerBlock SVULKAN2_PROFILE_CONCAT(svulkan2_profiler_block_,__LINE__)(#name);
#define SVULKAN2_PROFILE_BLOCK_BEGIN(name) ::svulkan2::ProfilerBlockBegin(#name);
#define SVULKAN2_PROFILE_BLOCK_END ::svulkan2::ProfilerBlockEnd();
