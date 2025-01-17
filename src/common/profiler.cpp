#include "svulkan2/common/profiler.h"
#ifdef SVULKAN2_CUDA_INTEROP
#include <nvtx3/nvToolsExt.h>
#endif

namespace svulkan2 {
#ifdef SVULKAN2_CUDA_INTEROP
void ProfilerEvent(char const *name) { nvtxMarkA(name); }
void ProfilerBlockBegin(char const *name) { nvtxRangePushA(name); }
void ProfilerBlockEnd() { nvtxRangePop(); }
#else
void ProfilerEvent(char const *name) { }
void ProfilerBlockBegin(char const *name) { }
void ProfilerBlockEnd() { }
#endif
} // namespace sapien
