#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/common/fs.h"
#include "svulkan2/common/log.h"


using namespace svulkan2;

int main() {
  GbufferPassParser gbuffer;
  gbuffer.parseGLSLFiles("../shader/default/gbuffer.vert", "../shader/default/gbuffer.frag");
  // DeferredPassParser deferred;
  // deferred.parseGLSLFiles("../shader/default/deferred.vert", "../shader/default/deferred.frag");
  return 0;
}
