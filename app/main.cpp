#include "svulkan2/common/fs.h"
#include "svulkan2/common/log.h"
#include "svulkan2/shader/shader.h"

using namespace svulkan2;
using namespace svulkan2::shader;

int main() {
  GbufferPassParser gbuffer;
  gbuffer.loadGLSLFiles("../shader/default/gbuffer.vert",
                        "../shader/default/gbuffer.frag");
  DeferredPassParser deferred;
  deferred.loadGLSLFiles("../shader/default/deferred.vert",
                         "../shader/default/deferred.frag");
  CompositePassParser composite;
  composite.loadGLSLFiles("../shader/default/composite.vert",
                         "../shader/default/composite.frag");
  return 0;
}
