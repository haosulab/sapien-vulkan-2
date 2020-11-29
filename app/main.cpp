#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/common/fs.h"

using namespace svulkan2;

int main() {
  GbufferShaderConfig gbuffer;
  gbuffer.parseGLSL("../shader/default/gbuffer.vert", "../shader/default/gbuffer.frag");
  return 0;
}
