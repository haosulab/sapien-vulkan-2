#pragma once

#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/shader/primitive.h"
#include "svulkan2/shader/primitive_shadow.h"
#include "svulkan2/shader/shadow.h"

#include <memory>
namespace svulkan2 {
namespace core {
class Context;
}

namespace shader {

struct ShaderPackDescription {
  std::string dirname;

  inline bool operator==(ShaderPackDescription const &other) const {
    return dirname == other.dirname;
  }
};

class ShaderPack {
public:
  enum class RenderTargetOperation { eNoOp, eRead, eColorWrite, eDepthWrite };
  ShaderPack(std::string const &dirname);

  std::shared_ptr<ShaderConfig> getShaderInputLayouts() const {
    return mShaderInputLayouts;
  }
  std::unordered_map<std::string,
                     std::vector<ShaderPack::RenderTargetOperation>>
  getTextureOperationTable() const {
    return mTextureOperationTable;
  }

  inline std::vector<std::shared_ptr<BaseParser>> getNonShadowPasses() const {
    return mNonShadowPasses;
  }

  inline std::vector<std::shared_ptr<GbufferPassParser>>
  getGbufferPasses() const {
    return mGbufferPasses;
  }

  inline std::vector<std::shared_ptr<PointPassParser>> getPointPasses() const {
    return mPointPasses;
  }

  inline std::vector<std::shared_ptr<LinePassParser>> getLinePasses() const {
    return mLinePasses;
  }

  inline std::shared_ptr<ShadowPassParser> getShadowPass() const {
    return mShadowPass;
  }

  inline std::shared_ptr<PointShadowParser> getPointShadowPass() const {
    return mPointShadowPass;
  }

  inline bool hasDeferredPass() const { return mHasDeferred; }
  inline bool hasLinePass() const { return !mLinePasses.empty(); }
  inline bool hasPointPass() const { return !mPointPasses.empty(); }

private:
  std::shared_ptr<ShaderConfig> generateShaderInputLayouts() const;
  std::unordered_map<std::string,
                     std::vector<ShaderPack::RenderTargetOperation>>
  generateTextureOperationTable() const;

  std::shared_ptr<ShadowPassParser> mShadowPass{};
  std::shared_ptr<PointShadowParser> mPointShadowPass{};

  std::vector<std::shared_ptr<BaseParser>> mNonShadowPasses;
  std::vector<std::shared_ptr<GbufferPassParser>> mGbufferPasses;
  std::vector<std::shared_ptr<PointPassParser>> mPointPasses;
  std::vector<std::shared_ptr<LinePassParser>> mLinePasses;

  std::unordered_map<std::string, std::vector<RenderTargetOperation>>
      mTextureOperationTable;

  std::shared_ptr<ShaderConfig> mShaderInputLayouts;

  bool mHasDeferred{};
};

} // namespace shader
} // namespace svulkan2
