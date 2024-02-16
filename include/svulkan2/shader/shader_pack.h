#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace svulkan2 {
namespace core {
class Context;
}

namespace shader {

struct ShaderConfig;
class BaseParser;
class GbufferPassParser;
class DeferredPassParser;
class PointShadowParser;
class ShadowPassParser;
class PointPassParser;
class LinePassParser;

struct ShaderPackDescription {
  std::string dirname;

  inline bool operator==(ShaderPackDescription const &other) const {
    return dirname == other.dirname;
  }
};

class ShaderPack {
public:
  ShaderPack(std::string const &dirname);

  std::shared_ptr<ShaderConfig> getShaderInputLayouts() const { return mShaderInputLayouts; }

  inline std::vector<std::shared_ptr<BaseParser>> getNonShadowPasses() const {
    return mNonShadowPasses;
  }
  inline std::vector<std::shared_ptr<GbufferPassParser>> getGbufferPasses() const {
    return mGbufferPasses;
  }
  inline std::vector<std::shared_ptr<PointPassParser>> getPointPasses() const {
    return mPointPasses;
  }

  inline std::vector<std::shared_ptr<LinePassParser>> getLinePasses() const { return mLinePasses; }
  inline std::shared_ptr<ShadowPassParser> getShadowPass() const { return mShadowPass; }
  inline std::shared_ptr<PointShadowParser> getPointShadowPass() const { return mPointShadowPass; }

  inline bool hasDeferredPass() const { return mHasDeferred; }
  inline bool hasLinePass() const { return !mLinePasses.empty(); }
  inline bool hasPointPass() const { return !mPointPasses.empty(); }

private:
  std::shared_ptr<ShaderConfig> generateShaderLayouts() const;
  void updateMaxLightCount();

  std::shared_ptr<ShadowPassParser> mShadowPass{};
  std::shared_ptr<PointShadowParser> mPointShadowPass{};

  std::vector<std::shared_ptr<BaseParser>> mNonShadowPasses;
  std::vector<std::shared_ptr<GbufferPassParser>> mGbufferPasses;
  std::vector<std::shared_ptr<PointPassParser>> mPointPasses;
  std::vector<std::shared_ptr<LinePassParser>> mLinePasses;

  uint32_t mMaxPointLightCount{};
  uint32_t mMaxPointShadowCount{};
  uint32_t mMaxDirectionalLightCount{};
  uint32_t mMaxDirectionalShadowCount{};
  uint32_t mMaxSpotLightCount{};
  uint32_t mMaxSpotShadowCount{};
  uint32_t mMaxTexturedLightCount{};
  uint32_t mMaxTexturedShadowCount{};

  std::shared_ptr<ShaderConfig> mShaderInputLayouts;

  bool mHasDeferred{};
};

} // namespace shader
} // namespace svulkan2
