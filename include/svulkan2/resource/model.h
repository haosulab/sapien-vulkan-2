#pragma once
#include "shape.h"
#include "svulkan2/core/as.h"
#include <vector>

namespace svulkan2 {
namespace resource {

struct ModelDescription {
  enum class SourceType { eFILE, eCUSTOM } source;
  std::string filename;
  inline bool operator==(ModelDescription const &other) const {
    return source == other.source && filename == other.filename;
  }
};

class SVModel {
public:
  /** create a model based on a prototype model. Their meshes and textures will
   * be shared but materials will be copied to allow modifications. It also
   * keeps a shared_ptr to the prototype model. */
  static std::shared_ptr<SVModel> FromPrototype(std::shared_ptr<SVModel> prototype);

  static std::shared_ptr<SVModel> FromFile(std::string const &filename);
  static std::shared_ptr<SVModel> FromData(std::vector<std::shared_ptr<SVShape>> shapes);

  /** get shapes. model will load if it is not loaded. */
  std::vector<std::shared_ptr<SVShape>> const &getShapes();

  /** Determine whether this model is loaded. If a model is specified by a file,
   *  it will only be loaded from disk when calling loadAsync. If it is
   *  specified by a prototype, the loading also only happens when calling
   *  loadAsync.
   */
  inline bool isLoaded() const { return mLoaded; }
  std::future<void> loadAsync();

  inline ModelDescription const &getDescription() const { return mDescription; }

  /** must be called only in 1 thread currently */
  void buildBLAS(bool update = false);
  void recordUpdateBLAS(vk::CommandBuffer commandBuffer);
  core::BLAS *getBLAS();

  ~SVModel();

private:
  SVModel();

  std::shared_ptr<SVModel> mPrototype;
  ModelDescription mDescription;
  std::vector<std::shared_ptr<SVShape>> mShapes;
  bool mLoaded{};
  std::mutex mLoadingMutex;

  std::unique_ptr<core::BLAS> mBLAS;
};

} // namespace resource
} // namespace svulkan2
