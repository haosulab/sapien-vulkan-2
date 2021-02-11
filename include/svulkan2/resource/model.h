#pragma once
#include "shape.h"
#include <vector>

namespace svulkan2 {
namespace resource {

struct ModelDescription {
  enum SourceType { eFILE, eCUSTOM } source;
  std::string filename;
  inline bool operator==(ModelDescription const &other) const {
    return source == other.source && filename == other.filename;
  }
};

class SVModel {
  ModelDescription mDescription;
  std::vector<std::shared_ptr<SVShape>> mShapes;

  bool mLoaded{};

  /** When manager is not null, it is used to avoid loading duplicated
   * subresources
   */
  class SVResourceManager *mManager;

  std::mutex mLoadingMutex;

public:
  static std::shared_ptr<SVModel> FromFile(std::string const &filename);
  static std::shared_ptr<SVModel>
  FromData(std::vector<std::shared_ptr<SVShape>> shapes);

  inline std::vector<std::shared_ptr<SVShape>> const &getShapes() const {
    return mShapes;
  }

  /** Determine whether this model is loaded. If a model is specified by a file,
   *  it is not loaded into shapes by default
   */
  inline bool isLoaded() const { return mLoaded; }

  // void load();
  std::future<void> loadAsync();

  inline ModelDescription const &getDescription() const { return mDescription; }

  inline void setManager(class SVResourceManager *manager) {
    mManager = manager;
  };

private:
  SVModel() = default;
};

} // namespace resource
} // namespace svulkan2
