#pragma once
#include <memory>
#include <string>
#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/composite.h"
class Config;// forward declaration

namespace svulkan2 {
namespace shader {
	class ShaderManager {
		std::shared_ptr<Config> mConfig;
		std::shared_ptr<GbufferPassParser> mGbufferPass;
		std::shared_ptr<DeferredPassParser> mDeferredPass;
		std::vector<std::shared_ptr<CompositePassParser>> mCompositePasses;
	public:
		ShaderManager(std::shared_ptr<Config> config=nullptr);
		void processShadersInFolder(std::string folder);
		
		std::shared_ptr<Config> getConfig() const {
			return mConfig;
		}
		std::shared_ptr<GbufferPassParser> getGbufferPass() const {
			return mGbufferPass;
		}
		std::shared_ptr<DeferredPassParser> getDeferredPass() const {
			return mDeferredPass;
		}
		std::vector<std::shared_ptr<CompositePassParser>> getCompositePasses() const {
			return mCompositePasses;
		}
	private:
		void validate() const;
	};

} // namespace shader
} // namespace svulkan2
