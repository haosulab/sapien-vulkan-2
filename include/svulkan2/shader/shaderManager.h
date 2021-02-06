#pragma once
#include <memory>
#include <string>
#include <map>
#include "svulkan2/shader/gbuffer.h"
#include "svulkan2/shader/deferred.h"
#include "svulkan2/shader/composite.h"

namespace svulkan2 {

	// TODO : use config.h for definition of RendererConfig and ShaderConfig.
	/** Renderer options configured by API */
	struct RendererConfig {
		std::string shaderDir;
		vk::Format renderTargetFormat; // R8G8B8A8Unorm, R32G32B32A32Sfloat
		vk::Format depthFormat;        // D32Sfloat
	};

	/** Options configured by the shaders  */
	struct ShaderConfig {
		enum MaterialPipeline { eMETALLIC, eSPECULAR } materialPipeline;
		std::shared_ptr<InputDataLayout> vertexLayout;
		std::shared_ptr<StructDataLayout> objectBufferLayout;
		std::shared_ptr<StructDataLayout> sceneBufferLayout;
		std::shared_ptr<StructDataLayout> cameraBufferLayout;
	};

namespace shader {
	enum class TextureOperation
	{
		eTextureNoOp,
		eTextureRead,
		eTextureWrite
	};

	class ShaderManager {
		unsigned int mNumPasses;
		std::shared_ptr<RendererConfig> mRenderConfig;
		std::shared_ptr<ShaderConfig> mShaderConfig;

		std::shared_ptr<GbufferPassParser> mGbufferPass;
		std::shared_ptr<DeferredPassParser> mDeferredPass;
		std::vector<std::shared_ptr<CompositePassParser>> mCompositePasses;
		std::map<std::weak_ptr<BaseParser>, unsigned int, std::owner_less<>> mPassIndex;
		std::unordered_map<std::string, std::vector<TextureOperation>> mTextureOperationTable;
		std::vector<vk::Pipeline> mPipelines;
	public:
		ShaderManager(std::shared_ptr<RendererConfig> config=nullptr);
		void processShadersInFolder(std::string folder);
		
		std::shared_ptr<RendererConfig> getConfig() const {
			return mRenderConfig;
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
		std::vector<vk::Pipeline> getPipelines() const {
			return mPipelines;
		}
		std::vector<vk::PipelineLayout> getPipelinesLayouts();// call only after createPipelines.

		std::vector<vk::Pipeline> createPipelines(vk::Device device, vk::CullModeFlags cullMode, vk::FrontFace frontFace,
													int numDirectionalLights = -1, int numPointLights = -1);
	private:
		void populateShaderConfig();
		void prepareTextureOperationTable();
		void preparePipelineLayout(vk::Device device);
		TextureOperation getNextOperation(std::string texName, std::shared_ptr<BaseParser> pass);
		TextureOperation getPrevOperation(std::string texName, std::shared_ptr<BaseParser> pass);
		std::unordered_map<std::string, std::pair<vk::ImageLayout, vk::ImageLayout>> getTextureLayouts(std::shared_ptr<BaseParser> pass,
			std::shared_ptr<OutputDataLayout> outputLayout);
	};

} // namespace shader
} // namespace svulkan2
