#include <filesystem>
#include <set>
#include "svulkan2\shader\shaderManager.h"

namespace fs = std::filesystem;

namespace svulkan2 {
namespace shader {

ShaderManager::ShaderManager(std::shared_ptr<RendererConfig> config) : mRenderConfig(config)
{
	numPasses = 0;
	mGbufferPass = std::make_shared<GbufferPassParser>();
	mDeferredPass = std::make_shared<DeferredPassParser>();
}

void ShaderManager::processShadersInFolder(std::string path)
{
	//TODO : raise error if gbuffer or deferred shaders are missing.
	std::string vsFile = path + "/gbuffer.vert";
	std::string fsFile = path + "/gbuffer.frag";
	mGbufferPass->loadGLSLFiles(vsFile, fsFile);
	mPassIndex[mGbufferPass] = numPasses++;

	vsFile = path + "/deferred.vert";
	fsFile = path + "/deferred.frag";
	mDeferredPass->loadGLSLFiles(vsFile, fsFile);
	mPassIndex[mDeferredPass] = numPasses++;

	int numCompositePasses = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		std::string filename = entry.path().filename().string();
		if (filename.substr(0, 9) == "composite" && filename.substr(filename.length() - 5, 5) == ".frag")
			numCompositePasses++;
	}

	mCompositePasses.resize(numCompositePasses);
	vsFile = path + "/composite.vert";
	for (int i = 0; i < numCompositePasses; i++) {
		fsFile = path + "/composite" + std::to_string(i) + ".frag";
		mCompositePasses[i] = std::make_shared<CompositePassParser>();
		mCompositePasses[i]->loadGLSLFiles(vsFile, fsFile);
		mPassIndex[mCompositePasses[i]] = numPasses++;
	}
	prepareTextureOperationTable();
}

inline std::string getOutTextureName(std::string variableName) {// remove "out" prefix
	return variableName.substr(3, std::string::npos);
}

inline std::string getInTextureName(std::string variableName) {// remove "sampler" prefix
	return variableName.substr(7, std::string::npos);
}

void ShaderManager::prepareTextureOperationTable() {
	//process gbuffer out textures:
	for (auto& elem : mGbufferPass->getTextureOutputLayout()->elements) {
		std::string texName = getOutTextureName(elem.second.name);
		mTextureOperationTable[texName] = std::vector<TextureOperation>(numPasses, TextureOperation::eTextureNoOp);
		mTextureOperationTable[texName][mPassIndex[mGbufferPass]] = TextureOperation::eTextureWrite;
	}

	// process input textures of deferred pass:
	for (auto& elem : mDeferredPass->getCombinedSamplerLayout()->elements) {
		std::string texName = getInTextureName(elem.second.name);
		if (texName == "Depth") {
			//TODO : validate presence of depth output in gbuffer pass
		}
		else if(texName == "Shadow"){
			//TODO : validate presence of depth output in shadow pass
		}
		if(mTextureOperationTable.find(texName) != mTextureOperationTable.end())// must be render target of previous pass
			mTextureOperationTable[texName][mPassIndex[mDeferredPass]] = TextureOperation::eTextureRead;
	}
	//process out textures of deferred paas:
	for (auto& elem : mDeferredPass->getTextureOutputLayout()->elements) {
		std::string texName = getOutTextureName(elem.second.name);
		if (mTextureOperationTable.find(texName) == mTextureOperationTable.end())
			mTextureOperationTable[texName] = std::vector<TextureOperation>(numPasses, TextureOperation::eTextureNoOp);
		mTextureOperationTable[texName][mPassIndex[mDeferredPass]] = TextureOperation::eTextureWrite;
	}

	for(int i = 0; i < mCompositePasses.size(); i++){
		auto compositePass = mCompositePasses[i];
		// process input textures of composite pass:
		for (auto& elem : compositePass->getCombinedSamplerLayout()->elements) {
			std::string texName = getInTextureName(elem.second.name);
			if (mTextureOperationTable.find(texName) != mTextureOperationTable.end())// must be render target of previous passes
				mTextureOperationTable[texName][mPassIndex[mCompositePasses[i]]] = TextureOperation::eTextureRead;
		}
		//add composite out texture to the set:
		for (auto& elem : compositePass->getTextureOutputLayout()->elements) {
			std::string texName = getOutTextureName(elem.second.name);
			if (mTextureOperationTable.find(texName) == mTextureOperationTable.end())
				mTextureOperationTable[texName] = std::vector<TextureOperation>(numPasses, TextureOperation::eTextureNoOp);
			mTextureOperationTable[texName][mPassIndex[mCompositePasses[i]]] = TextureOperation::eTextureWrite;
		}
	}
}
TextureOperation ShaderManager::getNextOperation(std::string texName, std::shared_ptr<BaseParser> pass) {
	TextureOperation nextOp = TextureOperation::eTextureNoOp;
	unsigned int i = mPassIndex[pass] + 1;
	while (i < numPasses) {
		TextureOperation op = mTextureOperationTable[texName][i];
		if (op != TextureOperation::eTextureNoOp)
			return op;
	}
	return nextOp;
}

std::unordered_map<std::string, vk::ImageLayout> ShaderManager::getTextureFinalLayoutPass(std::shared_ptr<BaseParser> pass, std::shared_ptr<OutputDataLayout> outputLayout) {
	std::unordered_map<std::string, vk::ImageLayout> finalLayouts;
	for (auto& elem : outputLayout->elements) {
		std::string texName = getOutTextureName(elem.second.name);
		TextureOperation op = getNextOperation(texName, pass);
		finalLayouts[texName] = op == TextureOperation::eTextureWrite ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	return finalLayouts;
}

} // namespace shader
} // namespace svulkan2
