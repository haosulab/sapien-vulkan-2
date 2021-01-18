#include <filesystem>
#include <set>
#include "svulkan2\shader\shaderManager.h"

namespace fs = std::filesystem;

namespace svulkan2 {
namespace shader {

ShaderManager::ShaderManager(std::shared_ptr<Config> config) : mConfig(config)
{
	mGbufferPass = std::make_shared<GbufferPassParser>();
	mDeferredPass = std::make_shared<DeferredPassParser>();
}

void ShaderManager::processShadersInFolder(std::string path)
{
	std::string vsFile = path + "/gbuffer.vert";
	std::string fsFile = path + "/gbuffer.frag";
	mGbufferPass->loadGLSLFiles(vsFile, fsFile);

	vsFile = path + "/deferred.vert";
	fsFile = path + "/deferred.frag";
	mDeferredPass->loadGLSLFiles(vsFile, fsFile);

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
	}
	validate();
}

inline std::string getOutTextureName(std::string variableName) {// remove "out" prefix
	return variableName.substr(3, std::string::npos);
}

inline std::string getInTextureName(std::string variableName) {// remove "sampler" prefix
	return variableName.substr(7, std::string::npos);
}

void ShaderManager::validate() const {
	std::set<std::string> outTextures;
	//add gbuffer out textures to the set:
	for (auto& elem : mGbufferPass->getTextureOutputLayout()->elements)
		outTextures.insert(getOutTextureName(elem.second.name));

	// validate input textures of deferred pass:
	for (auto& elem : mDeferredPass->getCombinedSamplerLayout()->elements) {
		std::string inputTextureName = getInTextureName(elem.second.name);
		if (inputTextureName == "Depth") {
			//TODO : validate presence of depth output in gbuffer pass
		}
		else if(inputTextureName == "Shadow"){
			//TODO : validate presence of depth output in shadow pass
		}
		else {
			ASSERT(outTextures.find(inputTextureName) != outTextures.end(), "Unknown input texture for Deferred pass");
		}
	}
	//add deferred out textures to the set:
	for (auto& elem : mDeferredPass->getTextureOutputLayout()->elements) 
		outTextures.insert(getOutTextureName(elem.second.name));

	for (auto compositePass : mCompositePasses) {
		// validate input textures of composite pass:
		for (auto& elem : compositePass->getCombinedSamplerLayout()->elements) {
			std::string inputTextureName = getInTextureName(elem.second.name);
			ASSERT(outTextures.find(inputTextureName) != outTextures.end(), "Unknown input texture for Composite pass");
		}
		//add composite out texture to the set:
		for (auto& elem : compositePass->getTextureOutputLayout()->elements)
			outTextures.insert(getOutTextureName(elem.second.name));

	}
}

} // namespace shader
} // namespace svulkan2
