#include <cstdlib>

#include "core/logger.h"
#include "context.h"

ELAINA_NAMESPACE_BEGIN

using namespace inter;
Context::SharedPtr gpContext;
CUDATrackedMemory CUDATrackedMemory::singleton;

void Context::initialize() {
	logInfo("Initializing device context");
	// initialize cuda 
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		logFatal("No CUDA capable devices found!");
	logInfo("Found " + to_string(numDevices) + " CUDA devices!");

	// set up context
	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	logInfo("Elaina running on device: " + string(deviceProps.name));

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		logError("Error querying current context: error code " + to_string(cuRes));

	// tracked cuda device memory management
	logInfo("Setting default memory manager to CUDA memory");
	set_default_resource(&CUDATrackedMemory::singleton);
	alloc = new Allocator(&CUDATrackedMemory::singleton);
	
	logSuccess("Context initialization completed. Hello from elaina!");
}

void Context::finalize(){
	// CUDA_SYNC_CHECK();
	cudaDeviceSynchronize();
	delete alloc;
	cuCtxDestroy(cudaContext);
	
	logSuccess("Context destroyed. Goodbye!");
}

void Context::terminate() { 
	finalize(); 
	abort();
}

void Context::setGlobalConfig(const json& config) { globalConfig = config; }
json Context::getGlobalConfig() const { return globalConfig; }
void Context::updateGlobalConfig(const json &config) { globalConfig.update(config); }

ELAINA_NAMESPACE_END