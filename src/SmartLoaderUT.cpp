#include <stdio.h>
#include <ppl.h>
#include <omp.h>
#include <BinaryIO.h>
#include <MathUtility.h>
#include <StopWatch.h>
#include <StopWatchSingletonAcrossDlls.h>

#include "SmartLoader_types.h"
#include "SmartLoader.h"


void TestFromBinFile()
{
	omp_set_num_threads(1);

	//std::string datasetPath = "D:\\git\\cpp\\SmartLoader\\SmartLoaderDataset\\Test1\\";
	//std::string xyzPath = "xyz$$c1_d_w3_h504000.bin", intensityPath = "intensity$$c1_d_w1_h504000.bin";

	std::string datasetPath = "D:\\git\\cpp\\SmartLoader\\SmartLoaderDataset\\Test2\\";
	std::string xyzPath = "xyz$$c1_d_w3_h17235.bin", intensityPath = "intensity$$c1_d_w1_h17235.bin";

	SmartLoaderConfigParam configParams;
	memset(&configParams, 0x00, sizeof(configParams));
	configParams.timeTagMs = 1;
	configParams.planeModelParameters[0] = -0.028009425848722;
	configParams.planeModelParameters[1] = -5.510213086381555e-04;
	configParams.planeModelParameters[2] = 0.999607503414154;
	configParams.planeModelParameters[3] = 2.846343278884888;
	configParams.useExternalProjectionMatrix = false;
	// configParams.externalProjectionMatrix is already set to zero
	double xyzLimits[6]= { -1.681562873721123, 1.918437126278877, -9999, 9999, -3.361619710922240, -2.361619710922240};
	memcpy(configParams.xyzLimits, xyzLimits, sizeof(configParams.xyzLimits));
	configParams.minNumPointsInPc = 64;
	configParams.minimumDistanceFromLoaderToPlaneMeter = 0.2;
	configParams.minPointsForReflector = 5;
	configParams.maximumTimeTagDiffMs = 3000;
	configParams.minimumIntensityReflectorValue = 100;
	configParams.loaderReflectorDiameterMeter = 0.18;
	configParams.reflectorMaxZaxisDistanceForOutlierMeter = 0.06;
	configParams.previousLoaderLocationToCurrentLocationMaximumDistanceMeter = 0.18;
	configParams.loaderReflectorMaxZaxisDistanceForOutlierMeter = 0.04;
	configParams.maxDistanceBetweenEachRayMeter = 0.07;
	configParams.debugMode = false;

	auto SD = std::make_unique<SmartLoaderStackData>();
	auto pd = std::make_unique<SmartLoaderPersistentData>();
	SD->pd = pd.get();

	SmartLoader_initialize(SD.get());

	int xyzNumBands, xyzNumBytePerBand, xyzWidth, xyzHeight;
	std::vector<unsigned char> xyzData;
	auto retVal = IAIRoboticsAlgorithms::BinaryIO::ReadBinary((datasetPath + xyzPath).c_str(), xyzNumBands, xyzNumBytePerBand, xyzWidth, xyzHeight, xyzData);
	assert(retVal); if (!retVal) return;

	int intensityNumBands, intensityNumBytePerBand, intensityWidth, intensityHeight;
	std::vector<unsigned char> intensityData;
	retVal = IAIRoboticsAlgorithms::BinaryIO::ReadBinary((datasetPath + intensityPath).c_str(), intensityNumBands, intensityNumBytePerBand, intensityWidth, intensityHeight, intensityData);
	assert(retVal); if (!retVal) return;

	int xyz_size[2] = { xyzHeight, xyzWidth }, intensity_size[2] = { intensityHeight, intensityWidth };
	
	std::vector<float> heightMap_res_data;
	heightMap_res_data.resize(1024 * 1024);
	int heightMap_res_size[2] = { 0,0 };

	for (int i = 0; i < 10; i++)
	{
		SmartLoaderStruct smartLoaderStruct;

		IAIRoboticsAlgorithms::StopWatch<>::GetInstance().Start();

		SmartLoader(SD.get(), &configParams, (double*)&xyzData[0], xyz_size, (double*)&intensityData[0], intensity_size,
			&smartLoaderStruct, &heightMap_res_data[0], heightMap_res_size);

		IAIRoboticsAlgorithms::StopWatch<>::GetInstance().Stop();
	}

	IAIRoboticsAlgorithms::StopWatch<>::GetInstance().PrintRunningTimeSummery();

	SmartLoader_terminate();
}


int main()
{
	TestFromBinFile();

	return 0;
}