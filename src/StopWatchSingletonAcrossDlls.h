#pragma once

#include "StopWatch.h"
#include "IAIRoboticsAlgorithms_export.h"


namespace IAIRoboticsAlgorithms
{
	class IAIROBOTICSALGORITHMS_EXPORTS_API StopWatchSingletonAcrossDlls : public StopWatch<std::chrono::milliseconds, std::chrono::high_resolution_clock>
	{
	public:
		static StopWatchSingletonAcrossDlls& GetInstanceBetweenMultipleDlls();
	};
}