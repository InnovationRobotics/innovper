#include "StopWatchSingletonAcrossDlls.h"


namespace IAIRoboticsAlgorithms
{
	StopWatchSingletonAcrossDlls& StopWatchSingletonAcrossDlls::GetInstanceBetweenMultipleDlls()
	{
		static StopWatchSingletonAcrossDlls instance;
		return instance;
	}
}
