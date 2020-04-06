#pragma once

#include <stdio.h>
#include <mclcppclass.h>

namespace IAIRoboticsAlgorithms
{
	namespace MclUtility
	{
		void QueryMwArray(const mwArray& data)
		{
			auto a1 = data.NumberOfDimensions();
			auto a2 = data.NumberOfElements();
			auto a3 = data.NumberOfFields();
			auto a4 = data.GetDimensions();
			printf("%s\n", (const char*)data.ToString());
		}
	}
}