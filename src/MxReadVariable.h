#pragma once

#include <mat.h>
#include <vector>
#include "IAIRoboticsAlgorithms_export.h"


namespace IAIRoboticsAlgorithms
{
	class IAIROBOTICSALGORITHMS_EXPORTS_API MatlabVariable
	{
	public:
		static MatlabVariable ReadMatlabVariable(const char* matFile, const char* variableName);

		static bool WriteMatFile(const char* dstMatFileName, void* srcDataToSave, int rows, int cols, int sizeOfElementInByte, mxClassID mxClassId = mxDOUBLE_CLASS,
			mxComplexity complexity = mxREAL, char* variableName = "varFromCpp");

		~MatlabVariable() {};

	public:
		// Should be get methods 
		int m_numDim, m_numElem, m_elementSize;
		bool m_isDouble, m_isComplex;
		std::vector<char> m_data;
		std::vector<int> m_dataSizeForeachDim;

	private:
		MatlabVariable() {};
	};
}