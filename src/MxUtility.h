#if 0
#pragma once

#include <mat.h>;
#include <assert.h>
#include <memory.h>
#include <vector>

namespace IAIRoboticsAlgorithms
{
	namespace MxUtility
	{
		//class IAIRobotics_EmxArray_real_T
		//{
		//public:
		//	IAIRobotics_EmxArray_real_T() {};
		//	~IAIRobotics_EmxArray_real_T() {};

		//	emxArray_real_T ReadMatlabVariable(const char* matFile, const char* variableName)
		//	{

		//		double* ReadMatlabVariable(const char* matFile, const char* variableName)
		//	}

		//	// emxArray_real_T scope is valid as long as this is valid
		//	emxArray_real_T GetPtrData()
		//	{
		//		emxArray_real_T outputData;
		//		outputData.data = &data[0];
		//		outputData.size = &size[0];
		//		outputData.allocatedSize = allocatedSize;
		//		outputData.canFreeData = false;

		//		return outputData;
		//	}

		//public:
		//	std::vector<double> data;
		//	std::vector<int> size;
		//	int allocatedSize;
		//	int numDimensions;
		//	boolean_T canFreeData;
		//};


		/*
		class MatlabVariable
		{
		public:
			static MatlabVariable ReadMatlabVariable(const char* matFile, const char* variableName)
			{
				MatlabVariable matlabDataVariable;

				// Load mxData from Matlab - for this test
				MATFile* pMat = matOpen(matFile, "r");
				assert(pMat);

				mxArray* arr = matGetVariable(pMat, variableName);

				assert(arr);
				assert(!mxIsEmpty(arr));

				matlabDataVariable.m_numDim = mxGetNumberOfDimensions(arr);
				matlabDataVariable.m_numElem = mxGetNumberOfElements(arr);
				matlabDataVariable.m_elementSize = mxGetElementSize(arr);
				matlabDataVariable.m_isDouble = mxIsDouble(arr);

				matlabDataVariable.m_dataSizeForeachDim.resize(matlabDataVariable.m_numDim);
				auto* dimArr = mxGetDimensions(arr);
				for (auto j = 0; j < matlabDataVariable.m_numDim; j++)
				{
					matlabDataVariable.m_dataSizeForeachDim[j] = dimArr[j];
				}

				const double* ptr = mxGetPr(arr);
				auto lenByte = matlabDataVariable.m_elementSize * matlabDataVariable.m_numElem;
				matlabDataVariable.m_data.resize(lenByte);
				memcpy(&matlabDataVariable.m_data[0], ptr, lenByte);

				mxDestroyArray(arr);
				matClose(pMat);

				return matlabDataVariable;
			}

			~MatlabVariable() {};

			emxArray_real_T GetDataAsEmx()
			{
				emxArray_real_T outputData;
				outputData.data = (double*)&m_data[0];
				outputData.size = &m_dataSizeForeachDim[0];
				outputData.allocatedSize = m_data.size();
				outputData.canFreeData = false;

				return outputData;
			}

		private:
			MatlabVariable() {};

			int m_numDim, m_numElem, m_elementSize;
			bool m_isDouble;
			std::vector<char> m_data;
			std::vector<int> m_dataSizeForeachDim;
		};
		*/
		// The caller is reposible to free() the memory 
		//double* ReadMatlabVariable(const char* matFile);

		// The caller is reposible to free() the memory 
#ifdef 0
		double* ReadMatlabVariable(const char* matFile, const char* variableName)
		{
			// Load mxData from Matlab - for this test
			MATFile* pMat = matOpen(matFile, "r");
			assert(pMat);

			mxArray* arr = matGetVariable(pMat, variableName);

			assert(arr);
			assert(!mxIsEmpty(arr));
			//assert(mxIsDouble(arr));
			auto numDim = mxGetNumberOfDimensions(arr);
			auto numElem = mxGetNumberOfElements(arr);
			auto elementSize = mxGetElementSize(arr);
			double* ptr = mxGetPr(arr);

			double* data = (double*)malloc(elementSize*numElem);

			memcpy(&data[0], ptr, sizeof(double)*numElem);

			mxDestroyArray(arr);
			matClose(pMat);

			return data;
		}


		bool WriteMatFile(const char* dstMatFileName, void* srcDataToSave, int rows, int cols, int sizeOfElementInByte, mxClassID mxClassId = mxDOUBLE_CLASS,
			mxComplexity complexity = mxREAL, char* variableName = "LocalFromCpp")
		{
			auto* pmat = matOpen(dstMatFileName, "w");
			if (pmat == NULL) {
				printf("Error creating file %s\n", dstMatFileName);
				printf("(Do you have write permission in this directory?)\n");
				assert(false);
				return false;
			}

			auto* numericMatrix = mxCreateNumericMatrix(rows, cols, mxClassId, complexity);
			assert(numericMatrix);

			memcpy(mxGetPr(numericMatrix), srcDataToSave, rows * cols * sizeOfElementInByte);

			auto status1 = matPutVariable(pmat, variableName, numericMatrix);
			assert(status1 == 0);

			mxDestroyArray(numericMatrix);

			auto status2 = matClose(pmat);
			if (status2 != 0) {
				printf("Error closing file %s\n", dstMatFileName);
				assert(false);
				return false;
			}

			return true;
		}
#endif 
		//emxArray_real_T *argInit_Unboundedx4_real_T(const char* matFile)
		//{
		//	emxArray_real_T *result;
		//	static int iv6[2] = { 2, 4 };

		//	int idx0;
		//	int idx1;

		//	/* Set the size of the array.
		//	Change this size to the value that the application requires. */
		//	result = emxCreateND_real_T(2, iv6);

		//	/* Loop over the array to initialize each element. */
		//	for (idx0 = 0; idx0 < result->size[0U]; idx0++) {
		//		for (idx1 = 0; idx1 < 4; idx1++) {
		//			/* Set the value of the array element.
		//			Change this value to the value that the application requires. */
		//			result->data[idx0 + result->size[0] * idx1] = argInit_real_T();
		//		}
		//	}

		//	return result;
		//}
	}
}
#endif