#include "MxReadVariable.h"
#include <assert.h>
#include <memory.h>
#include <matrix.h>
#include <mat.h>


// #define IAIRoboticsAlgorithms_SUPPORT_MATLAB_TYPES 
#ifdef IAIRoboticsAlgorithms_SUPPORT_MATLAB_TYPES 
/* Type Definitions */
#ifndef struct_emxArray_boolean_T
#define struct_emxArray_boolean_T

struct emxArray_boolean_T
{
	boolean_T *data;
	int *size;
	int allocatedSize;
	int numDimensions;
	boolean_T canFreeData;
};

#endif                                 /*struct_emxArray_boolean_T*/

#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T

struct emxArray_real_T
{
	double *data;
	int *size;
	int allocatedSize;
	int numDimensions;
	boolean_T canFreeData;
};

#endif     
#endif   


namespace IAIRoboticsAlgorithms
{
		MatlabVariable MatlabVariable::ReadMatlabVariable(const char* matFile, const char* variableName)
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
			matlabDataVariable.m_isComplex = mxIsComplex(arr);

			auto& dataSizeForeachDim = matlabDataVariable.m_dataSizeForeachDim;
			dataSizeForeachDim.resize(matlabDataVariable.m_numDim);
			auto* dimArr = mxGetDimensions(arr);
			for (auto j = 0; j < matlabDataVariable.m_numDim; j++)
			{
				dataSizeForeachDim[j] = dimArr[j];
			}

			const double* ptr = mxGetPr(arr);
			auto lenByte = matlabDataVariable.m_elementSize * matlabDataVariable.m_numElem;
			matlabDataVariable.m_data.resize(lenByte);
			memcpy(&matlabDataVariable.m_data[0], ptr, lenByte);

			mxDestroyArray(arr);
			matClose(pMat);

			return matlabDataVariable;
		}


		bool MatlabVariable::WriteMatFile(const char* dstMatFileName, void* srcDataToSave, int rows, int cols, int sizeOfElementInByte, mxClassID mxClassId,
			mxComplexity complexity, char* variableName)
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
#ifdef IAIRoboticsAlgorithms_SUPPORT_MATLAB_TYPES 
		emxArray_real_T GetDataAsEmx()
		{
			emxArray_real_T outputData;

			outputData.data = (double*)&m_data[0];
			outputData.size = &m_dataSizeForeachDim[0];
			outputData.allocatedSize = m_data.size();
			outputData.numDimensions = m_numDim;
			outputData.canFreeData = false;

			return outputData;
		}
#endif
};
