//
// File: PerceptionSmartLoader_emxutil.h
//
// MATLAB Coder version            : 4.1
// C/C++ source code generated on  : 05-May-2020 11:47:28
//
#ifndef PERCEPTIONSMARTLOADER_EMXUTIL_H
#define PERCEPTIONSMARTLOADER_EMXUTIL_H

// Include Files
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "PerceptionSmartLoader_types.h"

// Function Declarations
extern void emxEnsureCapacity_cell_wrap_13(emxArray_cell_wrap_13 *emxArray, int
  oldNumel);
extern void emxEnsureCapacity_cell_wrap_4(cell_wrap_4 data[64], int size[2], int
  oldNumel);
extern void emxEnsureCapacity_int32_T(emxArray_int32_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_real32_T(emxArray_real32_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_real_T(emxArray_real_T *emxArray, int oldNumel);
extern void emxFreeMatrix_cell_wrap_4(cell_wrap_4 pMatrix[2]);
extern void emxFree_cell_wrap_13(emxArray_cell_wrap_13 **pEmxArray);
extern void emxFree_cell_wrap_4_64x1(emxArray_cell_wrap_4_64x1 *pEmxArray);
extern void emxFree_int32_T(emxArray_int32_T **pEmxArray);
extern void emxFree_real32_T(emxArray_real32_T **pEmxArray);
extern void emxFree_real_T(emxArray_real_T **pEmxArray);
extern void emxInitMatrix_cell_wrap_4(cell_wrap_4 pMatrix[2]);
extern void emxInitStruct_struct_T(c_struct_T *pStruct);
extern void emxInit_cell_wrap_13(emxArray_cell_wrap_13 **pEmxArray, int
  numDimensions);
extern void emxInit_cell_wrap_4_64x1(emxArray_cell_wrap_4_64x1 *pEmxArray);
extern void emxInit_int32_T(emxArray_int32_T **pEmxArray, int numDimensions);
extern void emxInit_real32_T(emxArray_real32_T **pEmxArray, int numDimensions);
extern void emxInit_real_T(emxArray_real_T **pEmxArray, int numDimensions);

#endif

//
// File trailer for PerceptionSmartLoader_emxutil.h
//
// [EOF]
//
