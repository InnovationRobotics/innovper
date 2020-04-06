/*
 * SmartLoader_emxutil.h
 *
 * Code generation for function 'SmartLoader_emxutil'
 *
 */

#ifndef SMARTLOADER_EMXUTIL_H
#define SMARTLOADER_EMXUTIL_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "SmartLoader_types.h"

/* Function Declarations */
extern void emxCopyStruct_struct_T(d_struct_T *dst, const d_struct_T *src);
extern void emxEnsureCapacity_boolean_T(emxArray_boolean_T *emxArray, int
  oldNumel);
extern void emxEnsureCapacity_cell_wrap_0(cell_wrap_0 data[64], int size[2], int
  oldNumel);
extern void emxEnsureCapacity_int16_T(emxArray_int16_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_int32_T(emxArray_int32_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_real32_T(emxArray_real32_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_real_T(emxArray_real_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_struct_T(emxArray_struct_T *emxArray, int oldNumel);
extern void emxEnsureCapacity_struct_T1(b_emxArray_struct_T *emxArray, int
  oldNumel);
extern void emxFreeMatrix_cell_wrap_0(cell_wrap_0 pMatrix[2]);
extern void emxFreeStruct_struct_T(d_struct_T *pStruct);
extern void emxFree_boolean_T(emxArray_boolean_T **pEmxArray);
extern void emxFree_cell_wrap_0_64x1(emxArray_cell_wrap_0_64x1 *pEmxArray);
extern void emxFree_int16_T(emxArray_int16_T **pEmxArray);
extern void emxFree_int32_T(emxArray_int32_T **pEmxArray);
extern void emxFree_real32_T(emxArray_real32_T **pEmxArray);
extern void emxFree_real_T(emxArray_real_T **pEmxArray);
extern void emxFree_struct_T(b_emxArray_struct_T **pEmxArray);
extern void emxFree_struct_T1(emxArray_struct_T **pEmxArray);
extern void emxInitMatrix_cell_wrap_0(cell_wrap_0 pMatrix[2]);
extern void emxInitStruct_struct_T(b_struct_T *pStruct);
extern void emxInitStruct_struct_T1(d_struct_T *pStruct);
extern void emxInit_boolean_T(emxArray_boolean_T **pEmxArray, int numDimensions);
extern void emxInit_cell_wrap_0_64x1(emxArray_cell_wrap_0_64x1 *pEmxArray);
extern void emxInit_int16_T(emxArray_int16_T **pEmxArray, int numDimensions);
extern void emxInit_int32_T(emxArray_int32_T **pEmxArray, int numDimensions);
extern void emxInit_real32_T(emxArray_real32_T **pEmxArray, int numDimensions);
extern void emxInit_real_T(emxArray_real_T **pEmxArray, int numDimensions);
extern void emxInit_struct_T(b_emxArray_struct_T **pEmxArray, int numDimensions);
extern void emxInit_struct_T1(emxArray_struct_T **pEmxArray, int numDimensions);

#endif

/* End of code generation (SmartLoader_emxutil.h) */
