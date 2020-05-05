//
// File: PerceptionSmartLoader_emxutil.cpp
//
// MATLAB Coder version            : 4.1
// C/C++ source code generated on  : 05-May-2020 11:47:28
//

// Include Files
#include <stdlib.h>
#include <string.h>
#include "PerceptionSmartLoader.h"
#include "PerceptionSmartLoader_emxutil.h"

// Function Declarations
static void emxExpand_cell_wrap_13(emxArray_cell_wrap_13 *emxArray, int
  fromIndex, int toIndex);
static void emxExpand_cell_wrap_4_64x1(cell_wrap_4 data[64], int fromIndex, int
  toIndex);
static void emxFreeStruct_cell_wrap_4(cell_wrap_4 *pStruct);
static void emxInitStruct_cell_wrap_13(cell_wrap_13 *pStruct);
static void emxInitStruct_cell_wrap_4(cell_wrap_4 *pStruct);
static void emxTrim_cell_wrap_4_64x1(cell_wrap_4 data[64], int fromIndex, int
  toIndex);

// Function Definitions

//
// Arguments    : emxArray_cell_wrap_13 *emxArray
//                int fromIndex
//                int toIndex
// Return Type  : void
//
static void emxExpand_cell_wrap_13(emxArray_cell_wrap_13 *emxArray, int
  fromIndex, int toIndex)
{
  int i;
  for (i = fromIndex; i < toIndex; i++) {
    emxInitStruct_cell_wrap_13(&emxArray->data[i]);
  }
}

//
// Arguments    : cell_wrap_4 data[64]
//                int fromIndex
//                int toIndex
// Return Type  : void
//
static void emxExpand_cell_wrap_4_64x1(cell_wrap_4 data[64], int fromIndex, int
  toIndex)
{
  int i;
  for (i = fromIndex; i < toIndex; i++) {
    emxInitStruct_cell_wrap_4(&data[i]);
  }
}

//
// Arguments    : cell_wrap_4 *pStruct
// Return Type  : void
//
static void emxFreeStruct_cell_wrap_4(cell_wrap_4 *pStruct)
{
  emxFree_real32_T(&pStruct->f1);
}

//
// Arguments    : cell_wrap_13 *pStruct
// Return Type  : void
//
static void emxInitStruct_cell_wrap_13(cell_wrap_13 *pStruct)
{
  pStruct->f1.size[0] = 0;
  pStruct->f1.size[1] = 0;
}

//
// Arguments    : cell_wrap_4 *pStruct
// Return Type  : void
//
static void emxInitStruct_cell_wrap_4(cell_wrap_4 *pStruct)
{
  emxInit_real32_T(&pStruct->f1, 2);
}

//
// Arguments    : cell_wrap_4 data[64]
//                int fromIndex
//                int toIndex
// Return Type  : void
//
static void emxTrim_cell_wrap_4_64x1(cell_wrap_4 data[64], int fromIndex, int
  toIndex)
{
  int i;
  for (i = fromIndex; i < toIndex; i++) {
    emxFreeStruct_cell_wrap_4(&data[i]);
  }
}

//
// Arguments    : emxArray_cell_wrap_13 *emxArray
//                int oldNumel
// Return Type  : void
//
void emxEnsureCapacity_cell_wrap_13(emxArray_cell_wrap_13 *emxArray, int
  oldNumel)
{
  int newNumel;
  int i;
  void *newData;
  if (oldNumel < 0) {
    oldNumel = 0;
  }

  newNumel = 1;
  for (i = 0; i < emxArray->numDimensions; i++) {
    newNumel *= emxArray->size[i];
  }

  if (newNumel > emxArray->allocatedSize) {
    i = emxArray->allocatedSize;
    if (i < 16) {
      i = 16;
    }

    while (i < newNumel) {
      if (i > 1073741823) {
        i = MAX_int32_T;
      } else {
        i <<= 1;
      }
    }

    newData = calloc((unsigned int)i, sizeof(cell_wrap_13));
    if (emxArray->data != NULL) {
      memcpy(newData, emxArray->data, sizeof(cell_wrap_13) * oldNumel);
      if (emxArray->canFreeData) {
        free(emxArray->data);
      }
    }

    emxArray->data = (cell_wrap_13 *)newData;
    emxArray->allocatedSize = i;
    emxArray->canFreeData = true;
  }

  if (oldNumel > newNumel) {
    emxExpand_cell_wrap_13(emxArray, oldNumel, newNumel);
  }
}

//
// Arguments    : cell_wrap_4 data[64]
//                int size[2]
//                int oldNumel
// Return Type  : void
//
void emxEnsureCapacity_cell_wrap_4(cell_wrap_4 data[64], int size[2], int
  oldNumel)
{
  int newNumel;
  if (oldNumel < 0) {
    oldNumel = 0;
  }

  newNumel = size[0] * size[1];
  if (oldNumel > newNumel) {
    emxTrim_cell_wrap_4_64x1(data, newNumel, oldNumel);
  } else {
    if (oldNumel < newNumel) {
      emxExpand_cell_wrap_4_64x1(data, oldNumel, newNumel);
    }
  }
}

//
// Arguments    : emxArray_int32_T *emxArray
//                int oldNumel
// Return Type  : void
//
void emxEnsureCapacity_int32_T(emxArray_int32_T *emxArray, int oldNumel)
{
  int newNumel;
  int i;
  void *newData;
  if (oldNumel < 0) {
    oldNumel = 0;
  }

  newNumel = 1;
  for (i = 0; i < emxArray->numDimensions; i++) {
    newNumel *= emxArray->size[i];
  }

  if (newNumel > emxArray->allocatedSize) {
    i = emxArray->allocatedSize;
    if (i < 16) {
      i = 16;
    }

    while (i < newNumel) {
      if (i > 1073741823) {
        i = MAX_int32_T;
      } else {
        i <<= 1;
      }
    }

    newData = calloc((unsigned int)i, sizeof(int));
    if (emxArray->data != NULL) {
      memcpy(newData, emxArray->data, sizeof(int) * oldNumel);
      if (emxArray->canFreeData) {
        free(emxArray->data);
      }
    }

    emxArray->data = (int *)newData;
    emxArray->allocatedSize = i;
    emxArray->canFreeData = true;
  }
}

//
// Arguments    : emxArray_real32_T *emxArray
//                int oldNumel
// Return Type  : void
//
void emxEnsureCapacity_real32_T(emxArray_real32_T *emxArray, int oldNumel)
{
  int newNumel;
  int i;
  void *newData;
  if (oldNumel < 0) {
    oldNumel = 0;
  }

  newNumel = 1;
  for (i = 0; i < emxArray->numDimensions; i++) {
    newNumel *= emxArray->size[i];
  }

  if (newNumel > emxArray->allocatedSize) {
    i = emxArray->allocatedSize;
    if (i < 16) {
      i = 16;
    }

    while (i < newNumel) {
      if (i > 1073741823) {
        i = MAX_int32_T;
      } else {
        i <<= 1;
      }
    }

    newData = calloc((unsigned int)i, sizeof(float));
    if (emxArray->data != NULL) {
      memcpy(newData, emxArray->data, sizeof(float) * oldNumel);
      if (emxArray->canFreeData) {
        free(emxArray->data);
      }
    }

    emxArray->data = (float *)newData;
    emxArray->allocatedSize = i;
    emxArray->canFreeData = true;
  }
}

//
// Arguments    : emxArray_real_T *emxArray
//                int oldNumel
// Return Type  : void
//
void emxEnsureCapacity_real_T(emxArray_real_T *emxArray, int oldNumel)
{
  int newNumel;
  int i;
  void *newData;
  if (oldNumel < 0) {
    oldNumel = 0;
  }

  newNumel = 1;
  for (i = 0; i < emxArray->numDimensions; i++) {
    newNumel *= emxArray->size[i];
  }

  if (newNumel > emxArray->allocatedSize) {
    i = emxArray->allocatedSize;
    if (i < 16) {
      i = 16;
    }

    while (i < newNumel) {
      if (i > 1073741823) {
        i = MAX_int32_T;
      } else {
        i <<= 1;
      }
    }

    newData = calloc((unsigned int)i, sizeof(double));
    if (emxArray->data != NULL) {
      memcpy(newData, emxArray->data, sizeof(double) * oldNumel);
      if (emxArray->canFreeData) {
        free(emxArray->data);
      }
    }

    emxArray->data = (double *)newData;
    emxArray->allocatedSize = i;
    emxArray->canFreeData = true;
  }
}

//
// Arguments    : cell_wrap_4 pMatrix[2]
// Return Type  : void
//
void emxFreeMatrix_cell_wrap_4(cell_wrap_4 pMatrix[2])
{
  emxFreeStruct_cell_wrap_4(&pMatrix[0]);
  emxFreeStruct_cell_wrap_4(&pMatrix[1]);
}

//
// Arguments    : emxArray_cell_wrap_13 **pEmxArray
// Return Type  : void
//
void emxFree_cell_wrap_13(emxArray_cell_wrap_13 **pEmxArray)
{
  if (*pEmxArray != (emxArray_cell_wrap_13 *)NULL) {
    if (((*pEmxArray)->data != (cell_wrap_13 *)NULL) && (*pEmxArray)
        ->canFreeData) {
      free((*pEmxArray)->data);
    }

    free((*pEmxArray)->size);
    free(*pEmxArray);
    *pEmxArray = (emxArray_cell_wrap_13 *)NULL;
  }
}

//
// Arguments    : emxArray_cell_wrap_4_64x1 *pEmxArray
// Return Type  : void
//
void emxFree_cell_wrap_4_64x1(emxArray_cell_wrap_4_64x1 *pEmxArray)
{
  int numEl;
  int i;
  numEl = pEmxArray->size[0];
  numEl *= pEmxArray->size[1];
  for (i = 0; i < numEl; i++) {
    emxFreeStruct_cell_wrap_4(&pEmxArray->data[i]);
  }
}

//
// Arguments    : emxArray_int32_T **pEmxArray
// Return Type  : void
//
void emxFree_int32_T(emxArray_int32_T **pEmxArray)
{
  if (*pEmxArray != (emxArray_int32_T *)NULL) {
    if (((*pEmxArray)->data != (int *)NULL) && (*pEmxArray)->canFreeData) {
      free((*pEmxArray)->data);
    }

    free((*pEmxArray)->size);
    free(*pEmxArray);
    *pEmxArray = (emxArray_int32_T *)NULL;
  }
}

//
// Arguments    : emxArray_real32_T **pEmxArray
// Return Type  : void
//
void emxFree_real32_T(emxArray_real32_T **pEmxArray)
{
  if (*pEmxArray != (emxArray_real32_T *)NULL) {
    if (((*pEmxArray)->data != (float *)NULL) && (*pEmxArray)->canFreeData) {
      free((*pEmxArray)->data);
    }

    free((*pEmxArray)->size);
    free(*pEmxArray);
    *pEmxArray = (emxArray_real32_T *)NULL;
  }
}

//
// Arguments    : emxArray_real_T **pEmxArray
// Return Type  : void
//
void emxFree_real_T(emxArray_real_T **pEmxArray)
{
  if (*pEmxArray != (emxArray_real_T *)NULL) {
    if (((*pEmxArray)->data != (double *)NULL) && (*pEmxArray)->canFreeData) {
      free((*pEmxArray)->data);
    }

    free((*pEmxArray)->size);
    free(*pEmxArray);
    *pEmxArray = (emxArray_real_T *)NULL;
  }
}

//
// Arguments    : cell_wrap_4 pMatrix[2]
// Return Type  : void
//
void emxInitMatrix_cell_wrap_4(cell_wrap_4 pMatrix[2])
{
  emxInitStruct_cell_wrap_4(&pMatrix[0]);
  emxInitStruct_cell_wrap_4(&pMatrix[1]);
}

//
// Arguments    : c_struct_T *pStruct
// Return Type  : void
//
void emxInitStruct_struct_T(c_struct_T *pStruct)
{
  pStruct->smartLoaderStructHistory.size[0] = 0;
  pStruct->loaderTimeTatHistoryMs.size[0] = 0;
}

//
// Arguments    : emxArray_cell_wrap_13 **pEmxArray
//                int numDimensions
// Return Type  : void
//
void emxInit_cell_wrap_13(emxArray_cell_wrap_13 **pEmxArray, int numDimensions)
{
  emxArray_cell_wrap_13 *emxArray;
  int i;
  *pEmxArray = (emxArray_cell_wrap_13 *)malloc(sizeof(emxArray_cell_wrap_13));
  emxArray = *pEmxArray;
  emxArray->data = (cell_wrap_13 *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int *)malloc(sizeof(int) * numDimensions);
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = true;
  for (i = 0; i < numDimensions; i++) {
    emxArray->size[i] = 0;
  }
}

//
// Arguments    : emxArray_cell_wrap_4_64x1 *pEmxArray
// Return Type  : void
//
void emxInit_cell_wrap_4_64x1(emxArray_cell_wrap_4_64x1 *pEmxArray)
{
  pEmxArray->size[0] = 0;
  pEmxArray->size[1] = 0;
}

//
// Arguments    : emxArray_int32_T **pEmxArray
//                int numDimensions
// Return Type  : void
//
void emxInit_int32_T(emxArray_int32_T **pEmxArray, int numDimensions)
{
  emxArray_int32_T *emxArray;
  int i;
  *pEmxArray = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
  emxArray = *pEmxArray;
  emxArray->data = (int *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int *)malloc(sizeof(int) * numDimensions);
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = true;
  for (i = 0; i < numDimensions; i++) {
    emxArray->size[i] = 0;
  }
}

//
// Arguments    : emxArray_real32_T **pEmxArray
//                int numDimensions
// Return Type  : void
//
void emxInit_real32_T(emxArray_real32_T **pEmxArray, int numDimensions)
{
  emxArray_real32_T *emxArray;
  int i;
  *pEmxArray = (emxArray_real32_T *)malloc(sizeof(emxArray_real32_T));
  emxArray = *pEmxArray;
  emxArray->data = (float *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int *)malloc(sizeof(int) * numDimensions);
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = true;
  for (i = 0; i < numDimensions; i++) {
    emxArray->size[i] = 0;
  }
}

//
// Arguments    : emxArray_real_T **pEmxArray
//                int numDimensions
// Return Type  : void
//
void emxInit_real_T(emxArray_real_T **pEmxArray, int numDimensions)
{
  emxArray_real_T *emxArray;
  int i;
  *pEmxArray = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
  emxArray = *pEmxArray;
  emxArray->data = (double *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int *)malloc(sizeof(int) * numDimensions);
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = true;
  for (i = 0; i < numDimensions; i++) {
    emxArray->size[i] = 0;
  }
}

//
// File trailer for PerceptionSmartLoader_emxutil.cpp
//
// [EOF]
//
