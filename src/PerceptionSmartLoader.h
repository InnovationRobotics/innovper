//
// File: PerceptionSmartLoader.h
//
// MATLAB Coder version            : 4.1
// C/C++ source code generated on  : 05-May-2020 11:47:28
//
#ifndef PERCEPTIONSMARTLOADER_H
#define PERCEPTIONSMARTLOADER_H

// Include Files
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "PerceptionSmartLoader_types.h"

// Variable Declarations
extern omp_nest_lock_t emlrtNestLockGlobal;

// Function Declarations
extern void PerceptionSmartLoader(PerceptionSmartLoaderStackData *SD, const
  PerceptionSmartLoaderConfigParam *configParams, const double xyz_data[], const
  int xyz_size[2], const double intensity_data[], const int intensity_size[1],
  PerceptionSmartLoaderStruct *smartLoaderStruct, float heightMap_res_data[],
  int heightMap_res_size[2]);
extern void PerceptionSmartLoader_initialize(PerceptionSmartLoaderStackData *SD);
extern void PerceptionSmartLoader_terminate(PerceptionSmartLoaderStackData *SD);

#endif

//
// File trailer for PerceptionSmartLoader.h
//
// [EOF]
//
