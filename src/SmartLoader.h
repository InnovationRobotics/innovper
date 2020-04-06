/*
 * SmartLoader.h
 *
 * Code generation for function 'SmartLoader'
 *
 */

#ifndef SMARTLOADER_H
#define SMARTLOADER_H

/* Include files */
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "omp.h"
#include "SmartLoader_types.h"

/* Variable Declarations */
extern omp_nest_lock_t emlrtNestLockGlobal;

/* Function Declarations */
extern void SmartLoader(SmartLoaderStackData *SD, const SmartLoaderConfigParam
  *configParams, const double xyz_data[], const int xyz_size[2], const double
  intensity_data[], const int intensity_size[1], SmartLoaderStruct
  *smartLoaderStruct, float heightMap_res_data[], int heightMap_res_size[2]);
extern void SmartLoader_initialize(SmartLoaderStackData *SD);
extern void SmartLoader_terminate();

#endif

/* End of code generation (SmartLoader.h) */
