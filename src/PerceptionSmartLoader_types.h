//
// File: PerceptionSmartLoader_types.h
//
// MATLAB Coder version            : 4.1
// C/C++ source code generated on  : 05-May-2020 11:47:28
//
#ifndef PERCEPTIONSMARTLOADER_TYPES_H
#define PERCEPTIONSMARTLOADER_TYPES_H

// Include Files
#include "rtwtypes.h"

// Type Definitions
typedef struct {
  unsigned long timeTagMs;
  double planeModelParameters[4];
  double maxDistanceToPlaneMeter;
  double pcAlignmentProjMat[12];
  double xyzLimits[6];
  double minNumPointsInPc;
  double minimumDistanceFromLoaderToPlaneMeter;
  double minPointsForReflector;
  unsigned long maximumTimeTagDiffMs;
  double minimumIntensityReflectorValue;
  double loaderReflectorDiameterMeter;
  double loaderWhiteHatMeter;
  double loaderCenterToBackwardPointMeter;
  double locationsBiasMeter;
  double loaderWidthMeter;
  double loaderHeightMeter;
  double reflectorMaxZaxisDistanceForOutlierMeter;
  double previousLoaderLocationToCurrentLocationMaximumDistanceMeter;
  double loaderReflectorMaxZaxisDistanceForOutlierMeter;
  double maxDistanceBetweenEachRayMeter;
  double heightMapResolutionMeterToPixel;
  double maxDistanceFromThePlaneForLoaderYawCalculation;
  double yawEstimationMinPercentageOfPointsInLoaderBody;
  double yawEstimationMinNumPointsInLoaderBody;
  double loaderYawAngleSmoothWeight;
  double loaderToShovelYawAngleSmoothWeight;
  boolean_T debugMode;
} PerceptionSmartLoaderConfigParam;

enum PerceptionSmartLoaderReturnValue
{
  PerceptionSmartLoaderReturnValue_eSuccess = 0,// Default value
  PerceptionSmartLoaderReturnValue_eFailed = -1,
  PerceptionSmartLoaderReturnValue_eFailedNotEnoughPoints = -2,
  PerceptionSmartLoaderReturnValue_eFailedNotEnoughReflectorPoints = -3,
  PerceptionSmartLoaderReturnValue_eFailedLoaderLocation = -4
};

typedef struct {
  boolean_T heightMapStatus;
  double loaderLoc[3];
  boolean_T loaderLocStatus;
  double shovelLoc[3];
  boolean_T shovelLocStatus;
  double loaderYawAngleDeg;
  double loaderYawAngleDegSmooth;
  boolean_T loaderYawAngleStatus;
  double loaderToShovelYawAngleDeg;
  double loaderToShovelYawAngleDegSmooth;
  boolean_T loaderToShovelYawAngleDegStatus;
  PerceptionSmartLoaderReturnValue status;
} b_struct_T;

typedef struct {
  b_struct_T data[32];
  int size[1];
} emxArray_struct_T_32;

struct emxArray_uint64_T_32
{
  unsigned long data[32];
  int size[1];
};

typedef struct {
  boolean_T isInitialized;
  emxArray_struct_T_32 smartLoaderStructHistory;
  emxArray_uint64_T_32 loaderTimeTatHistoryMs;
} c_struct_T;

struct emxArray_real32_T
{
  float *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

struct emxArray_real_T
{
  double *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

typedef struct {
  c_struct_T SmartLoaderGlobal;
  emxArray_real32_T *heightMap_resPersistent;
  boolean_T heightMap_resPersistent_not_empty;
  emxArray_real_T *allImgInds;
  unsigned int state[625];
} PerceptionSmartLoaderPersistentData;

struct emxArray_boolean_T_32767
{
  boolean_T data[32767];
  int size[1];
};

struct emxArray_real32_T_2x504000
{
  float data[1008000];
  int size[2];
};

struct emxArray_real_T_2x32767
{
  double data[65534];
  int size[2];
};

struct emxArray_real_T_32767
{
  double data[32767];
  int size[1];
};

struct emxArray_real_T_504000x1
{
  double data[504000];
  int size[2];
};

struct emxArray_uint32_T_504000
{
  unsigned int data[504000];
  int size[1];
};

struct shvxKQt6KTNMHnha10Kc8WD_tag
{
  emxArray_real32_T_2x504000 X;
  char Distance[9];
  double BucketSize;
  emxArray_real_T_504000x1 wasnanIdx;
  int numNodes;
  emxArray_real_T_32767 cutDim;
  emxArray_real_T_32767 cutVal;
  emxArray_real_T_2x32767 lowerBounds;
  emxArray_real_T_2x32767 upperBounds;
  emxArray_real_T_32767 leftChild;
  emxArray_real_T_32767 rightChild;
  emxArray_boolean_T_32767 leafNode;
  double nx_nonan;
  emxArray_uint32_T_504000 idxAll;
  emxArray_real_T_32767 idxDim;
};

typedef shvxKQt6KTNMHnha10Kc8WD_tag struct_T;
typedef struct {
  union
  {
    struct {
      float x_data[1008000];
    } f0;

    struct {
      double tmp_data[1512000];
      float b_tmp_data[1512000];
      float temp1_data[504000];
    } f1;

    struct {
      float x_data[1007998];
    } f2;

    struct {
      float Y_data[1007998];
    } f3;

    struct {
      float unusedU3_data[504000];
    } f4;

    struct {
      boolean_T b_data[504000];
    } f5;

    struct {
      float ycol_data[504000];
    } f6;

    struct {
      int iwork_data[504000];
    } f7;

    struct {
      float ycol_data[504000];
    } f8;

    struct {
      int iwork_data[504000];
    } f9;

    struct {
      double srcHomogenious_data[2016000];
      double dst_data[1512000];
      int tmp_data[504000];
      int b_tmp_data[504000];
      boolean_T indices_data[504000];
    } f10;
  } u1;

  union
  {
    struct {
      float ptr_data[1008000];
      float tmp_data[1007998];
      float xs_data[504000];
      float ys_data[504000];
    } f11;

    struct {
      float x_data[504000];
    } f12;

    struct {
      int previdx_data[504000];
      int moved_data[504000];
      float d_data[504000];
      int nidx_data[504000];
    } f13;

    struct {
      int idx_data[504000];
    } f14;
  } u2;

  union
  {
    struct {
      float tmp_data[504000];
      float b_tmp_data[504000];
      int c_tmp_data[504000];
      boolean_T inliearsInd_data[504000];
    } f15;

    struct {
      float sampleDist_data[504001];
      float d_data[504000];
    } f16;
  } u3;

  union
  {
    struct {
      float D_data[1008000];
      int idx_data[504000];
    } f17;
  } u4;

  union
  {
    struct {
      int idx_data[504000];
    } f18;

    struct {
      float Y_data[8388608];
      float X_data[1008000];
      int noNanCol_data[504000];
    } f19;
  } u5;

  union
  {
    struct {
      float ptCloudSenceReflectorsXyz_data[1512000];
      float b_ptCloudSenceReflectorsXyz_data[1512000];
      float pcFirstXyz_data[1512000];
      float pcSecondXyz_data[1512000];
      float ptCloudShovelReflectorsXyz_data[1512000];
      double kmeansIdx_data[504000];
      float kmeansDistanceMat_data[1008000];
      float c_ptCloudSenceReflectorsXyz_data[1008000];
      int iidx_data[504000];
      float tmp_data[504000];
      int b_tmp_data[504000];
      float c_tmp_data[504000];
      float x_data[504000];
      int d_tmp_data[504000];
      int e_tmp_data[504000];
      int f_tmp_data[504000];
      int g_tmp_data[504000];
      int h_tmp_data[504000];
      boolean_T ptCloudSenceReflectorsInd_data[504000];
      boolean_T isInsidePolygon_data[504000];
      boolean_T b_ptCloudSenceReflectorsInd_data[504000];
    } f20;

    struct {
      struct_T expl_temp;
      int notnan_data[504000];
      unsigned int tempIdx_data[504000];
      float x_data[504000];
      int iidx_data[504000];
      int tmp_data[504000];
      int b_tmp_data[252000];
      double lowerBoundsTemp_data[65534];
      double upperBoundsTemp_data[65534];
      double cgstruct_lowerBounds_data[65534];
      double cgstruct_upperBounds_data[65534];
      double cutValTemp_data[32767];
      double leftChildTemp_data[32767];
      double rightChildTemp_data[32767];
      double cgstruct_cutVal_data[32767];
      double cgstruct_leftChild_data[32767];
      int cutDimTemp_data[32767];
      short c_tmp_data[32767];
    } f21;
  } u6;

  struct {
    int tmp_data[4194304];
    float xyzInsideImg_data[1512000];
    float xyzSortedUnique_data[1512000];
    boolean_T missingPixelsLogical_data[4194304];
    float xyRounded_data[1008000];
    float b_xyRounded_data[1008000];
    double idx_data[504000];
    int imgLinearInd_data[504000];
    float b_tmp_data[504000];
    int c_tmp_data[504000];
    boolean_T d_tmp_data[504000];
    boolean_T e_tmp_data[504000];
  } f22;

  struct {
    float ptCloudSenceXyz_data[1512000];
    float ptCloudSenceIntensity_data[504000];
  } f23;

  PerceptionSmartLoaderPersistentData *pd;
} PerceptionSmartLoaderStackData;

typedef struct {
  boolean_T heightMapStatus;
  double loaderLoc[3];
  boolean_T loaderLocStatus;
  double shovelLoc[3];
  boolean_T shovelLocStatus;
  double loaderYawAngleDeg;
  double loaderYawAngleDegSmooth;
  boolean_T loaderYawAngleStatus;
  double loaderToShovelYawAngleDeg;
  double loaderToShovelYawAngleDegSmooth;
  boolean_T loaderToShovelYawAngleDegStatus;
  PerceptionSmartLoaderReturnValue status;
} PerceptionSmartLoaderStruct;

struct emxArray_real32_T_504000
{
  float data[504000];
  int size[1];
};

struct sYMfczL2DZgOzhVqAZTDfYC_tag
{
  emxArray_real32_T_504000 D;
  emxArray_uint32_T_504000 b_I;
};

typedef sYMfczL2DZgOzhVqAZTDfYC_tag d_struct_T;
typedef struct {
  union
  {
    struct {
      int idx_data[504000];
      float x_data[504000];
      int iwork_data[504000];
      float xwork_data[504000];
    } f0;

    struct {
      double tmp_data[504000];
      int b_tmp_data[504000];
      double idxDim_data[32766];
    } f1;

    struct {
      double x_data[32767];
    } f2;
  } u1;

  union
  {
    struct {
      float vwork_data[504000];
      int iidx_data[504000];
    } f3;
  } u2;

  union
  {
    struct {
      float X_data[1008000];
      float diffAllDim_data[1008000];
      float tmp_data[504001];
      unsigned int b_tmp_data[504001];
      float aDistOut_data[504000];
      float distInP_data[504000];
      int iidx_data[504000];
      unsigned int node_idx_start_data[504000];
      int c_tmp_data[503999];
    } f4;
  } u3;

  union
  {
    struct {
      d_struct_T r1;
      unsigned int node_idx_this_data[504000];
      double nodeStack_data[32768];
      double obj_rightChild_data[32768];
      double tmp_data[32767];
    } f5;
  } u4;

  union
  {
    struct {
      int tmp_data[504000];
    } f6;
  } u5;
} PerceptionSmartLoaderTLS;

struct sR9nrfT5CUH1cTOx6GipQw_tag
{
  emxArray_real_T_504000x1 f1;
};

typedef sR9nrfT5CUH1cTOx6GipQw_tag cell_wrap_13;
struct sfwI8zOKrNsirWOkLmXxW2D_tag
{
  emxArray_real32_T *f1;
};

typedef sfwI8zOKrNsirWOkLmXxW2D_tag cell_wrap_4;
struct emxArray_sR9nrfT5CUH1cTOx6GipQw_tag
{
  cell_wrap_13 *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

typedef emxArray_sR9nrfT5CUH1cTOx6GipQw_tag emxArray_cell_wrap_13;
struct emxArray_sfwI8zOKrNsirWOkLmXxW2D_tag_64x1
{
  cell_wrap_4 data[64];
  int size[2];
};

typedef emxArray_sfwI8zOKrNsirWOkLmXxW2D_tag_64x1 emxArray_cell_wrap_4_64x1;
struct emxArray_int32_T
{
  int *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

#endif

//
// File trailer for PerceptionSmartLoader_types.h
//
// [EOF]
//
