/*
 * SmartLoader_types.h
 *
 * Code generation for function 'SmartLoader'
 *
 */

#ifndef SMARTLOADER_TYPES_H
#define SMARTLOADER_TYPES_H

/* Include files */
#include "rtwtypes.h"

/* Type Definitions */
typedef struct {
  unsigned long long timeTagMs;
  double planeModelParameters[4];
  boolean_T useExternalProjectionMatrix;
  double externalProjectionMatrix[12];
  double xyzLimits[6];
  double minNumPointsInPc;
  double minimumDistanceFromLoaderToPlaneMeter;
  double minPointsForReflector;
  unsigned long long maximumTimeTagDiffMs;
  double minimumIntensityReflectorValue;
  double loaderReflectorDiameterMeter;
  double reflectorMaxZaxisDistanceForOutlierMeter;
  double previousLoaderLocationToCurrentLocationMaximumDistanceMeter;
  double loaderReflectorMaxZaxisDistanceForOutlierMeter;
  double maxDistanceBetweenEachRayMeter;
  boolean_T debugMode;
} SmartLoaderConfigParam;

struct emxArray_real_T_3x32
{
  double data[96];
  int size[2];
};

struct emxArray_uint64_T_32
{
  unsigned long long data[32];
  int size[1];
};

typedef struct {
  boolean_T isInitialized;
  emxArray_real_T_3x32 loaderLocHistory;
  emxArray_uint64_T_32 loaderTimeTatHistoryMs;
} b_struct_T;

typedef struct {
  b_struct_T SmartLoaderGlobal;
  unsigned int state[625];
} SmartLoaderPersistentData;

typedef struct {
  union
  {
    struct {
      double x_data[1007998];
    } f0;

    struct {
      double Y_data[1007998];
      boolean_T logIndY_data[503999];
    } f1;

    struct {
      double unusedU3_data[504000];
    } f2;

    struct {
      double vwork_data[504000];
    } f3;

    struct {
      boolean_T b_data[504000];
    } f4;

    struct {
      double pc2_data[1512000];
      double tmp_data[1048576];
      float refmap_data[1048576];
      double x_data[504000];
      double y_data[504000];
      double z_data[504000];
      boolean_T B_data[1048576];
      boolean_T b_B_data[1048576];
    } f5;
  } u1;

  union
  {
    struct {
      double ptr_data[1008000];
      double tmp_data[1007998];
      double xs_data[504000];
      double ys_data[504000];
    } f6;

    struct {
      double x_data[504000];
    } f7;

    struct {
      double d_data[504000];
      int previdx_data[504000];
      int moved_data[504000];
      int nidx_data[504000];
    } f8;
  } u2;

  union
  {
    struct {
      double tmp_data[504000];
      double b_tmp_data[504000];
      int c_tmp_data[504000];
      boolean_T inliearsInd_data[504000];
    } f9;

    struct {
      double sampleDist_data[504001];
      double d_data[504000];
    } f10;
  } u3;

  struct {
    double D_data[1008000];
    int idx_data[504000];
  } f11;

  struct {
    double X_data[1512000];
    int idx_data[504000];
    boolean_T wasnan_data[504000];
  } f12;

  struct {
    double srcHomogenious_data[2016000];
    double pcTrans_data[1512000];
    double ptCloudSenceXyz_data[1512000];
    double ptCloudSenceReflectorsXyz_data[1512000];
    double b_ptCloudSenceReflectorsXyz_data[1512000];
    double ptCloudShovelReflectorsXyz_data[1512000];
    double kmeansDistanceMat_data[1008000];
    double kmeansIdx_data[504000];
    double tmp_data[504000];
    double b_tmp_data[504000];
    int iidx_data[504000];
    int c_tmp_data[504000];
    int d_tmp_data[504000];
    int e_tmp_data[504000];
    int f_tmp_data[504000];
    int g_tmp_data[504000];
    boolean_T temp30_data[504000];
    boolean_T h_tmp_data[504000];
  } f13;

  SmartLoaderPersistentData *pd;
} SmartLoaderStackData;

typedef struct {
  double loaderLoc[3];
  double shvelLoc[3];
  double shovelHeadingVec[3];
  double shovelHeadingVec2D[2];
  double loaderYawAngleDeg;
  boolean_T loaderYawAngleStatus;
  boolean_T status;
} SmartLoaderStruct;

struct emxArray_boolean_T
{
  boolean_T *data;
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
  double Area;
  double Centroid[2];
  double BoundingBox[4];
  double MajorAxisLength;
  double MinorAxisLength;
  double Eccentricity;
  double Orientation;
  emxArray_boolean_T *Image;
  emxArray_boolean_T *FilledImage;
  double FilledArea;
  double EulerNumber;
  double Extrema[16];
  double EquivDiameter;
  double Extent;
  emxArray_real_T *PixelIdxList;
  emxArray_real_T *PixelList;
  double Perimeter;
  emxArray_real_T *PixelValues;
  double WeightedCentroid[2];
  double MeanIntensity;
  double MinIntensity;
  double MaxIntensity;
  emxArray_real_T *SubarrayIdx;
  double SubarrayIdxLengths[2];
} d_struct_T;

typedef struct {
  d_struct_T *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
} b_emxArray_struct_T;

typedef struct {
  double Orientation;
  double Area;
  double MajorAxisLength;
  double MinorAxisLength;
  double Centroid[2];
} c_struct_T;

struct s3U6Uw2X9E9EZF94blNA9KG_tag
{
  emxArray_real_T *f1;
};

typedef s3U6Uw2X9E9EZF94blNA9KG_tag cell_wrap_0;
struct emxArray_s3U6Uw2X9E9EZF94blNA9KG_tag_64x1
{
  cell_wrap_0 data[64];
  int size[2];
};

typedef emxArray_s3U6Uw2X9E9EZF94blNA9KG_tag_64x1 emxArray_cell_wrap_0_64x1;
struct emxArray_int16_T
{
  short *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

struct emxArray_int32_T
{
  int *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

struct emxArray_real32_T
{
  float *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

typedef struct {
  c_struct_T *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
} emxArray_struct_T;

#endif

/* End of code generation (SmartLoader_types.h) */
