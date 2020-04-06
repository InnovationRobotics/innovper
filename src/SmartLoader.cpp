/*
 * SmartLoader.cpp
 *
 * Code generation for function 'SmartLoader'
 *
 */

/* Include files */
#include <cmath>
#include <string.h>
#include "rt_nonfinite.h"
#include <math.h>
#include "SmartLoader.h"
#include "SmartLoader_emxutil.h"
#include "libmwmorphop_binary_tbb.h"
#include "libmwmorphop_flat_tbb.h"
#include "libmwmorphop_ocv.h"

/* Type Definitions */
typedef struct {
  boolean_T Area;
  boolean_T Centroid;
  boolean_T BoundingBox;
  boolean_T MajorAxisLength;
  boolean_T MinorAxisLength;
  boolean_T Eccentricity;
  boolean_T Orientation;
  boolean_T Image;
  boolean_T FilledImage;
  boolean_T FilledArea;
  boolean_T EulerNumber;
  boolean_T Extrema;
  boolean_T EquivDiameter;
  boolean_T Extent;
  boolean_T PixelIdxList;
  boolean_T PixelList;
  boolean_T Perimeter;
  boolean_T PixelValues;
  boolean_T WeightedCentroid;
  boolean_T MeanIntensity;
  boolean_T MinIntensity;
  boolean_T MaxIntensity;
  boolean_T SubarrayIdx;
} struct_T;

/* Variable Definitions */
omp_nest_lock_t emlrtNestLockGlobal;

/* Function Declarations */
static void CalcPlaneToPointDistance(const double planeModelParameters[4], const
  double srcPoints[3], double *distanceFromPointToPlane, boolean_T
  *isPointAbovePlane);
static void ClusterPoints2D(SmartLoaderStackData *SD, double ptr_data[], int
  ptr_size[2], double maxDistance, cell_wrap_0 clustersXs_data[], int
  clustersXs_size[2], cell_wrap_0 clustersYs_data[], int clustersYs_size[2]);
static void ComputeArea(b_emxArray_struct_T *stats, struct_T
  *statsAlreadyComputed);
static void ComputeCentroid(const double imageSize[2], b_emxArray_struct_T
  *stats, struct_T *statsAlreadyComputed);
static void ComputeEllipseParams(const double imageSize[2], b_emxArray_struct_T *
  stats, struct_T *statsAlreadyComputed);
static void ComputePixelIdxList(const double L_data[], const int L_size[2],
  double numObjs, b_emxArray_struct_T *stats, struct_T *statsAlreadyComputed);
static void ComputePixelList(const double imageSize[2], b_emxArray_struct_T
  *stats, struct_T *statsAlreadyComputed);
static void CreateHeightMap(SmartLoaderStackData *SD, SmartLoaderStruct
  *smartLoaderStruct, const double xyz_data[], const int xyz_size[2], float
  heightMap_res_data[], int heightMap_res_size[2]);
static void FilterPointCloudAccordingToZdifferences(SmartLoaderStackData *SD,
  const double pc_data[], const int pc_size[2], double diffThreshold, double
  pcFiltered_data[], int pcFiltered_size[2]);
static void SmartLoaderGlobalInit(SmartLoaderStackData *SD);
static boolean_T all(const boolean_T x[2]);
static boolean_T any(const boolean_T x[2]);
static void b_abs(const double x_data[], const int x_size[1], double y_data[],
                  int y_size[1]);
static int b_bsearch(const double x_data[], const int x_size[1], double xi);
static void b_distfun(double D_data[], const double X_data[], const int X_size[2],
                      const double C[6], const int crows[2], int ncrows);
static void b_gcentroids(double C[6], int counts[2], const double X_data[],
  const int X_size[2], const int idx_data[], int clusters);
static void b_imdilate(const emxArray_real32_T *A, emxArray_real32_T *B);
static boolean_T b_isfinite(double x);
static double b_norm(const double x[3]);
static void b_nullAssignment(SmartLoaderStackData *SD, double x_data[], int
  x_size[2], const int idx_data[]);
static double b_rand(SmartLoaderStackData *SD);
static void b_repmat(const double a[2], double varargin_1, double b_data[], int
                     b_size[2]);
static void b_round(double x_data[], int x_size[1]);
static void b_sort(SmartLoaderStackData *SD, double x_data[], int x_size[1], int
                   idx_data[], int idx_size[1]);
static void b_sqrt(double x_data[], int x_size[1]);
static double b_sum(const emxArray_real_T *x);
static void batchUpdate(SmartLoaderStackData *SD, const double X_data[], const
  int X_size[2], int idx_data[], int idx_size[1], double C[6], double D_data[],
  int D_size[2], int counts[2], boolean_T *converged, int *iter);
static void bwlabel(const boolean_T varargin_1_data[], const int
                    varargin_1_size[2], double L_data[], int L_size[2]);
static void c_imdilate(const emxArray_real32_T *A, emxArray_real32_T *B);
static void c_sum(const double x_data[], const int x_size[2], double y_data[],
                  int y_size[1]);
static int countEmpty(int empties[2], const int counts[2], const int changed[2],
                      int nchanged);
static void d_imdilate(const emxArray_real32_T *A, emxArray_real32_T *B);
static void distfun(double D_data[], const double X_data[], const int X_size[2],
                    const double C[6], int crows);
static int div_s32(int numerator, int denominator);
static void eml_rand_mt19937ar_stateful_init(SmartLoaderStackData *SD);
static int findchanged(SmartLoaderStackData *SD, int changed[2], const int
  idx_data[], const int previdx_data[], const int moved_data[], const int
  moved_size[1], int nmoved);
static void gcentroids(double C[6], int counts[2], const double X_data[], const
  int X_size[2], const int idx_data[], const int clusters[2], int nclusters);
static boolean_T ifWhileCond(const boolean_T x_data[]);
static void imdilate(const emxArray_real32_T *A, emxArray_real32_T *B);
static void initializeStatsStruct(double numObjs, b_emxArray_struct_T *stats,
  struct_T *statsAlreadyComputed);
static void inv(const double x[4], double y[4]);
static void kmeans(SmartLoaderStackData *SD, double X_data[], int X_size[2],
                   double idxbest_data[], int idxbest_size[1], double Cbest[6],
                   double varargout_1[2], double varargout_2_data[], int
                   varargout_2_size[2]);
static void local_kmeans(SmartLoaderStackData *SD, const double X_data[], const
  int X_size[2], int idxbest_data[], int idxbest_size[1], double Cbest[6],
  double varargout_1[2], double varargout_2_data[], int varargout_2_size[2]);
static void loopBody(SmartLoaderStackData *SD, const double X_data[], const int
                     X_size[2], double *totsumD, int idx_data[], int idx_size[1],
                     double C[6], double sumD[2], double D_data[], int D_size[2]);
static void mean(const double x_data[], const int x_size[2], double y[3]);
static void med3(double v_data[], int nv, int ia, int ib);
static double median(SmartLoaderStackData *SD, const double x_data[], const int
                     x_size[1]);
static void medmed(double v_data[], int nv, int ia);
static void merge(emxArray_int32_T *idx, emxArray_real_T *x, int offset, int np,
                  int nq, emxArray_int32_T *iwork, emxArray_real_T *xwork);
static void merge_block(emxArray_int32_T *idx, emxArray_real_T *x, int offset,
  int n, int preSortLevel, emxArray_int32_T *iwork, emxArray_real_T *xwork);
static double minOrMaxRealFloatVector(const emxArray_real_T *x);
static void mindim2(const double D_data[], const int D_size[2], double d_data[],
                    int d_size[1], int idx_data[], int idx_size[1]);
static double nCk(double n, double k);
static void nchoosek(const double x_data[], const int x_size[2], emxArray_real_T
                     *y);
static double nestedIter(const emxArray_real_T *x, int vlen);
static void nullAssignment(const cell_wrap_0 x_data[], const boolean_T idx_data[],
  cell_wrap_0 b_x_data[], int x_size[2]);
static void pdist(const emxArray_real_T *Xin, emxArray_real_T *Y);
static void pdist2(SmartLoaderStackData *SD, const emxArray_real_T *Xin, const
                   double Yin_data[], const int Yin_size[2], emxArray_real_T *D);
static int pivot(double v_data[], int *ip, int ia, int ib);
static void populateOutputStatsStructure(emxArray_struct_T *outstats, const
  b_emxArray_struct_T *stats);
static void power(const emxArray_real_T *a, emxArray_real_T *y);
static void quickselect(double v_data[], int n, int vlen, double *vn, int
  *nfirst, int *nlast);
static void regionprops(const double varargin_1_data[], const int
  varargin_1_size[2], emxArray_struct_T *outstats);
static void repmat(int varargin_1, double b_data[], int b_size[1]);
static double rt_powd_snf(double u0, double u1);
static double rt_roundd_snf(double u);
static void simpleRandperm(SmartLoaderStackData *SD, int n, int idx_data[], int
  idx_size[1]);
static void sort(emxArray_real_T *x);
static void sortIdx(emxArray_real_T *x, emxArray_int32_T *idx);
static void squareform(const emxArray_real_T *Y, emxArray_real_T *Z);
static double sum(const emxArray_real_T *x);
static int thirdOfFive(const double v_data[], int ia, int ib);
static void vecnorm(const double x[6], double y[2]);
static double vmedian(SmartLoaderStackData *SD, double v_data[], int v_size[1],
                      int n);

/* Function Definitions */

/*
 * function [distanceFromPointToPlane, isPointAbovePlane] = CalcPlaneToPointDistance(planeModelParameters, srcPoints)
 */
static void CalcPlaneToPointDistance(const double planeModelParameters[4], const
  double srcPoints[3], double *distanceFromPointToPlane, boolean_T
  *isPointAbovePlane)
{
  double temp1;

  /*  The function calculate the plane to point or a vector of points distance */
  /*  Input arguments:  */
  /*  model - plane model - type of XXX */
  /*  srcPoints - matrix of Nx3 of 3d points */
  /*  Output arguments:  */
  /*  distanceFromPointToPlane - distance for each point, size of Nx1 */
  /*  isPointAbovePlane - boolean - represet is the current point is above or below the plane - above or below is related to the normal vector  */
  /*  of the plane */
  /* assert(isa(planeModelParameters, 'double')); */
  /* 'CalcPlaneToPointDistance:14' assert(size(planeModelParameters,1) == 1); */
  /* 'CalcPlaneToPointDistance:15' assert(size(planeModelParameters,2) == 4); */
  /* 'CalcPlaneToPointDistance:18' modelParametersRepmat = repmat(planeModelParameters(1:3), size(srcPoints,1), 1); */
  /* 'CalcPlaneToPointDistance:20' temp1 = sum(srcPoints .* modelParametersRepmat, 2) + planeModelParameters(4); */
  temp1 = ((srcPoints[0] * planeModelParameters[0] + srcPoints[1] *
            planeModelParameters[1]) + srcPoints[2] * planeModelParameters[2]) +
    planeModelParameters[3];

  /* 'CalcPlaneToPointDistance:22' distanceFromPointToPlane = abs(temp1) / norm(planeModelParameters(1:3)); */
  *distanceFromPointToPlane = std::abs(temp1) / b_norm(*(double (*)[3])&
    planeModelParameters[0]);

  /* 'CalcPlaneToPointDistance:24' isPointAbovePlane = temp1 > 0; */
  *isPointAbovePlane = (temp1 > 0.0);
}

/*
 * function [clustersXs, clustersYs] = ClusterPoints2D(ptr, maxDistance)
 */
static void ClusterPoints2D(SmartLoaderStackData *SD, double ptr_data[], int
  ptr_size[2], double maxDistance, cell_wrap_0 clustersXs_data[], int
  clustersXs_size[2], cell_wrap_0 clustersYs_data[], int clustersYs_size[2])
{
  int i24;
  int input_sizes_idx_0;
  int loop_ub;
  double cellArrayIndex;
  emxArray_real_T *distanceMat;
  cell_wrap_0 reshapes[2];
  emxArray_real_T *ex;
  emxArray_int32_T *idx;
  cell_wrap_0 b_reshapes[2];
  cell_wrap_0 c_reshapes[2];
  emxArray_real_T *d_reshapes;
  int m;
  boolean_T empty_non_axis_sizes;
  signed char input_sizes_idx_1;
  int i25;
  int i26;
  int n;
  int i27;
  int i28;
  unsigned int distanceMat_idx_0;
  double b;
  double minVal_data[1];
  int minInd_data[1];
  boolean_T b_minVal_data[1];
  boolean_T exitg1;
  int tmp_size[2];
  int b_minInd_data[1];

  /*  Cell {end+1} command only works with vectors, not matrices, that's why we changed the output to two different cells */
  /*  Must set upper bound for cell array compilation */
  /* coder.varsize('clustersYs', 'clustersXs', [1 M], [0 1]); */
  /* 'ClusterPoints2D:10' percisionMode = 'double'; */
  /* 'ClusterPoints2D:12' clustersXs = cell(1,SmartLoaderCompilationConstants.MaxNumClusters); */
  /* 'ClusterPoints2D:13' clustersXs = coder.nullcopy(clustersXs); */
  i24 = clustersXs_size[0] * clustersXs_size[1];
  clustersXs_size[1] = 64;
  clustersXs_size[0] = 1;
  emxEnsureCapacity_cell_wrap_0(clustersXs_data, clustersXs_size, i24);

  /* 'ClusterPoints2D:14' for i = 1:SmartLoaderCompilationConstants.MaxNumClusters */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[1].f1->size[1] = 0;
  clustersXs_data[1].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[2].f1->size[1] = 0;
  clustersXs_data[2].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[3].f1->size[1] = 0;
  clustersXs_data[3].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[4].f1->size[1] = 0;
  clustersXs_data[4].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[5].f1->size[1] = 0;
  clustersXs_data[5].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[6].f1->size[1] = 0;
  clustersXs_data[6].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[7].f1->size[1] = 0;
  clustersXs_data[7].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[8].f1->size[1] = 0;
  clustersXs_data[8].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[9].f1->size[1] = 0;
  clustersXs_data[9].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[10].f1->size[1] = 0;
  clustersXs_data[10].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[11].f1->size[1] = 0;
  clustersXs_data[11].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[12].f1->size[1] = 0;
  clustersXs_data[12].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[13].f1->size[1] = 0;
  clustersXs_data[13].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[14].f1->size[1] = 0;
  clustersXs_data[14].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[15].f1->size[1] = 0;
  clustersXs_data[15].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[16].f1->size[1] = 0;
  clustersXs_data[16].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[17].f1->size[1] = 0;
  clustersXs_data[17].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[18].f1->size[1] = 0;
  clustersXs_data[18].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[19].f1->size[1] = 0;
  clustersXs_data[19].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[20].f1->size[1] = 0;
  clustersXs_data[20].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[21].f1->size[1] = 0;
  clustersXs_data[21].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[22].f1->size[1] = 0;
  clustersXs_data[22].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[23].f1->size[1] = 0;
  clustersXs_data[23].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[24].f1->size[1] = 0;
  clustersXs_data[24].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[25].f1->size[1] = 0;
  clustersXs_data[25].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[26].f1->size[1] = 0;
  clustersXs_data[26].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[27].f1->size[1] = 0;
  clustersXs_data[27].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[28].f1->size[1] = 0;
  clustersXs_data[28].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[29].f1->size[1] = 0;
  clustersXs_data[29].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[30].f1->size[1] = 0;
  clustersXs_data[30].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[31].f1->size[1] = 0;
  clustersXs_data[31].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[32].f1->size[1] = 0;
  clustersXs_data[32].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[33].f1->size[1] = 0;
  clustersXs_data[33].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[34].f1->size[1] = 0;
  clustersXs_data[34].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[35].f1->size[1] = 0;
  clustersXs_data[35].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[36].f1->size[1] = 0;
  clustersXs_data[36].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[37].f1->size[1] = 0;
  clustersXs_data[37].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[38].f1->size[1] = 0;
  clustersXs_data[38].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[39].f1->size[1] = 0;
  clustersXs_data[39].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[40].f1->size[1] = 0;
  clustersXs_data[40].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[41].f1->size[1] = 0;
  clustersXs_data[41].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[42].f1->size[1] = 0;
  clustersXs_data[42].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[43].f1->size[1] = 0;
  clustersXs_data[43].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[44].f1->size[1] = 0;
  clustersXs_data[44].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[45].f1->size[1] = 0;
  clustersXs_data[45].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[46].f1->size[1] = 0;
  clustersXs_data[46].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[47].f1->size[1] = 0;
  clustersXs_data[47].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[48].f1->size[1] = 0;
  clustersXs_data[48].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[49].f1->size[1] = 0;
  clustersXs_data[49].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[50].f1->size[1] = 0;
  clustersXs_data[50].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[51].f1->size[1] = 0;
  clustersXs_data[51].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[52].f1->size[1] = 0;
  clustersXs_data[52].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[53].f1->size[1] = 0;
  clustersXs_data[53].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[54].f1->size[1] = 0;
  clustersXs_data[54].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[55].f1->size[1] = 0;
  clustersXs_data[55].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[56].f1->size[1] = 0;
  clustersXs_data[56].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[57].f1->size[1] = 0;
  clustersXs_data[57].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[58].f1->size[1] = 0;
  clustersXs_data[58].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[59].f1->size[1] = 0;
  clustersXs_data[59].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[60].f1->size[1] = 0;
  clustersXs_data[60].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[61].f1->size[1] = 0;
  clustersXs_data[61].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[62].f1->size[1] = 0;
  clustersXs_data[62].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:15' clustersXs{i} = zeros(0,0,percisionMode); */
  clustersXs_data[63].f1->size[1] = 0;
  clustersXs_data[63].f1->size[0] = 0;

  /*  clustersXs{i} = single(1); */
  /* 'ClusterPoints2D:18' clustersYs = cell(1,SmartLoaderCompilationConstants.MaxNumClusters); */
  /* 'ClusterPoints2D:19' clustersYs = coder.nullcopy(clustersYs); */
  i24 = clustersYs_size[0] * clustersYs_size[1];
  clustersYs_size[1] = 64;
  clustersYs_size[0] = 1;
  emxEnsureCapacity_cell_wrap_0(clustersYs_data, clustersYs_size, i24);

  /* 'ClusterPoints2D:20' for i = 1:SmartLoaderCompilationConstants.MaxNumClusters */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[1].f1->size[1] = 0;
  clustersYs_data[1].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[2].f1->size[1] = 0;
  clustersYs_data[2].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[3].f1->size[1] = 0;
  clustersYs_data[3].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[4].f1->size[1] = 0;
  clustersYs_data[4].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[5].f1->size[1] = 0;
  clustersYs_data[5].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[6].f1->size[1] = 0;
  clustersYs_data[6].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[7].f1->size[1] = 0;
  clustersYs_data[7].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[8].f1->size[1] = 0;
  clustersYs_data[8].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[9].f1->size[1] = 0;
  clustersYs_data[9].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[10].f1->size[1] = 0;
  clustersYs_data[10].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[11].f1->size[1] = 0;
  clustersYs_data[11].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[12].f1->size[1] = 0;
  clustersYs_data[12].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[13].f1->size[1] = 0;
  clustersYs_data[13].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[14].f1->size[1] = 0;
  clustersYs_data[14].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[15].f1->size[1] = 0;
  clustersYs_data[15].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[16].f1->size[1] = 0;
  clustersYs_data[16].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[17].f1->size[1] = 0;
  clustersYs_data[17].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[18].f1->size[1] = 0;
  clustersYs_data[18].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[19].f1->size[1] = 0;
  clustersYs_data[19].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[20].f1->size[1] = 0;
  clustersYs_data[20].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[21].f1->size[1] = 0;
  clustersYs_data[21].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[22].f1->size[1] = 0;
  clustersYs_data[22].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[23].f1->size[1] = 0;
  clustersYs_data[23].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[24].f1->size[1] = 0;
  clustersYs_data[24].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[25].f1->size[1] = 0;
  clustersYs_data[25].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[26].f1->size[1] = 0;
  clustersYs_data[26].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[27].f1->size[1] = 0;
  clustersYs_data[27].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[28].f1->size[1] = 0;
  clustersYs_data[28].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[29].f1->size[1] = 0;
  clustersYs_data[29].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[30].f1->size[1] = 0;
  clustersYs_data[30].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[31].f1->size[1] = 0;
  clustersYs_data[31].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[32].f1->size[1] = 0;
  clustersYs_data[32].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[33].f1->size[1] = 0;
  clustersYs_data[33].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[34].f1->size[1] = 0;
  clustersYs_data[34].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[35].f1->size[1] = 0;
  clustersYs_data[35].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[36].f1->size[1] = 0;
  clustersYs_data[36].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[37].f1->size[1] = 0;
  clustersYs_data[37].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[38].f1->size[1] = 0;
  clustersYs_data[38].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[39].f1->size[1] = 0;
  clustersYs_data[39].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[40].f1->size[1] = 0;
  clustersYs_data[40].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[41].f1->size[1] = 0;
  clustersYs_data[41].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[42].f1->size[1] = 0;
  clustersYs_data[42].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[43].f1->size[1] = 0;
  clustersYs_data[43].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[44].f1->size[1] = 0;
  clustersYs_data[44].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[45].f1->size[1] = 0;
  clustersYs_data[45].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[46].f1->size[1] = 0;
  clustersYs_data[46].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[47].f1->size[1] = 0;
  clustersYs_data[47].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[48].f1->size[1] = 0;
  clustersYs_data[48].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[49].f1->size[1] = 0;
  clustersYs_data[49].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[50].f1->size[1] = 0;
  clustersYs_data[50].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[51].f1->size[1] = 0;
  clustersYs_data[51].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[52].f1->size[1] = 0;
  clustersYs_data[52].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[53].f1->size[1] = 0;
  clustersYs_data[53].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[54].f1->size[1] = 0;
  clustersYs_data[54].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[55].f1->size[1] = 0;
  clustersYs_data[55].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[56].f1->size[1] = 0;
  clustersYs_data[56].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[57].f1->size[1] = 0;
  clustersYs_data[57].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[58].f1->size[1] = 0;
  clustersYs_data[58].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[59].f1->size[1] = 0;
  clustersYs_data[59].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[60].f1->size[1] = 0;
  clustersYs_data[60].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[61].f1->size[1] = 0;
  clustersYs_data[61].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[62].f1->size[1] = 0;
  clustersYs_data[62].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /* 'ClusterPoints2D:21' clustersYs{i} = zeros(0,0,percisionMode); */
  clustersYs_data[63].f1->size[1] = 0;
  clustersYs_data[63].f1->size[0] = 0;

  /*  clustersYs{i} = single(1); */
  /*  hold the first cluster - this is also helps for matlab coder determine the cell array type */
  /* 'ClusterPoints2D:26' coder.varsize('ys','xs','ys2','xs2', [SmartLoaderCompilationConstants.MaxPointCloudSize 1], [1 0]); */
  /*  coder.varsize('ptr', [SmartLoaderCompilationConstants.MaxPointCloudSize 2], [1 0]); */
  /* 'ClusterPoints2D:28' xs = ptr(end,1); */
  input_sizes_idx_0 = (ptr_size[0] - 1) << 1;
  SD->u2.f6.xs_data[0] = ptr_data[input_sizes_idx_0];

  /* 'ClusterPoints2D:29' ys = ptr(end,2); */
  SD->u2.f6.ys_data[0] = ptr_data[1 + input_sizes_idx_0];

  /*  remove the last point */
  /* 'ClusterPoints2D:32' ptr = ptr(1:(end-1),:); */
  if (1 > ptr_size[0] - 1) {
    loop_ub = 0;
  } else {
    loop_ub = ptr_size[0] - 1;
  }

  for (i24 = 0; i24 < loop_ub; i24++) {
    input_sizes_idx_0 = i24 << 1;
    SD->u2.f6.ptr_data[input_sizes_idx_0] = ptr_data[input_sizes_idx_0];
    input_sizes_idx_0++;
    SD->u2.f6.ptr_data[input_sizes_idx_0] = ptr_data[input_sizes_idx_0];
  }

  ptr_size[1] = 2;
  ptr_size[0] = loop_ub;
  loop_ub <<= 1;
  if (0 <= loop_ub - 1) {
    memcpy(&ptr_data[0], &SD->u2.f6.ptr_data[0], (unsigned int)(loop_ub * (int)
            sizeof(double)));
  }

  /* 'ClusterPoints2D:34' clustersXs{1} = xs; */
  i24 = clustersXs_data[0].f1->size[0] * clustersXs_data[0].f1->size[1];
  clustersXs_data[0].f1->size[1] = 1;
  clustersXs_data[0].f1->size[0] = 1;
  emxEnsureCapacity_real_T(clustersXs_data[0].f1, i24);
  clustersXs_data[0].f1->data[0] = SD->u2.f6.xs_data[0];

  /* 'ClusterPoints2D:35' clustersYs{1} = ys; */
  i24 = clustersYs_data[0].f1->size[0] * clustersYs_data[0].f1->size[1];
  clustersYs_data[0].f1->size[1] = 1;
  clustersYs_data[0].f1->size[0] = 1;
  emxEnsureCapacity_real_T(clustersYs_data[0].f1, i24);
  clustersYs_data[0].f1->data[0] = SD->u2.f6.ys_data[0];

  /* 'ClusterPoints2D:38' cellArrayIndex = 1; */
  cellArrayIndex = 1.0;

  /* 'ClusterPoints2D:39' while ~isempty(ptr) */
  emxInit_real_T(&distanceMat, 2);
  emxInitMatrix_cell_wrap_0(reshapes);
  emxInit_real_T(&ex, 1);
  emxInit_int32_T(&idx, 1);
  emxInitMatrix_cell_wrap_0(b_reshapes);
  emxInitMatrix_cell_wrap_0(c_reshapes);
  emxInit_real_T(&d_reshapes, 2);
  while (ptr_size[0] != 0) {
    /*  Calculate the distance from the current cluster to all the other points - */
    /*  calcualte the distance from all the points in a cluster to all the other points */
    /* 'ClusterPoints2D:43' ptrTmp = [clustersXs{cellArrayIndex} clustersYs{cellArrayIndex}]; */
    i24 = (int)cellArrayIndex - 1;
    if ((clustersXs_data[i24].f1->size[0] != 0) && (clustersXs_data[(int)
         cellArrayIndex - 1].f1->size[1] != 0)) {
      m = clustersXs_data[(int)cellArrayIndex - 1].f1->size[0];
    } else if ((clustersYs_data[i24].f1->size[0] != 0) && (clustersYs_data[(int)
                cellArrayIndex - 1].f1->size[1] != 0)) {
      m = clustersYs_data[(int)cellArrayIndex - 1].f1->size[0];
    } else {
      m = clustersXs_data[(int)cellArrayIndex - 1].f1->size[0];
      if (m <= 0) {
        m = 0;
      }

      if (clustersYs_data[(int)cellArrayIndex - 1].f1->size[0] > m) {
        m = clustersYs_data[(int)cellArrayIndex - 1].f1->size[0];
      }
    }

    empty_non_axis_sizes = (m == 0);
    if (empty_non_axis_sizes || ((clustersXs_data[(int)cellArrayIndex - 1]
          .f1->size[0] != 0) && (clustersXs_data[(int)cellArrayIndex - 1]
          .f1->size[1] != 0))) {
      input_sizes_idx_1 = (signed char)clustersXs_data[(int)cellArrayIndex - 1].
        f1->size[1];
    } else {
      input_sizes_idx_1 = 0;
    }

    loop_ub = input_sizes_idx_1;
    if ((input_sizes_idx_1 == clustersXs_data[(int)cellArrayIndex - 1].f1->size
         [1]) && (m == clustersXs_data[(int)cellArrayIndex - 1].f1->size[0])) {
      i25 = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
      reshapes[0].f1->size[1] = input_sizes_idx_1;
      reshapes[0].f1->size[0] = m;
      emxEnsureCapacity_real_T(reshapes[0].f1, i25);
      loop_ub = input_sizes_idx_1 * m;
      for (i25 = 0; i25 < loop_ub; i25++) {
        reshapes[0].f1->data[i25] = clustersXs_data[(int)cellArrayIndex - 1].
          f1->data[i25];
      }
    } else {
      i25 = 0;
      i26 = 0;
      n = 0;
      i27 = 0;
      i28 = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
      reshapes[0].f1->size[1] = input_sizes_idx_1;
      reshapes[0].f1->size[0] = m;
      emxEnsureCapacity_real_T(reshapes[0].f1, i28);
      for (i28 = 0; i28 < m * loop_ub; i28++) {
        reshapes[0].f1->data[i26 + reshapes[0].f1->size[1] * i25] =
          clustersXs_data[(int)cellArrayIndex - 1].f1->data[i27 +
          clustersXs_data[(int)cellArrayIndex - 1].f1->size[1] * n];
        i25++;
        n++;
        if (i25 > reshapes[0].f1->size[0] - 1) {
          i25 = 0;
          i26++;
        }

        if (n > clustersXs_data[(int)cellArrayIndex - 1].f1->size[0] - 1) {
          n = 0;
          i27++;
        }
      }
    }

    if (empty_non_axis_sizes || ((clustersYs_data[(int)cellArrayIndex - 1]
          .f1->size[0] != 0) && (clustersYs_data[(int)cellArrayIndex - 1]
          .f1->size[1] != 0))) {
      input_sizes_idx_1 = (signed char)clustersYs_data[(int)cellArrayIndex - 1].
        f1->size[1];
    } else {
      input_sizes_idx_1 = 0;
    }

    loop_ub = input_sizes_idx_1;
    if ((input_sizes_idx_1 == clustersYs_data[(int)cellArrayIndex - 1].f1->size
         [1]) && (m == clustersYs_data[(int)cellArrayIndex - 1].f1->size[0])) {
      i25 = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
      reshapes[1].f1->size[1] = input_sizes_idx_1;
      reshapes[1].f1->size[0] = m;
      emxEnsureCapacity_real_T(reshapes[1].f1, i25);
      loop_ub = input_sizes_idx_1 * m;
      for (i25 = 0; i25 < loop_ub; i25++) {
        reshapes[1].f1->data[i25] = clustersYs_data[(int)cellArrayIndex - 1].
          f1->data[i25];
      }
    } else {
      i25 = 0;
      i26 = 0;
      n = 0;
      i27 = 0;
      i28 = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
      reshapes[1].f1->size[1] = input_sizes_idx_1;
      reshapes[1].f1->size[0] = m;
      emxEnsureCapacity_real_T(reshapes[1].f1, i28);
      for (i28 = 0; i28 < m * loop_ub; i28++) {
        reshapes[1].f1->data[i26 + reshapes[1].f1->size[1] * i25] =
          clustersYs_data[(int)cellArrayIndex - 1].f1->data[i27 +
          clustersYs_data[(int)cellArrayIndex - 1].f1->size[1] * n];
        i25++;
        n++;
        if (i25 > reshapes[1].f1->size[0] - 1) {
          i25 = 0;
          i26++;
        }

        if (n > clustersYs_data[(int)cellArrayIndex - 1].f1->size[0] - 1) {
          n = 0;
          i27++;
        }
      }
    }

    /* 'ClusterPoints2D:44' distanceMat = pdist2(ptrTmp, ptr); */
    i25 = d_reshapes->size[0] * d_reshapes->size[1];
    d_reshapes->size[1] = reshapes[0].f1->size[1] + reshapes[1].f1->size[1];
    d_reshapes->size[0] = reshapes[0].f1->size[0];
    emxEnsureCapacity_real_T(d_reshapes, i25);
    loop_ub = reshapes[0].f1->size[0];
    for (i25 = 0; i25 < loop_ub; i25++) {
      input_sizes_idx_0 = reshapes[0].f1->size[1];
      for (i26 = 0; i26 < input_sizes_idx_0; i26++) {
        d_reshapes->data[i26 + d_reshapes->size[1] * i25] = reshapes[0].f1->
          data[i26 + reshapes[0].f1->size[1] * i25];
      }
    }

    loop_ub = reshapes[1].f1->size[0];
    for (i25 = 0; i25 < loop_ub; i25++) {
      input_sizes_idx_0 = reshapes[1].f1->size[1];
      for (i26 = 0; i26 < input_sizes_idx_0; i26++) {
        d_reshapes->data[(i26 + reshapes[0].f1->size[1]) + d_reshapes->size[1] *
          i25] = reshapes[1].f1->data[i26 + reshapes[1].f1->size[1] * i25];
      }
    }

    pdist2(SD, d_reshapes, ptr_data, ptr_size, distanceMat);

    /* 'ClusterPoints2D:45' if size(distanceMat,1) == 1 */
    if (distanceMat->size[0] == 1) {
      /* 'ClusterPoints2D:46' [minVal, minInd] = min(distanceMat,[],2); */
      n = distanceMat->size[1];
      i25 = ex->size[0];
      ex->size[0] = 1;
      emxEnsureCapacity_real_T(ex, i25);
      i25 = idx->size[0];
      idx->size[0] = 1;
      emxEnsureCapacity_int32_T(idx, i25);
      idx->data[0] = 1;
      ex->data[0] = distanceMat->data[0];
      for (loop_ub = 2; loop_ub <= n; loop_ub++) {
        b = distanceMat->data[loop_ub - 1];
        empty_non_axis_sizes = ((!rtIsNaN(b)) && (rtIsNaN(ex->data[0]) ||
          (ex->data[0] > b)));
        if (empty_non_axis_sizes) {
          ex->data[0] = distanceMat->data[loop_ub - 1];
          idx->data[0] = loop_ub;
        }
      }

      loop_ub = ex->size[0];
      for (i25 = 0; i25 < loop_ub; i25++) {
        minVal_data[i25] = ex->data[i25];
      }

      loop_ub = idx->size[0];
      for (i25 = 0; i25 < loop_ub; i25++) {
        minInd_data[i25] = idx->data[i25];
      }
    } else {
      /* 'ClusterPoints2D:47' else */
      /* 'ClusterPoints2D:48' [minValTmp, minIndTmp] = min(distanceMat,[],2); */
      m = distanceMat->size[0];
      n = distanceMat->size[1];
      distanceMat_idx_0 = (unsigned int)distanceMat->size[0];
      i25 = ex->size[0];
      ex->size[0] = (int)distanceMat_idx_0;
      emxEnsureCapacity_real_T(ex, i25);
      i25 = idx->size[0];
      idx->size[0] = distanceMat->size[0];
      emxEnsureCapacity_int32_T(idx, i25);
      loop_ub = distanceMat->size[0];
      for (i25 = 0; i25 < loop_ub; i25++) {
        idx->data[i25] = 1;
      }

      if (distanceMat->size[0] >= 1) {
        for (input_sizes_idx_0 = 0; input_sizes_idx_0 < m; input_sizes_idx_0++)
        {
          ex->data[input_sizes_idx_0] = distanceMat->data[distanceMat->size[1] *
            input_sizes_idx_0];
          for (loop_ub = 2; loop_ub <= n; loop_ub++) {
            b = distanceMat->data[(loop_ub + distanceMat->size[1] *
              input_sizes_idx_0) - 1];
            empty_non_axis_sizes = ((!rtIsNaN(b)) && (rtIsNaN(ex->
              data[input_sizes_idx_0]) || (ex->data[input_sizes_idx_0] > b)));
            if (empty_non_axis_sizes) {
              ex->data[input_sizes_idx_0] = distanceMat->data[(loop_ub +
                distanceMat->size[1] * input_sizes_idx_0) - 1];
              idx->data[input_sizes_idx_0] = loop_ub;
            }
          }
        }
      }

      /* 'ClusterPoints2D:49' [~, minMinInd] = min(minValTmp); */
      n = ex->size[0];
      if (ex->size[0] <= 2) {
        if (ex->size[0] == 1) {
          input_sizes_idx_0 = 1;
        } else if ((ex->data[0] > ex->data[1]) || (rtIsNaN(ex->data[0]) &&
                    (!rtIsNaN(ex->data[1])))) {
          input_sizes_idx_0 = 2;
        } else {
          input_sizes_idx_0 = 1;
        }
      } else {
        if (!rtIsNaN(ex->data[0])) {
          input_sizes_idx_0 = 1;
        } else {
          input_sizes_idx_0 = 0;
          loop_ub = 2;
          exitg1 = false;
          while ((!exitg1) && (loop_ub <= ex->size[0])) {
            if (!rtIsNaN(ex->data[loop_ub - 1])) {
              input_sizes_idx_0 = loop_ub;
              exitg1 = true;
            } else {
              loop_ub++;
            }
          }
        }

        if (input_sizes_idx_0 == 0) {
          input_sizes_idx_0 = 1;
        } else {
          b = ex->data[input_sizes_idx_0 - 1];
          i25 = input_sizes_idx_0 + 1;
          for (loop_ub = i25; loop_ub <= n; loop_ub++) {
            if (b > ex->data[loop_ub - 1]) {
              b = ex->data[loop_ub - 1];
              input_sizes_idx_0 = loop_ub;
            }
          }
        }
      }

      /* 'ClusterPoints2D:49' ~ */
      /* 'ClusterPoints2D:51' minVal = minValTmp(minMinInd); */
      minVal_data[0] = ex->data[input_sizes_idx_0 - 1];

      /* 'ClusterPoints2D:52' minInd = minIndTmp(minMinInd); */
      minInd_data[0] = idx->data[input_sizes_idx_0 - 1];
    }

    /* 'ClusterPoints2D:56' if minVal <= maxDistance */
    b_minVal_data[0] = (minVal_data[0] <= maxDistance);
    if (ifWhileCond(b_minVal_data)) {
      /*  Same cluster */
      /* 'ClusterPoints2D:58' clustersXs{cellArrayIndex} = [clustersXs{cellArrayIndex}; ptr(minInd,1)]; */
      if ((clustersXs_data[(int)cellArrayIndex - 1].f1->size[0] != 0) &&
          (clustersXs_data[(int)cellArrayIndex - 1].f1->size[1] != 0)) {
        input_sizes_idx_1 = (signed char)clustersXs_data[(int)cellArrayIndex - 1]
          .f1->size[1];
      } else {
        input_sizes_idx_1 = 1;
      }

      if ((input_sizes_idx_1 == 0) || ((clustersXs_data[(int)cellArrayIndex - 1]
            .f1->size[0] != 0) && (clustersXs_data[(int)cellArrayIndex - 1]
            .f1->size[1] != 0))) {
        input_sizes_idx_0 = clustersXs_data[(int)cellArrayIndex - 1].f1->size[0];
      } else {
        input_sizes_idx_0 = 0;
      }

      loop_ub = input_sizes_idx_1;
      if ((input_sizes_idx_1 == clustersXs_data[(int)cellArrayIndex - 1]
           .f1->size[1]) && (input_sizes_idx_0 == clustersXs_data[(int)
                             cellArrayIndex - 1].f1->size[0])) {
        i25 = b_reshapes[0].f1->size[0] * b_reshapes[0].f1->size[1];
        b_reshapes[0].f1->size[1] = input_sizes_idx_1;
        b_reshapes[0].f1->size[0] = input_sizes_idx_0;
        emxEnsureCapacity_real_T(b_reshapes[0].f1, i25);
        loop_ub = input_sizes_idx_1 * input_sizes_idx_0;
        for (i25 = 0; i25 < loop_ub; i25++) {
          b_reshapes[0].f1->data[i25] = clustersXs_data[(int)cellArrayIndex - 1]
            .f1->data[i25];
        }
      } else {
        i25 = 0;
        i26 = 0;
        n = 0;
        i27 = 0;
        i28 = b_reshapes[0].f1->size[0] * b_reshapes[0].f1->size[1];
        b_reshapes[0].f1->size[1] = input_sizes_idx_1;
        b_reshapes[0].f1->size[0] = input_sizes_idx_0;
        emxEnsureCapacity_real_T(b_reshapes[0].f1, i28);
        for (i28 = 0; i28 < input_sizes_idx_0 * loop_ub; i28++) {
          b_reshapes[0].f1->data[i26 + b_reshapes[0].f1->size[1] * i25] =
            clustersXs_data[(int)cellArrayIndex - 1].f1->data[i27 +
            clustersXs_data[(int)cellArrayIndex - 1].f1->size[1] * n];
          i25++;
          n++;
          if (i25 > b_reshapes[0].f1->size[0] - 1) {
            i25 = 0;
            i26++;
          }

          if (n > clustersXs_data[(int)cellArrayIndex - 1].f1->size[0] - 1) {
            n = 0;
            i27++;
          }
        }
      }

      loop_ub = input_sizes_idx_1;
      i25 = 0;
      i26 = 0;
      n = b_reshapes[1].f1->size[0] * b_reshapes[1].f1->size[1];
      b_reshapes[1].f1->size[1] = input_sizes_idx_1;
      b_reshapes[1].f1->size[0] = 1;
      emxEnsureCapacity_real_T(b_reshapes[1].f1, n);
      for (n = 0; n < loop_ub; n++) {
        b_reshapes[1].f1->data[i25] = ptr_data[(minInd_data[i26] - 1) << 1];
        i26++;
        i25++;
      }

      i25 = clustersXs_data[i24].f1->size[0] * clustersXs_data[(int)
        cellArrayIndex - 1].f1->size[1];
      clustersXs_data[(int)cellArrayIndex - 1].f1->size[1] = b_reshapes[0]
        .f1->size[1];
      clustersXs_data[(int)cellArrayIndex - 1].f1->size[0] = b_reshapes[0]
        .f1->size[0] + b_reshapes[1].f1->size[0];
      emxEnsureCapacity_real_T(clustersXs_data[(int)cellArrayIndex - 1].f1, i25);
      loop_ub = b_reshapes[0].f1->size[0];
      for (i25 = 0; i25 < loop_ub; i25++) {
        input_sizes_idx_0 = b_reshapes[0].f1->size[1];
        for (i26 = 0; i26 < input_sizes_idx_0; i26++) {
          clustersXs_data[(int)cellArrayIndex - 1].f1->data[i26 +
            clustersXs_data[(int)cellArrayIndex - 1].f1->size[1] * i25] =
            b_reshapes[0].f1->data[i26 + b_reshapes[0].f1->size[1] * i25];
        }
      }

      loop_ub = b_reshapes[1].f1->size[0];
      for (i25 = 0; i25 < loop_ub; i25++) {
        input_sizes_idx_0 = b_reshapes[1].f1->size[1];
        for (i26 = 0; i26 < input_sizes_idx_0; i26++) {
          clustersXs_data[(int)cellArrayIndex - 1].f1->data[i26 +
            clustersXs_data[(int)cellArrayIndex - 1].f1->size[1] * (i25 +
            b_reshapes[0].f1->size[0])] = b_reshapes[1].f1->data[i26 +
            b_reshapes[1].f1->size[1] * i25];
        }
      }

      /* 'ClusterPoints2D:59' clustersYs{cellArrayIndex} = [clustersYs{cellArrayIndex}; ptr(minInd,2)]; */
      if ((clustersYs_data[(int)cellArrayIndex - 1].f1->size[0] != 0) &&
          (clustersYs_data[(int)cellArrayIndex - 1].f1->size[1] != 0)) {
        input_sizes_idx_1 = (signed char)clustersYs_data[(int)cellArrayIndex - 1]
          .f1->size[1];
      } else {
        input_sizes_idx_1 = 1;
      }

      if ((input_sizes_idx_1 == 0) || ((clustersYs_data[(int)cellArrayIndex - 1]
            .f1->size[0] != 0) && (clustersYs_data[(int)cellArrayIndex - 1]
            .f1->size[1] != 0))) {
        input_sizes_idx_0 = clustersYs_data[(int)cellArrayIndex - 1].f1->size[0];
      } else {
        input_sizes_idx_0 = 0;
      }

      loop_ub = input_sizes_idx_1;
      if ((input_sizes_idx_1 == clustersYs_data[(int)cellArrayIndex - 1]
           .f1->size[1]) && (input_sizes_idx_0 == clustersYs_data[(int)
                             cellArrayIndex - 1].f1->size[0])) {
        i25 = c_reshapes[0].f1->size[0] * c_reshapes[0].f1->size[1];
        c_reshapes[0].f1->size[1] = input_sizes_idx_1;
        c_reshapes[0].f1->size[0] = input_sizes_idx_0;
        emxEnsureCapacity_real_T(c_reshapes[0].f1, i25);
        loop_ub = input_sizes_idx_1 * input_sizes_idx_0;
        for (i25 = 0; i25 < loop_ub; i25++) {
          c_reshapes[0].f1->data[i25] = clustersYs_data[(int)cellArrayIndex - 1]
            .f1->data[i25];
        }
      } else {
        i25 = 0;
        i26 = 0;
        n = 0;
        i27 = 0;
        i28 = c_reshapes[0].f1->size[0] * c_reshapes[0].f1->size[1];
        c_reshapes[0].f1->size[1] = input_sizes_idx_1;
        c_reshapes[0].f1->size[0] = input_sizes_idx_0;
        emxEnsureCapacity_real_T(c_reshapes[0].f1, i28);
        for (i28 = 0; i28 < input_sizes_idx_0 * loop_ub; i28++) {
          c_reshapes[0].f1->data[i26 + c_reshapes[0].f1->size[1] * i25] =
            clustersYs_data[(int)cellArrayIndex - 1].f1->data[i27 +
            clustersYs_data[(int)cellArrayIndex - 1].f1->size[1] * n];
          i25++;
          n++;
          if (i25 > c_reshapes[0].f1->size[0] - 1) {
            i25 = 0;
            i26++;
          }

          if (n > clustersYs_data[(int)cellArrayIndex - 1].f1->size[0] - 1) {
            n = 0;
            i27++;
          }
        }
      }

      loop_ub = input_sizes_idx_1;
      i25 = 0;
      i26 = 0;
      n = c_reshapes[1].f1->size[0] * c_reshapes[1].f1->size[1];
      c_reshapes[1].f1->size[1] = input_sizes_idx_1;
      c_reshapes[1].f1->size[0] = 1;
      emxEnsureCapacity_real_T(c_reshapes[1].f1, n);
      for (n = 0; n < loop_ub; n++) {
        c_reshapes[1].f1->data[i25] = ptr_data[1 + ((minInd_data[i26] - 1) << 1)];
        i26++;
        i25++;
      }

      i24 = clustersYs_data[i24].f1->size[0] * clustersYs_data[(int)
        cellArrayIndex - 1].f1->size[1];
      clustersYs_data[(int)cellArrayIndex - 1].f1->size[1] = c_reshapes[0]
        .f1->size[1];
      clustersYs_data[(int)cellArrayIndex - 1].f1->size[0] = c_reshapes[0]
        .f1->size[0] + c_reshapes[1].f1->size[0];
      emxEnsureCapacity_real_T(clustersYs_data[(int)cellArrayIndex - 1].f1, i24);
      loop_ub = c_reshapes[0].f1->size[0];
      for (i24 = 0; i24 < loop_ub; i24++) {
        input_sizes_idx_0 = c_reshapes[0].f1->size[1];
        for (i25 = 0; i25 < input_sizes_idx_0; i25++) {
          clustersYs_data[(int)cellArrayIndex - 1].f1->data[i25 +
            clustersYs_data[(int)cellArrayIndex - 1].f1->size[1] * i24] =
            c_reshapes[0].f1->data[i25 + c_reshapes[0].f1->size[1] * i24];
        }
      }

      loop_ub = c_reshapes[1].f1->size[0];
      for (i24 = 0; i24 < loop_ub; i24++) {
        input_sizes_idx_0 = c_reshapes[1].f1->size[1];
        for (i25 = 0; i25 < input_sizes_idx_0; i25++) {
          clustersYs_data[(int)cellArrayIndex - 1].f1->data[i25 +
            clustersYs_data[(int)cellArrayIndex - 1].f1->size[1] * (i24 +
            c_reshapes[0].f1->size[0])] = c_reshapes[1].f1->data[i25 +
            c_reshapes[1].f1->size[1] * i24];
        }
      }

      /* 'ClusterPoints2D:61' ptr(minInd,:) = []; */
      tmp_size[1] = 2;
      tmp_size[0] = ptr_size[0];
      loop_ub = 2 * ptr_size[0];
      if (0 <= loop_ub - 1) {
        memcpy(&SD->u2.f6.tmp_data[0], &ptr_data[0], (unsigned int)(loop_ub *
                (int)sizeof(double)));
      }

      b_minInd_data[0] = minInd_data[0];
      b_nullAssignment(SD, SD->u2.f6.tmp_data, tmp_size, b_minInd_data);
      ptr_size[1] = 2;
      ptr_size[0] = tmp_size[0];
      loop_ub = tmp_size[1] * tmp_size[0];
      if (0 <= loop_ub - 1) {
        memcpy(&ptr_data[0], &SD->u2.f6.tmp_data[0], (unsigned int)(loop_ub *
                (int)sizeof(double)));
      }
    } else {
      /* 'ClusterPoints2D:62' else */
      /*  new cluster */
      /* 'ClusterPoints2D:64' xs2 = ptr(minInd,1); */
      input_sizes_idx_0 = (minInd_data[0] - 1) << 1;

      /* 'ClusterPoints2D:65' cellArrayIndex = cellArrayIndex + 1; */
      cellArrayIndex++;

      /* 'ClusterPoints2D:66' clustersXs{cellArrayIndex} = xs2; */
      i24 = (int)cellArrayIndex - 1;
      i25 = clustersXs_data[i24].f1->size[0] * clustersXs_data[(int)
        cellArrayIndex - 1].f1->size[1];
      clustersXs_data[(int)cellArrayIndex - 1].f1->size[1] = 1;
      clustersXs_data[(int)cellArrayIndex - 1].f1->size[0] = 1;
      emxEnsureCapacity_real_T(clustersXs_data[(int)cellArrayIndex - 1].f1, i25);
      clustersXs_data[(int)cellArrayIndex - 1].f1->data[0] =
        ptr_data[input_sizes_idx_0];

      /* 'ClusterPoints2D:68' ys2 = ptr(minInd,2); */
      minVal_data[0] = ptr_data[1 + input_sizes_idx_0];

      /* 'ClusterPoints2D:69' clustersYs{cellArrayIndex} = ys2; */
      i24 = clustersYs_data[i24].f1->size[0] * clustersYs_data[(int)
        cellArrayIndex - 1].f1->size[1];
      clustersYs_data[(int)cellArrayIndex - 1].f1->size[1] = 1;
      clustersYs_data[(int)cellArrayIndex - 1].f1->size[0] = 1;
      emxEnsureCapacity_real_T(clustersYs_data[(int)cellArrayIndex - 1].f1, i24);
      clustersYs_data[(int)cellArrayIndex - 1].f1->data[0] = ptr_data[1 +
        input_sizes_idx_0];

      /* 'ClusterPoints2D:71' ptr(minInd,:) = []; */
      tmp_size[1] = 2;
      tmp_size[0] = ptr_size[0];
      loop_ub = 2 * ptr_size[0];
      if (0 <= loop_ub - 1) {
        memcpy(&SD->u2.f6.tmp_data[0], &ptr_data[0], (unsigned int)(loop_ub *
                (int)sizeof(double)));
      }

      b_minInd_data[0] = minInd_data[0];
      b_nullAssignment(SD, SD->u2.f6.tmp_data, tmp_size, b_minInd_data);
      ptr_size[1] = 2;
      ptr_size[0] = tmp_size[0];
      loop_ub = tmp_size[1] * tmp_size[0];
      if (0 <= loop_ub - 1) {
        memcpy(&ptr_data[0], &SD->u2.f6.tmp_data[0], (unsigned int)(loop_ub *
                (int)sizeof(double)));
      }
    }
  }

  emxFree_real_T(&d_reshapes);
  emxFreeMatrix_cell_wrap_0(c_reshapes);
  emxFreeMatrix_cell_wrap_0(b_reshapes);
  emxFree_int32_T(&idx);
  emxFree_real_T(&ex);
  emxFreeMatrix_cell_wrap_0(reshapes);
  emxFree_real_T(&distanceMat);
}

/*
 *
 */
static void ComputeArea(b_emxArray_struct_T *stats, struct_T
  *statsAlreadyComputed)
{
  int i44;
  int k;
  if (!statsAlreadyComputed->Area) {
    statsAlreadyComputed->Area = true;
    i44 = stats->size[0];
    for (k = 0; k < i44; k++) {
      stats->data[k].Area = stats->data[k].PixelIdxList->size[0];
    }
  }
}

/*
 *
 */
static void ComputeCentroid(const double imageSize[2], b_emxArray_struct_T
  *stats, struct_T *statsAlreadyComputed)
{
  int i47;
  int k;
  int vlen;
  double y[2];
  int b_k;
  if (!statsAlreadyComputed->Centroid) {
    statsAlreadyComputed->Centroid = true;
    ComputePixelList(imageSize, stats, statsAlreadyComputed);
    i47 = stats->size[0];
    for (k = 0; k < i47; k++) {
      vlen = stats->data[k].PixelList->size[0];
      if (stats->data[k].PixelList->size[0] == 0) {
        y[0] = 0.0;
        y[1] = 0.0;
      } else {
        y[0] = stats->data[k].PixelList->data[0];
        y[1] = stats->data[k].PixelList->data[1];
        for (b_k = 2; b_k <= vlen; b_k++) {
          if (vlen >= 2) {
            y[0] += stats->data[k].PixelList->data[(b_k - 1) << 1];
            y[1] += stats->data[k].PixelList->data[1 + ((b_k - 1) << 1)];
          }
        }
      }

      vlen = stats->data[k].PixelList->size[0];
      stats->data[k].Centroid[0] = y[0] / (double)vlen;
      stats->data[k].Centroid[1] = y[1] / (double)vlen;
    }
  }
}

/*
 *
 */
static void ComputeEllipseParams(const double imageSize[2], b_emxArray_struct_T *
  stats, struct_T *statsAlreadyComputed)
{
  int i48;
  emxArray_real_T *x;
  emxArray_real_T *y;
  emxArray_real_T *b_x;
  int k;
  int loop_ub;
  double uxx;
  int i49;
  double uyy;
  double uxy;
  double common_tmp;
  double b_common_tmp;
  double d9;
  if (statsAlreadyComputed->MajorAxisLength &&
      statsAlreadyComputed->MinorAxisLength && statsAlreadyComputed->Orientation
      && statsAlreadyComputed->Eccentricity) {
  } else {
    statsAlreadyComputed->MajorAxisLength = true;
    statsAlreadyComputed->MinorAxisLength = true;
    statsAlreadyComputed->Eccentricity = true;
    statsAlreadyComputed->Orientation = true;
    ComputePixelList(imageSize, stats, statsAlreadyComputed);
    ComputeCentroid(imageSize, stats, statsAlreadyComputed);
    i48 = stats->size[0];
    emxInit_real_T(&x, 1);
    emxInit_real_T(&y, 1);
    emxInit_real_T(&b_x, 1);
    for (k = 0; k < i48; k++) {
      if (stats->data[k].PixelList->size[0] == 0) {
        stats->data[k].MajorAxisLength = 0.0;
        stats->data[k].MinorAxisLength = 0.0;
        stats->data[k].Eccentricity = 0.0;
        stats->data[k].Orientation = 0.0;
      } else {
        loop_ub = stats->data[k].PixelList->size[0];
        uxx = stats->data[k].Centroid[0];
        i49 = x->size[0];
        x->size[0] = loop_ub;
        emxEnsureCapacity_real_T(x, i49);
        for (i49 = 0; i49 < loop_ub; i49++) {
          x->data[i49] = stats->data[k].PixelList->data[i49 << 1] - uxx;
        }

        loop_ub = stats->data[k].PixelList->size[0];
        uxx = stats->data[k].Centroid[1];
        i49 = y->size[0];
        y->size[0] = loop_ub;
        emxEnsureCapacity_real_T(y, i49);
        for (i49 = 0; i49 < loop_ub; i49++) {
          y->data[i49] = -(stats->data[k].PixelList->data[1 + (i49 << 1)] - uxx);
        }

        power(x, b_x);
        uxx = b_sum(b_x) / (double)x->size[0] + 0.083333333333333329;
        power(y, b_x);
        uyy = b_sum(b_x) / (double)x->size[0] + 0.083333333333333329;
        i49 = b_x->size[0];
        b_x->size[0] = x->size[0];
        emxEnsureCapacity_real_T(b_x, i49);
        loop_ub = x->size[0];
        for (i49 = 0; i49 < loop_ub; i49++) {
          b_x->data[i49] = x->data[i49] * y->data[i49];
        }

        uxy = b_sum(b_x) / (double)x->size[0];
        common_tmp = uxx - uyy;
        b_common_tmp = std::sqrt(rt_powd_snf(common_tmp, 2.0) + 4.0 *
          rt_powd_snf(uxy, 2.0));
        d9 = uxx + uyy;
        stats->data[k].MajorAxisLength = 2.8284271247461903 * std::sqrt(d9 +
          b_common_tmp);
        stats->data[k].MinorAxisLength = 2.8284271247461903 * std::sqrt(d9 -
          b_common_tmp);
        stats->data[k].Eccentricity = 2.0 * std::sqrt(rt_powd_snf(stats->data[k]
          .MajorAxisLength / 2.0, 2.0) - rt_powd_snf(stats->data[k].
          MinorAxisLength / 2.0, 2.0)) / stats->data[k].MajorAxisLength;
        if (uyy > uxx) {
          uyy = (uyy - uxx) + std::sqrt(rt_powd_snf(uyy - uxx, 2.0) + 4.0 *
            rt_powd_snf(uxy, 2.0));
          uxx = 2.0 * uxy;
        } else {
          uyy = 2.0 * uxy;
          uxx = common_tmp + b_common_tmp;
        }

        if ((uyy == 0.0) && (uxx == 0.0)) {
          stats->data[k].Orientation = 0.0;
        } else {
          stats->data[k].Orientation = 57.295779513082323 * std::atan(uyy / uxx);
        }
      }
    }

    emxFree_real_T(&b_x);
    emxFree_real_T(&y);
    emxFree_real_T(&x);
  }
}

/*
 *
 */
static void ComputePixelIdxList(const double L_data[], const int L_size[2],
  double numObjs, b_emxArray_struct_T *stats, struct_T *statsAlreadyComputed)
{
  emxArray_real_T *regionLengths;
  int i40;
  int jRow;
  emxArray_real_T *regionIndices;
  int i41;
  double d6;
  int j;
  int idx;
  emxArray_int32_T *idxCount_tmp;
  int k;
  emxArray_int32_T *idxCount;
  int q;
  statsAlreadyComputed->PixelIdxList = true;
  if (numObjs != 0.0) {
    emxInit_real_T(&regionLengths, 1);
    i40 = regionLengths->size[0];
    jRow = (int)numObjs;
    regionLengths->size[0] = jRow;
    emxEnsureCapacity_real_T(regionLengths, i40);
    for (i40 = 0; i40 < jRow; i40++) {
      regionLengths->data[i40] = 0.0;
    }

    i40 = L_size[0];
    for (jRow = 0; jRow < i40; jRow++) {
      i41 = L_size[1];
      for (j = 0; j < i41; j++) {
        idx = (int)L_data[j + L_size[1] * jRow];
        if (idx > 0) {
          regionLengths->data[idx - 1]++;
        }
      }
    }

    emxInit_real_T(&regionIndices, 1);
    d6 = sum(regionLengths);
    i40 = regionIndices->size[0];
    regionIndices->size[0] = (int)d6;
    emxEnsureCapacity_real_T(regionIndices, i40);
    jRow = 2;
    if (regionLengths->size[0] != 1) {
      jRow = 1;
    }

    if ((1 == jRow) && (regionLengths->size[0] != 1)) {
      i40 = regionLengths->size[0];
      for (k = 0; k <= i40 - 2; k++) {
        regionLengths->data[k + 1] += regionLengths->data[k];
      }
    }

    emxInit_int32_T(&idxCount_tmp, 1);
    i40 = idxCount_tmp->size[0];
    idxCount_tmp->size[0] = 1 + regionLengths->size[0];
    emxEnsureCapacity_int32_T(idxCount_tmp, i40);
    idxCount_tmp->data[0] = 0;
    jRow = regionLengths->size[0];
    for (i40 = 0; i40 < jRow; i40++) {
      idxCount_tmp->data[i40 + 1] = (int)regionLengths->data[i40];
    }

    emxFree_real_T(&regionLengths);
    emxInit_int32_T(&idxCount, 1);
    i40 = idxCount->size[0];
    idxCount->size[0] = idxCount_tmp->size[0];
    emxEnsureCapacity_int32_T(idxCount, i40);
    jRow = idxCount_tmp->size[0];
    for (i40 = 0; i40 < jRow; i40++) {
      idxCount->data[i40] = idxCount_tmp->data[i40];
    }

    j = 1;
    jRow = 1;
    i40 = L_size[0];
    for (k = 0; k < i40; k++) {
      i41 = L_size[1];
      for (q = 0; q < i41; q++) {
        idx = (int)L_data[q + L_size[1] * k] - 1;
        if (idx + 1 > 0) {
          idxCount->data[idx]++;
          regionIndices->data[idxCount->data[idx] - 1] = jRow;
        }

        jRow += L_size[0];
      }

      j++;
      jRow = j;
    }

    emxFree_int32_T(&idxCount);
    i40 = stats->size[0];
    for (k = 0; k < i40; k++) {
      if (idxCount_tmp->data[k] + 1 > idxCount_tmp->data[k + 1]) {
        i41 = 1;
        q = 0;
      } else {
        i41 = idxCount_tmp->data[k] + 1;
        q = idxCount_tmp->data[k + 1];
      }

      j = stats->data[k].PixelIdxList->size[0];
      jRow = (q - i41) + 1;
      stats->data[k].PixelIdxList->size[0] = jRow;
      emxEnsureCapacity_real_T(stats->data[k].PixelIdxList, j);
      for (q = 0; q < jRow; q++) {
        stats->data[k].PixelIdxList->data[q] = regionIndices->data[(i41 + q) - 1];
      }

      sort(stats->data[k].PixelIdxList);
    }

    emxFree_int32_T(&idxCount_tmp);
    emxFree_real_T(&regionIndices);
  }
}

/*
 *
 */
static void ComputePixelList(const double imageSize[2], b_emxArray_struct_T
  *stats, struct_T *statsAlreadyComputed)
{
  int i45;
  emxArray_int32_T *v1;
  emxArray_int32_T *vk;
  double b_imageSize;
  int k;
  int i46;
  int loop_ub;
  int unnamed_idx_1;
  if (!statsAlreadyComputed->PixelList) {
    statsAlreadyComputed->PixelList = true;
    i45 = stats->size[0];
    emxInit_int32_T(&v1, 1);
    emxInit_int32_T(&vk, 1);
    b_imageSize = imageSize[0];
    for (k = 0; k < i45; k++) {
      if (stats->data[k].PixelIdxList->size[0] != 0) {
        i46 = v1->size[0];
        v1->size[0] = stats->data[k].PixelIdxList->size[0];
        emxEnsureCapacity_int32_T(v1, i46);
        loop_ub = stats->data[k].PixelIdxList->size[0];
        for (i46 = 0; i46 < loop_ub; i46++) {
          v1->data[i46] = (int)stats->data[k].PixelIdxList->data[i46] - 1;
        }

        i46 = vk->size[0];
        vk->size[0] = v1->size[0];
        emxEnsureCapacity_int32_T(vk, i46);
        loop_ub = v1->size[0];
        for (i46 = 0; i46 < loop_ub; i46++) {
          vk->data[i46] = div_s32(v1->data[i46], (int)b_imageSize);
        }

        i46 = v1->size[0];
        emxEnsureCapacity_int32_T(v1, i46);
        loop_ub = v1->size[0];
        for (i46 = 0; i46 < loop_ub; i46++) {
          v1->data[i46] -= vk->data[i46] * (int)b_imageSize;
        }

        i46 = v1->size[0];
        emxEnsureCapacity_int32_T(v1, i46);
        loop_ub = v1->size[0];
        for (i46 = 0; i46 < loop_ub; i46++) {
          v1->data[i46]++;
        }

        i46 = vk->size[0];
        emxEnsureCapacity_int32_T(vk, i46);
        loop_ub = vk->size[0];
        for (i46 = 0; i46 < loop_ub; i46++) {
          vk->data[i46]++;
        }

        loop_ub = vk->size[0];
        unnamed_idx_1 = v1->size[0];
        i46 = stats->data[k].PixelList->size[0] * stats->data[k].PixelList->
          size[1];
        stats->data[k].PixelList->size[1] = 2;
        stats->data[k].PixelList->size[0] = loop_ub;
        emxEnsureCapacity_real_T(stats->data[k].PixelList, i46);
        for (i46 = 0; i46 < loop_ub; i46++) {
          stats->data[k].PixelList->data[i46 << 1] = vk->data[i46];
        }

        for (i46 = 0; i46 < unnamed_idx_1; i46++) {
          stats->data[k].PixelList->data[1 + (i46 << 1)] = v1->data[i46];
        }
      } else {
        stats->data[k].PixelList->size[1] = 2;
        stats->data[k].PixelList->size[0] = 0;
      }
    }

    emxFree_int32_T(&vk);
    emxFree_int32_T(&v1);
  }
}

/*
 * function [smartLoaderStruct, heightMap_res, deg_loaderProperties] = CreateHeightMap(smartLoaderStruct, xyz)
 */
static void CreateHeightMap(SmartLoaderStackData *SD, SmartLoaderStruct
  *smartLoaderStruct, const double xyz_data[], const int xyz_size[2], float
  heightMap_res_data[], int heightMap_res_size[2])
{
  int i37;
  int idx;
  double minx;
  int m;
  boolean_T exitg1;
  int i38;
  double d5;
  double miny;
  double minz;
  int x_size[1];
  int y_size[1];
  int na;
  int z_size[1];
  emxArray_real_T y_data;
  emxArray_real_T x_data;
  boolean_T imgDims[2];
  emxArray_real32_T *img;
  emxArray_real32_T *temp;
  short outsize_idx_0;
  float minval_data[1024];
  float b;
  int B_size[2];
  boolean_T nhood[9];
  double asizeT[2];
  double nsizeT[2];
  double b_nsizeT[2];
  double c_nsizeT[2];
  double d_nsizeT[2];
  emxArray_struct_T *datablob;
  double e_nsizeT[2];
  int tmp_size[2];
  emxArray_real_T *areaArr;

  /* 'CreateHeightMap:3' coder.varsize('heightMap_res', [SmartLoaderCompilationConstants.HeightMapMaxDimSize SmartLoaderCompilationConstants.HeightMapMaxDimSize], [1 1]); */
  /* 'CreateHeightMap:4' if ~coder.target('Matlab') */
  /* 'CreateHeightMap:5' deg_loaderProperties = zeros(0,0,'uint8'); */
  /* 'CreateHeightMap:8' Nx=0.01; */
  /* res 1cm */
  /* 'CreateHeightMap:9' Ny=0.01; */
  /* res 1cm */
  /* 'CreateHeightMap:10' Nz=0.01; */
  /* res 1cm */
  /*  reduce to min */
  /* 'CreateHeightMap:13' pc=xyz; */
  /* 'CreateHeightMap:14' minx=min(pc(:,1)); */
  i37 = xyz_size[0];
  if (xyz_size[0] <= 2) {
    if (xyz_size[0] == 1) {
      minx = xyz_data[0];
    } else if ((xyz_data[0] > xyz_data[3]) || (rtIsNaN(xyz_data[0]) && (!rtIsNaN
                 (xyz_data[3])))) {
      minx = xyz_data[3];
    } else {
      minx = xyz_data[0];
    }
  } else {
    if (!rtIsNaN(xyz_data[0])) {
      idx = 1;
    } else {
      idx = 0;
      m = 2;
      exitg1 = false;
      while ((!exitg1) && (m <= xyz_size[0])) {
        if (!rtIsNaN(xyz_data[3 * (m - 1)])) {
          idx = m;
          exitg1 = true;
        } else {
          m++;
        }
      }
    }

    if (idx == 0) {
      minx = xyz_data[0];
    } else {
      minx = xyz_data[3 * (idx - 1)];
      i38 = idx + 1;
      for (m = i38; m <= i37; m++) {
        d5 = xyz_data[3 * (m - 1)];
        if (minx > d5) {
          minx = d5;
        }
      }
    }
  }

  /* 'CreateHeightMap:15' miny=min(pc(:,2)); */
  i37 = xyz_size[0];
  if (xyz_size[0] <= 2) {
    if (xyz_size[0] == 1) {
      miny = xyz_data[1];
    } else if ((xyz_data[1] > xyz_data[4]) || (rtIsNaN(xyz_data[1]) && (!rtIsNaN
                 (xyz_data[4])))) {
      miny = xyz_data[4];
    } else {
      miny = xyz_data[1];
    }
  } else {
    if (!rtIsNaN(xyz_data[1])) {
      idx = 1;
    } else {
      idx = 0;
      m = 2;
      exitg1 = false;
      while ((!exitg1) && (m <= xyz_size[0])) {
        if (!rtIsNaN(xyz_data[1 + 3 * (m - 1)])) {
          idx = m;
          exitg1 = true;
        } else {
          m++;
        }
      }
    }

    if (idx == 0) {
      miny = xyz_data[1];
    } else {
      miny = xyz_data[1 + 3 * (idx - 1)];
      i38 = idx + 1;
      for (m = i38; m <= i37; m++) {
        d5 = xyz_data[1 + 3 * (m - 1)];
        if (miny > d5) {
          miny = d5;
        }
      }
    }
  }

  /* 'CreateHeightMap:16' minz=min(pc(:,3)); */
  i37 = xyz_size[0];
  if (xyz_size[0] <= 2) {
    if (xyz_size[0] == 1) {
      minz = xyz_data[2];
    } else if ((xyz_data[2] > xyz_data[5]) || (rtIsNaN(xyz_data[2]) && (!rtIsNaN
                 (xyz_data[5])))) {
      minz = xyz_data[5];
    } else {
      minz = xyz_data[2];
    }
  } else {
    if (!rtIsNaN(xyz_data[2])) {
      idx = 1;
    } else {
      idx = 0;
      m = 2;
      exitg1 = false;
      while ((!exitg1) && (m <= xyz_size[0])) {
        if (!rtIsNaN(xyz_data[2 + 3 * (m - 1)])) {
          idx = m;
          exitg1 = true;
        } else {
          m++;
        }
      }
    }

    if (idx == 0) {
      minz = xyz_data[2];
    } else {
      minz = xyz_data[2 + 3 * (idx - 1)];
      i38 = idx + 1;
      for (m = i38; m <= i37; m++) {
        d5 = xyz_data[2 + 3 * (m - 1)];
        if (minz > d5) {
          minz = d5;
        }
      }
    }
  }

  /* 'CreateHeightMap:17' pc2=[pc(:,1)-minx,pc(:,2)-miny,pc(:,3)-minz]; */
  x_size[0] = xyz_size[0];
  m = xyz_size[0];
  for (i37 = 0; i37 < m; i37++) {
    SD->u1.f5.x_data[i37] = xyz_data[3 * i37] - minx;
  }

  y_size[0] = xyz_size[0];
  m = xyz_size[0];
  for (i37 = 0; i37 < m; i37++) {
    SD->u1.f5.y_data[i37] = xyz_data[1 + 3 * i37] - miny;
  }

  m = xyz_size[0];
  for (i37 = 0; i37 < m; i37++) {
    SD->u1.f5.z_data[i37] = xyz_data[2 + 3 * i37] - minz;
  }

  na = x_size[0];
  idx = y_size[0];
  for (i37 = 0; i37 < na; i37++) {
    SD->u1.f5.pc2_data[3 * i37] = SD->u1.f5.x_data[i37];
  }

  for (i37 = 0; i37 < idx; i37++) {
    SD->u1.f5.pc2_data[1 + 3 * i37] = SD->u1.f5.y_data[i37];
  }

  m = xyz_size[0];
  for (i37 = 0; i37 < m; i37++) {
    SD->u1.f5.pc2_data[2 + 3 * i37] = SD->u1.f5.z_data[i37];
  }

  /*  Divide to pixels */
  /* 'CreateHeightMap:20' x=[round(pc2(:,1)*100)./100]/Nx+1; */
  x_size[0] = na;
  for (i37 = 0; i37 < na; i37++) {
    SD->u1.f5.x_data[i37] = SD->u1.f5.pc2_data[3 * i37] * 100.0;
  }

  b_round(SD->u1.f5.x_data, x_size);
  m = x_size[0];
  for (i37 = 0; i37 < m; i37++) {
    SD->u1.f5.x_data[i37] = SD->u1.f5.x_data[i37] / 100.0 / 0.01 + 1.0;
  }

  /* 'CreateHeightMap:21' y=[round(pc2(:,2)*100)./100]/Ny+1; */
  y_size[0] = na;
  for (i37 = 0; i37 < na; i37++) {
    SD->u1.f5.y_data[i37] = SD->u1.f5.pc2_data[1 + 3 * i37] * 100.0;
  }

  b_round(SD->u1.f5.y_data, y_size);
  m = y_size[0];
  for (i37 = 0; i37 < m; i37++) {
    SD->u1.f5.y_data[i37] = SD->u1.f5.y_data[i37] / 100.0 / 0.01 + 1.0;
  }

  /* 'CreateHeightMap:22' z=round(pc2(:,3)/Nz); */
  z_size[0] = na;
  for (i37 = 0; i37 < na; i37++) {
    SD->u1.f5.z_data[i37] = SD->u1.f5.pc2_data[2 + 3 * i37] / 0.01;
  }

  b_round(SD->u1.f5.z_data, z_size);

  /* 'CreateHeightMap:23' imgDims = int32(ceil([max(y) max(x)])); */
  y_data.data = &SD->u1.f5.y_data[0];
  y_data.size = &y_size[0];
  y_data.allocatedSize = 504000;
  y_data.numDimensions = 1;
  y_data.canFreeData = false;
  d5 = minOrMaxRealFloatVector(&y_data);
  x_data.data = &SD->u1.f5.x_data[0];
  x_data.size = &x_size[0];
  x_data.allocatedSize = 504000;
  x_data.numDimensions = 1;
  x_data.canFreeData = false;
  minx = minOrMaxRealFloatVector(&x_data);

  /* 'CreateHeightMap:24' if any(imgDims == 0) */
  i37 = (int)std::ceil(d5);
  idx = i37;
  imgDims[0] = (i37 == 0);
  i37 = (int)std::ceil(minx);
  imgDims[1] = (i37 == 0);
  if (any(imgDims)) {
    /* 'CreateHeightMap:25' smartLoaderStruct.loaderYawAngleDeg = 0; */
    smartLoaderStruct->loaderYawAngleDeg = 0.0;

    /* 'CreateHeightMap:26' smartLoaderStruct.loaderYawAngleStatus = false; */
    smartLoaderStruct->loaderYawAngleStatus = false;

    /* 'CreateHeightMap:28' heightMap_res = zeros(0,0,'single'); */
    heightMap_res_size[1] = 0;
    heightMap_res_size[0] = 0;

    /* 'CreateHeightMap:30' if coder.target('Matlab') */
  } else {
    emxInit_real32_T(&img, 2);

    /* 'CreateHeightMap:36' img=zeros(imgDims,'single'); */
    i38 = img->size[0] * img->size[1];
    img->size[1] = i37;
    img->size[0] = idx;
    emxEnsureCapacity_real32_T(img, i38);
    m = i37 * idx;
    for (i37 = 0; i37 < m; i37++) {
      img->data[i37] = 0.0F;
    }

    /* 'CreateHeightMap:37' linearInd = sub2ind(size(img),y, x); */
    /* 'CreateHeightMap:38' pc3=[round(x) round(y) z]; */
    b_round(SD->u1.f5.x_data, x_size);
    b_round(SD->u1.f5.y_data, y_size);
    na = x_size[0];
    idx = y_size[0];
    for (i37 = 0; i37 < na; i37++) {
      SD->u1.f5.pc2_data[3 * i37] = SD->u1.f5.x_data[i37];
    }

    for (i37 = 0; i37 < idx; i37++) {
      SD->u1.f5.pc2_data[1 + 3 * i37] = SD->u1.f5.y_data[i37];
    }

    m = z_size[0];
    for (i37 = 0; i37 < m; i37++) {
      SD->u1.f5.pc2_data[2 + 3 * i37] = SD->u1.f5.z_data[i37];
    }

    /*  Fill the image */
    /* 'CreateHeightMap:41' for i=1:length(pc3) */
    if (na > 3) {
      idx = na;
    } else {
      idx = 3;
    }

    if (na == 0) {
      i37 = 0;
    } else {
      i37 = idx;
    }

    for (idx = 0; idx < i37; idx++) {
      /* 'CreateHeightMap:42' img(pc3(i,2),pc3(i,1))=pc3(i,3); */
      img->data[((int)SD->u1.f5.pc2_data[3 * idx] + img->size[1] * ((int)
                  SD->u1.f5.pc2_data[1 + 3 * idx] - 1)) - 1] = (float)
        SD->u1.f5.pc2_data[2 + 3 * idx];
    }

    emxInit_real32_T(&temp, 2);

    /*  figure, imagesc(img) */
    /*  Dilation */
    /* 'CreateHeightMap:47' temp=img; */
    /* 'CreateHeightMap:48' temp=imdilate(temp,ones(10,2)); */
    imdilate(img, temp);

    /* 'CreateHeightMap:49' temp=imdilate(temp,ones(1,3)); */
    i37 = img->size[0] * img->size[1];
    img->size[1] = temp->size[1];
    img->size[0] = temp->size[0];
    emxEnsureCapacity_real32_T(img, i37);
    m = temp->size[1] * temp->size[0];
    for (i37 = 0; i37 < m; i37++) {
      img->data[i37] = temp->data[i37];
    }

    b_imdilate(img, temp);

    /* 'CreateHeightMap:50' temp=imdilate(temp,ones(3,1)); */
    i37 = img->size[0] * img->size[1];
    img->size[1] = temp->size[1];
    img->size[0] = temp->size[0];
    emxEnsureCapacity_real32_T(img, i37);
    m = temp->size[1] * temp->size[0];
    for (i37 = 0; i37 < m; i37++) {
      img->data[i37] = temp->data[i37];
    }

    c_imdilate(img, temp);

    /* 'CreateHeightMap:51' heightMap_res=imdilate(temp,ones(2,2)); */
    d_imdilate(temp, img);
    heightMap_res_size[1] = img->size[1];
    heightMap_res_size[0] = img->size[0];
    m = img->size[1] * img->size[0];
    emxFree_real32_T(&temp);
    for (i37 = 0; i37 < m; i37++) {
      heightMap_res_data[i37] = img->data[i37];
    }

    /*  figure, imagesc(heightMap_res) */
    /* 'CreateHeightMap:54' a=heightMap_res(50:end-50,:); */
    if (50 > img->size[0] - 50) {
      i37 = -1;
      i38 = 1;
    } else {
      i37 = 48;
      i38 = img->size[0] - 49;
    }

    /*  figure, imagesc(a) */
    /* 'CreateHeightMap:56' refmap=repmat(min(a,[],1),size(heightMap_res,1),1); */
    m = i38 - i37;
    i38 = img->size[1];
    idx = img->size[1];
    if (idx >= 1) {
      for (na = 0; na < i38; na++) {
        minval_data[na] = img->data[na + img->size[1] * (i37 + 1)];
      }

      for (idx = 2; idx <= m - 2; idx++) {
        for (na = 0; na < i38; na++) {
          b = img->data[na + img->size[1] * (i37 + idx)];
          if ((!rtIsNaNF(b)) && (rtIsNaNF(minval_data[na]) || (minval_data[na] >
                b))) {
            minval_data[na] = img->data[na + img->size[1] * (i37 + idx)];
          }
        }
      }
    }

    outsize_idx_0 = (short)img->size[0];
    if ((outsize_idx_0 != 0) && ((short)i38 != 0)) {
      i37 = img->size[0] - 1;
      for (idx = 0; idx <= i37; idx++) {
        na = (short)i38;
        for (m = 0; m < na; m++) {
          SD->u1.f5.refmap_data[m + (short)i38 * idx] = minval_data[m];
        }
      }
    }

    /*  figure, imagesc(refmap) */
    /* 'CreateHeightMap:58' bin_onlyloader=imerode((heightMap_res-refmap)>10,ones(3)); */
    idx = img->size[1];
    na = img->size[0];
    m = img->size[1] * img->size[0];
    for (i37 = 0; i37 < m; i37++) {
      SD->u1.f5.B_data[i37] = (img->data[i37] - SD->u1.f5.refmap_data[i37] >
        10.0F);
    }

    emxFree_real32_T(&img);
    B_size[1] = idx;
    B_size[0] = na;
    for (i37 = 0; i37 < 9; i37++) {
      nhood[i37] = true;
    }

    asizeT[0] = idx;
    asizeT[1] = na;
    nsizeT[0] = 3.0;
    nsizeT[1] = 3.0;
    erode_binary_ones33_tbb(&SD->u1.f5.B_data[0], asizeT, 2.0, nhood, nsizeT,
      2.0, &SD->u1.f5.b_B_data[0]);

    /*  figure, imagesc(bin_onlyloader) */
    /* 'CreateHeightMap:60' bin_onlyloader=imerode(bin_onlyloader,ones(3)); */
    for (i37 = 0; i37 < 9; i37++) {
      nhood[i37] = true;
    }

    asizeT[0] = idx;
    asizeT[1] = na;
    b_nsizeT[0] = 3.0;
    b_nsizeT[1] = 3.0;
    erode_binary_ones33_tbb(&SD->u1.f5.b_B_data[0], asizeT, 2.0, nhood, b_nsizeT,
      2.0, &SD->u1.f5.B_data[0]);

    /* 'CreateHeightMap:61' bin_onlyloader=imerode(bin_onlyloader,ones(3)); */
    for (i37 = 0; i37 < 9; i37++) {
      nhood[i37] = true;
    }

    asizeT[0] = idx;
    asizeT[1] = na;
    c_nsizeT[0] = 3.0;
    c_nsizeT[1] = 3.0;
    erode_binary_ones33_tbb(&SD->u1.f5.B_data[0], asizeT, 2.0, nhood, c_nsizeT,
      2.0, &SD->u1.f5.b_B_data[0]);

    /* 'CreateHeightMap:62' bin_onlyloader=imdilate(bin_onlyloader,ones(3)); */
    for (i37 = 0; i37 < 9; i37++) {
      nhood[i37] = true;
    }

    asizeT[0] = idx;
    asizeT[1] = na;
    d_nsizeT[0] = 3.0;
    d_nsizeT[1] = 3.0;
    dilate_binary_ones33_tbb(&SD->u1.f5.b_B_data[0], asizeT, 2.0, nhood,
      d_nsizeT, 2.0, &SD->u1.f5.B_data[0]);

    /* 'CreateHeightMap:63' bin_onlyloader=imdilate(bin_onlyloader,ones(3)); */
    for (i37 = 0; i37 < 9; i37++) {
      nhood[i37] = true;
    }

    emxInit_struct_T1(&datablob, 1);
    asizeT[0] = idx;
    asizeT[1] = na;
    e_nsizeT[0] = 3.0;
    e_nsizeT[1] = 3.0;
    dilate_binary_ones33_tbb(&SD->u1.f5.B_data[0], asizeT, 2.0, nhood, e_nsizeT,
      2.0, &SD->u1.f5.b_B_data[0]);

    /*  figure, imagesc(bin_onlyloader) */
    /*  Determine the loader yaw angle */
    /* 'CreateHeightMap:67' datablob = regionprops(bwlabel(bin_onlyloader),'Orientation','area','MajorAxisLength','MinorAxisLength','Centroid'); */
    bwlabel(SD->u1.f5.b_B_data, B_size, SD->u1.f5.tmp_data, tmp_size);
    regionprops(SD->u1.f5.tmp_data, tmp_size, datablob);

    /* 'CreateHeightMap:68' if isempty(datablob) */
    if (datablob->size[0] == 0) {
      /* 'CreateHeightMap:69' smartLoaderStruct.loaderYawAngleDeg = 0; */
      smartLoaderStruct->loaderYawAngleDeg = 0.0;

      /* 'CreateHeightMap:70' smartLoaderStruct.loaderYawAngleStatus = false; */
      smartLoaderStruct->loaderYawAngleStatus = false;

      /* 'CreateHeightMap:71' if coder.target('Matlab') */
    } else if (datablob->size[0] == 1) {
      /* 'CreateHeightMap:75' elseif numel(datablob) == 1 */
      /* 'CreateHeightMap:76' smartLoaderStruct.loaderYawAngleDeg = datablob(1).Orientation; */
      smartLoaderStruct->loaderYawAngleDeg = datablob->data[0].Orientation;

      /* 'CreateHeightMap:77' smartLoaderStruct.loaderYawAngleStatus = true; */
      smartLoaderStruct->loaderYawAngleStatus = true;

      /* 'CreateHeightMap:79' if coder.target('Matlab') */
    } else {
      emxInit_real_T(&areaArr, 2);

      /* 'CreateHeightMap:82' else */
      /*   numel(datablob) > 1 */
      /*  pick the blob with the largest size */
      /*  [~,ind] = max([datablob.Area]); --> is not supported in matlab coder, we'll write down the command that way:  */
      /* 'CreateHeightMap:86' areaArr = zeros(1,numel(datablob)); */
      i37 = areaArr->size[0] * areaArr->size[1];
      areaArr->size[1] = datablob->size[0];
      areaArr->size[0] = 1;
      emxEnsureCapacity_real_T(areaArr, i37);
      m = datablob->size[0];
      for (i37 = 0; i37 < m; i37++) {
        areaArr->data[i37] = 0.0;
      }

      /* 'CreateHeightMap:87' for i = 1:numel(datablob) */
      i37 = datablob->size[0];
      for (idx = 0; idx < i37; idx++) {
        /* 'CreateHeightMap:88' areaArr(i) = datablob(i).Area; */
        areaArr->data[idx] = datablob->data[idx].Area;
      }

      /* 'CreateHeightMap:90' [~,ind] = max(areaArr); */
      na = areaArr->size[1];
      if (areaArr->size[1] <= 2) {
        if ((areaArr->data[0] < areaArr->data[1]) || (rtIsNaN(areaArr->data[0]) &&
             (!rtIsNaN(areaArr->data[1])))) {
          idx = 2;
        } else {
          idx = 1;
        }
      } else {
        if (!rtIsNaN(areaArr->data[0])) {
          idx = 1;
        } else {
          idx = 0;
          m = 2;
          exitg1 = false;
          while ((!exitg1) && (m <= areaArr->size[1])) {
            if (!rtIsNaN(areaArr->data[m - 1])) {
              idx = m;
              exitg1 = true;
            } else {
              m++;
            }
          }
        }

        if (idx == 0) {
          idx = 1;
        } else {
          minx = areaArr->data[idx - 1];
          i37 = idx + 1;
          for (m = i37; m <= na; m++) {
            if (minx < areaArr->data[m - 1]) {
              minx = areaArr->data[m - 1];
              idx = m;
            }
          }
        }
      }

      emxFree_real_T(&areaArr);

      /* 'CreateHeightMap:90' ~ */
      /* 'CreateHeightMap:92' smartLoaderStruct.loaderYawAngleDeg = datablob(ind).Orientation; */
      smartLoaderStruct->loaderYawAngleDeg = datablob->data[idx - 1].Orientation;

      /* 'CreateHeightMap:93' smartLoaderStruct.loaderYawAngleStatus = true; */
      smartLoaderStruct->loaderYawAngleStatus = true;

      /* 'CreateHeightMap:95' if coder.target('Matlab') */
    }

    emxFree_struct_T1(&datablob);

    /*  */
    /* 'CreateHeightMap:102' if false && coder.target('Matlab') */
  }
}

/*
 * function [pcFiltered] = FilterPointCloudAccordingToZdifferences(pc, diffThreshold)
 */
static void FilterPointCloudAccordingToZdifferences(SmartLoaderStackData *SD,
  const double pc_data[], const int pc_size[2], double diffThreshold, double
  pcFiltered_data[], int pcFiltered_size[2])
{
  int loop_ub;
  int i;
  int iv1[1];
  double zMedian;
  int tmp_size[1];
  int b_tmp_size[1];
  int trueCount;
  int partialTrueCount;

  /* FILTERPOINTCLOUDACCORDINGTOZDIFFERENCES Summary of this function goes here */
  /*    Detailed explanation goes here */
  /* 'FilterPointCloudAccordingToZdifferences:5' isMatlab2019B = false; */
  /* zCor = pc.Location(:,3); */
  /* 'FilterPointCloudAccordingToZdifferences:8' zCor = pc(:,3); */
  loop_ub = pc_size[0];
  for (i = 0; i < loop_ub; i++) {
    SD->u3.f9.tmp_data[i] = pc_data[2 + 3 * i];
  }

  /* 'FilterPointCloudAccordingToZdifferences:9' zMedian = median(zCor); */
  iv1[0] = pc_size[0];
  zMedian = median(SD, SD->u3.f9.tmp_data, iv1);

  /* 'FilterPointCloudAccordingToZdifferences:10' assert(numel(zMedian) == 1); */
  /* 'FilterPointCloudAccordingToZdifferences:11' inliearsInd = abs(zCor - zMedian) <= diffThreshold; */
  loop_ub = pc_size[0];
  tmp_size[0] = pc_size[0];
  for (i = 0; i < loop_ub; i++) {
    SD->u3.f9.b_tmp_data[i] = SD->u3.f9.tmp_data[i] - zMedian;
  }

  b_abs(SD->u3.f9.b_tmp_data, tmp_size, SD->u3.f9.tmp_data, b_tmp_size);
  loop_ub = b_tmp_size[0];
  for (i = 0; i < loop_ub; i++) {
    SD->u3.f9.inliearsInd_data[i] = (SD->u3.f9.tmp_data[i] <= diffThreshold);
  }

  /*  sum(loaderReflectorPtrInd) */
  /* 'FilterPointCloudAccordingToZdifferences:14' if isMatlab2019B */
  /* 'FilterPointCloudAccordingToZdifferences:16' else */
  /* pcFiltered = select(pc, find(inliearsInd)); */
  /* 'FilterPointCloudAccordingToZdifferences:18' pcFiltered = pc(inliearsInd, :); */
  loop_ub = b_tmp_size[0] - 1;
  trueCount = 0;
  for (i = 0; i <= loop_ub; i++) {
    if (SD->u3.f9.inliearsInd_data[i]) {
      trueCount++;
    }
  }

  partialTrueCount = 0;
  for (i = 0; i <= loop_ub; i++) {
    if (SD->u3.f9.inliearsInd_data[i]) {
      SD->u3.f9.c_tmp_data[partialTrueCount] = i + 1;
      partialTrueCount++;
    }
  }

  pcFiltered_size[1] = 3;
  pcFiltered_size[0] = trueCount;
  for (i = 0; i < trueCount; i++) {
    loop_ub = 3 * (SD->u3.f9.c_tmp_data[i] - 1);
    pcFiltered_data[3 * i] = pc_data[loop_ub];
    pcFiltered_data[1 + 3 * i] = pc_data[1 + loop_ub];
    pcFiltered_data[2 + 3 * i] = pc_data[2 + loop_ub];
  }

  /*  figure, PlotPointCloud(pc); */
  /*  figure, PlotPointCloud(pcFiltered); */
}

/*
 * function [SmartLoaderGlobalStruct] = SmartLoaderGlobalInit
 */
static void SmartLoaderGlobalInit(SmartLoaderStackData *SD)
{
  /* 'SmartLoaderGlobalInit:7' percisionMode = 'double'; */
  /*  % boolean represent whether or not the smart loader global has been initialized */
  /*  % N*3 vector that hold the history of the loader location */
  /*  % N*1 vector that hold the history of the loader location's time tag */
  /* 'SmartLoaderGlobalInit:10' SmartLoaderGlobalStruct = struct('isInitialized', true, ... % boolean represent whether or not the smart loader global has been initialized */
  /* 'SmartLoaderGlobalInit:11'     'loaderLocHistory', zeros(0,3,percisionMode), ... % N*3 vector that hold the history of the loader location */
  /* 'SmartLoaderGlobalInit:12'     'loaderTimeTatHistoryMs', zeros(0,1,'uint64') ... % N*1 vector that hold the history of the loader location's time tag */
  /* 'SmartLoaderGlobalInit:13'     ); */
  /* 'SmartLoaderGlobalInit:15' SmartLoaderGlobal = SmartLoaderGlobalStruct; */
  SD->pd->SmartLoaderGlobal.isInitialized = true;
  SD->pd->SmartLoaderGlobal.loaderLocHistory.size[1] = 3;
  SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] = 0;
  SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] = 0;
}

/*
 *
 */
static boolean_T all(const boolean_T x[2])
{
  boolean_T y;
  int k;
  boolean_T exitg1;
  y = true;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 2)) {
    if (!x[k]) {
      y = false;
      exitg1 = true;
    } else {
      k++;
    }
  }

  return y;
}

/*
 *
 */
static boolean_T any(const boolean_T x[2])
{
  boolean_T y;
  int k;
  boolean_T exitg1;
  y = false;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 2)) {
    if (x[k]) {
      y = true;
      exitg1 = true;
    } else {
      k++;
    }
  }

  return y;
}

/*
 *
 */
static void b_abs(const double x_data[], const int x_size[1], double y_data[],
                  int y_size[1])
{
  int i23;
  int k;
  y_size[0] = x_size[0];
  if (x_size[0] != 0) {
    i23 = x_size[0];
    for (k = 0; k < i23; k++) {
      y_data[k] = std::abs(x_data[k]);
    }
  }
}

/*
 *
 */
static int b_bsearch(const double x_data[], const int x_size[1], double xi)
{
  int n;
  int high_i;
  int low_ip1;
  int mid_i;
  high_i = x_size[0];
  n = 1;
  low_ip1 = 2;
  while (high_i > low_ip1) {
    mid_i = (n >> 1) + (high_i >> 1);
    if (((n & 1) == 1) && ((high_i & 1) == 1)) {
      mid_i++;
    }

    if (xi >= x_data[mid_i - 1]) {
      n = mid_i;
      low_ip1 = mid_i + 1;
    } else {
      high_i = mid_i;
    }
  }

  return n;
}

/*
 *
 */
static void b_distfun(double D_data[], const double X_data[], const int X_size[2],
                      const double C[6], const int crows[2], int ncrows)
{
  int n;
  int i;
  int cr;
  int r;
  int i52;
  n = X_size[0] - 1;
  for (i = 0; i < ncrows; i++) {
    cr = crows[i] - 1;
    for (r = 0; r <= n; r++) {
      D_data[cr + (r << 1)] = rt_powd_snf(X_data[3 * r] - C[3 * (crows[i] - 1)],
        2.0);
    }

    for (r = 0; r <= n; r++) {
      i52 = cr + (r << 1);
      D_data[i52] += rt_powd_snf(X_data[3 * r + 1] - C[3 * cr + 1], 2.0);
    }

    for (r = 0; r <= n; r++) {
      D_data[cr + (r << 1)] += rt_powd_snf(X_data[3 * r + 2] - C[3 * cr + 2],
        2.0);
    }
  }
}

/*
 *
 */
static void b_gcentroids(double C[6], int counts[2], const double X_data[],
  const int X_size[2], const int idx_data[], int clusters)
{
  int n;
  int i56;
  int i57;
  int i58;
  int cc;
  int i;
  n = X_size[0];
  counts[clusters - 1] = 0;
  i56 = 3 * (clusters - 1);
  C[i56] = rtNaN;
  i57 = 1 + i56;
  C[i57] = rtNaN;
  i58 = 2 + i56;
  C[i58] = rtNaN;
  cc = 0;
  C[i56] = 0.0;
  C[i57] = 0.0;
  C[i58] = 0.0;
  for (i = 0; i < n; i++) {
    if (idx_data[i] == clusters) {
      cc++;
      C[i56] += X_data[3 * i];
      C[i57] += X_data[1 + 3 * i];
      C[i58] += X_data[2 + 3 * i];
    }
  }

  counts[clusters - 1] = cc;
  C[i56] /= (double)cc;
  C[i57] /= (double)cc;
  C[i58] /= (double)cc;
}

/*
 *
 */
static void b_imdilate(const emxArray_real32_T *A, emxArray_real32_T *B)
{
  int i8;
  boolean_T is2DInput;
  double asizeT[2];
  boolean_T nhood[3];
  int tmp;
  double nsizeT[2];
  int end;
  boolean_T b2;
  boolean_T b3;
  int i9;
  int i;
  emxArray_int32_T *r8;
  int i10;
  int i11;
  emxArray_int32_T *r9;
  emxArray_int32_T *r10;
  i8 = B->size[0] * B->size[1];
  B->size[1] = A->size[1];
  B->size[0] = A->size[0];
  emxEnsureCapacity_real32_T(B, i8);
  is2DInput = ((A->size[0] != 0) && (A->size[1] != 0));
  if (is2DInput) {
    asizeT[0] = A->size[0];
    asizeT[1] = A->size[1];
    nhood[0] = true;
    nhood[1] = true;
    nhood[2] = true;
    tmp = (int)asizeT[0];
    asizeT[0] = asizeT[1];
    asizeT[1] = tmp;
    nsizeT[0] = 3.0;
    nsizeT[1] = 1.0;
    dilate_real32_ocv(&A->data[0], asizeT, nhood, nsizeT, &B->data[0]);
    end = B->size[0] * B->size[1] - 1;
    tmp = 0;
    b2 = true;
    b3 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i8 = B->size[1] * B->size[0];
    i9 = 0;
    for (i = 0; i <= end; i++) {
      if (b3 || (i >= i8)) {
        i9 = 0;
        b2 = true;
      } else if (b2) {
        b2 = false;
        i9 = B->size[1];
        i10 = B->size[0];
        i9 = i % i10 * i9 + i / i10;
      } else {
        i10 = B->size[1];
        i11 = i10 * B->size[0] - 1;
        if (i9 > MAX_int32_T - i10) {
          i9 = B->size[1];
          i10 = B->size[0];
          i9 = i % i10 * i9 + i / i10;
        } else {
          i9 += i10;
          if (i9 > i11) {
            i9 -= i11;
          }
        }
      }

      if (B->data[i9] >= 3.402823466E+38F) {
        tmp++;
      }
    }

    emxInit_int32_T(&r8, 1);
    i8 = r8->size[0];
    r8->size[0] = tmp;
    emxEnsureCapacity_int32_T(r8, i8);
    tmp = 0;
    b2 = true;
    b3 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i8 = B->size[1] * B->size[0];
    i9 = 0;
    for (i = 0; i <= end; i++) {
      if (b3 || (i >= i8)) {
        i9 = 0;
        b2 = true;
      } else if (b2) {
        b2 = false;
        i9 = B->size[1];
        i10 = B->size[0];
        i9 = i % i10 * i9 + i / i10;
      } else {
        i10 = B->size[1];
        i11 = i10 * B->size[0] - 1;
        if (i9 > MAX_int32_T - i10) {
          i9 = B->size[1];
          i10 = B->size[0];
          i9 = i % i10 * i9 + i / i10;
        } else {
          i9 += i10;
          if (i9 > i11) {
            i9 -= i11;
          }
        }
      }

      if (B->data[i9] >= 3.402823466E+38F) {
        r8->data[tmp] = i + 1;
        tmp++;
      }
    }

    emxInit_int32_T(&r9, 1);
    i8 = r9->size[0];
    r9->size[0] = r8->size[0];
    emxEnsureCapacity_int32_T(r9, i8);
    tmp = r8->size[0];
    for (i8 = 0; i8 < tmp; i8++) {
      r9->data[i8] = r8->data[i8] - 1;
    }

    i8 = B->size[1];
    i9 = B->size[0];
    tmp = r8->size[0] - 1;
    for (i10 = 0; i10 <= tmp; i10++) {
      B->data[r9->data[i10] % i9 * i8 + r9->data[i10] / i9] = rtInfF;
    }

    emxFree_int32_T(&r9);
    end = B->size[0] * B->size[1] - 1;
    tmp = 0;
    b2 = true;
    b3 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i8 = B->size[1] * B->size[0];
    i9 = 0;
    for (i = 0; i <= end; i++) {
      if (b3 || (i >= i8)) {
        i9 = 0;
        b2 = true;
      } else if (b2) {
        b2 = false;
        i9 = B->size[1];
        i10 = B->size[0];
        i9 = i % i10 * i9 + i / i10;
      } else {
        i10 = B->size[1];
        i11 = i10 * B->size[0] - 1;
        if (i9 > MAX_int32_T - i10) {
          i9 = B->size[1];
          i10 = B->size[0];
          i9 = i % i10 * i9 + i / i10;
        } else {
          i9 += i10;
          if (i9 > i11) {
            i9 -= i11;
          }
        }
      }

      if (B->data[i9] <= 1.17549435E-38F) {
        tmp++;
      }
    }

    emxInit_int32_T(&r10, 1);
    i8 = r10->size[0];
    r10->size[0] = tmp;
    emxEnsureCapacity_int32_T(r10, i8);
    tmp = 0;
    b2 = true;
    b3 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i8 = B->size[1] * B->size[0];
    i9 = 0;
    for (i = 0; i <= end; i++) {
      if (b3 || (i >= i8)) {
        i9 = 0;
        b2 = true;
      } else if (b2) {
        b2 = false;
        i9 = B->size[1];
        i10 = B->size[0];
        i9 = i % i10 * i9 + i / i10;
      } else {
        i10 = B->size[1];
        i11 = i10 * B->size[0] - 1;
        if (i9 > MAX_int32_T - i10) {
          i9 = B->size[1];
          i10 = B->size[0];
          i9 = i % i10 * i9 + i / i10;
        } else {
          i9 += i10;
          if (i9 > i11) {
            i9 -= i11;
          }
        }
      }

      if (B->data[i9] <= 1.17549435E-38F) {
        r10->data[tmp] = i + 1;
        tmp++;
      }
    }

    i8 = r8->size[0];
    r8->size[0] = r10->size[0];
    emxEnsureCapacity_int32_T(r8, i8);
    tmp = r10->size[0];
    for (i8 = 0; i8 < tmp; i8++) {
      r8->data[i8] = r10->data[i8] - 1;
    }

    i8 = B->size[1];
    i9 = B->size[0];
    tmp = r10->size[0] - 1;
    emxFree_int32_T(&r10);
    for (i10 = 0; i10 <= tmp; i10++) {
      B->data[r8->data[i10] % i9 * i8 + r8->data[i10] / i9] = rtMinusInfF;
    }

    emxFree_int32_T(&r8);
  } else {
    asizeT[0] = A->size[0];
    asizeT[1] = A->size[1];
    nhood[0] = true;
    nhood[1] = true;
    nhood[2] = true;
    tmp = (int)asizeT[0];
    asizeT[0] = asizeT[1];
    asizeT[1] = tmp;
    nsizeT[0] = 3.0;
    nsizeT[1] = 1.0;
    dilate_flat_real32_tbb(&A->data[0], asizeT, 2.0, nhood, nsizeT, 2.0,
      &B->data[0]);
  }
}

/*
 *
 */
static boolean_T b_isfinite(double x)
{
  return (!rtIsInf(x)) && (!rtIsNaN(x));
}

/*
 *
 */
static double b_norm(const double x[3])
{
  double y;
  double scale;
  double absxk;
  double t;
  scale = 3.3121686421112381E-170;
  absxk = std::abs(x[0]);
  if (absxk > 3.3121686421112381E-170) {
    y = 1.0;
    scale = absxk;
  } else {
    t = absxk / 3.3121686421112381E-170;
    y = t * t;
  }

  absxk = std::abs(x[1]);
  if (absxk > scale) {
    t = scale / absxk;
    y = 1.0 + y * t * t;
    scale = absxk;
  } else {
    t = absxk / scale;
    y += t * t;
  }

  absxk = std::abs(x[2]);
  if (absxk > scale) {
    t = scale / absxk;
    y = 1.0 + y * t * t;
    scale = absxk;
  } else {
    t = absxk / scale;
    y += t * t;
  }

  return scale * std::sqrt(y);
}

/*
 *
 */
static void b_nullAssignment(SmartLoaderStackData *SD, double x_data[], int
  x_size[2], const int idx_data[])
{
  int nrows;
  int i62;
  int i;
  nrows = x_size[0] - 1;
  i62 = idx_data[0];
  for (i = i62; i <= nrows; i++) {
    x_data[(i - 1) << 1] = x_data[i << 1];
  }

  for (i = i62; i <= nrows; i++) {
    x_data[1 + ((i - 1) << 1)] = x_data[1 + (i << 1)];
  }

  if (1 > nrows) {
    i = 0;
  } else {
    i = x_size[0] - 1;
  }

  for (i62 = 0; i62 < i; i62++) {
    nrows = i62 << 1;
    SD->u1.f0.x_data[nrows] = x_data[nrows];
    nrows++;
    SD->u1.f0.x_data[nrows] = x_data[nrows];
  }

  x_size[1] = 2;
  x_size[0] = i;
  for (i62 = 0; i62 < i; i62++) {
    nrows = i62 << 1;
    x_data[nrows] = SD->u1.f0.x_data[nrows];
    nrows++;
    x_data[nrows] = SD->u1.f0.x_data[nrows];
  }
}

/*
 *
 */
static double b_rand(SmartLoaderStackData *SD)
{
  double r;
  int j;
  unsigned int u[2];
  unsigned int mti;
  int kk;
  unsigned int y;

  /* ========================= COPYRIGHT NOTICE ============================ */
  /*  This is a uniform (0,1) pseudorandom number generator based on:        */
  /*                                                                         */
  /*  A C-program for MT19937, with initialization improved 2002/1/26.       */
  /*  Coded by Takuji Nishimura and Makoto Matsumoto.                        */
  /*                                                                         */
  /*  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,      */
  /*  All rights reserved.                                                   */
  /*                                                                         */
  /*  Redistribution and use in source and binary forms, with or without     */
  /*  modification, are permitted provided that the following conditions     */
  /*  are met:                                                               */
  /*                                                                         */
  /*    1. Redistributions of source code must retain the above copyright    */
  /*       notice, this list of conditions and the following disclaimer.     */
  /*                                                                         */
  /*    2. Redistributions in binary form must reproduce the above copyright */
  /*       notice, this list of conditions and the following disclaimer      */
  /*       in the documentation and/or other materials provided with the     */
  /*       distribution.                                                     */
  /*                                                                         */
  /*    3. The names of its contributors may not be used to endorse or       */
  /*       promote products derived from this software without specific      */
  /*       prior written permission.                                         */
  /*                                                                         */
  /*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS    */
  /*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT      */
  /*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  */
  /*  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT  */
  /*  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,  */
  /*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT       */
  /*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  */
  /*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY  */
  /*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT    */
  /*  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE */
  /*  OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */
  /*                                                                         */
  /* =============================   END   ================================= */
  do {
    for (j = 0; j < 2; j++) {
      mti = SD->pd->state[624] + 1U;
      if (mti >= 625U) {
        for (kk = 0; kk < 227; kk++) {
          y = (SD->pd->state[kk] & 2147483648U) | (SD->pd->state[kk + 1] &
            2147483647U);
          if ((y & 1U) == 0U) {
            y >>= 1U;
          } else {
            y = y >> 1U ^ 2567483615U;
          }

          SD->pd->state[kk] = SD->pd->state[kk + 397] ^ y;
        }

        for (kk = 0; kk < 396; kk++) {
          y = (SD->pd->state[kk + 227] & 2147483648U) | (SD->pd->state[kk + 228]
            & 2147483647U);
          if ((y & 1U) == 0U) {
            y >>= 1U;
          } else {
            y = y >> 1U ^ 2567483615U;
          }

          SD->pd->state[kk + 227] = SD->pd->state[kk] ^ y;
        }

        y = (SD->pd->state[623] & 2147483648U) | (SD->pd->state[0] & 2147483647U);
        if ((y & 1U) == 0U) {
          y >>= 1U;
        } else {
          y = y >> 1U ^ 2567483615U;
        }

        SD->pd->state[623] = SD->pd->state[396] ^ y;
        mti = 1U;
      }

      y = SD->pd->state[(int)mti - 1];
      SD->pd->state[624] = mti;
      y ^= y >> 11U;
      y ^= y << 7U & 2636928640U;
      y ^= y << 15U & 4022730752U;
      u[j] = y ^ y >> 18U;
    }

    u[0] >>= 5U;
    u[1] >>= 6U;
    r = 1.1102230246251565E-16 * ((double)u[0] * 6.7108864E+7 + (double)u[1]);
  } while (r == 0.0);

  return r;
}

/*
 *
 */
static void b_repmat(const double a[2], double varargin_1, double b_data[], int
                     b_size[2])
{
  int b_size_tmp_tmp;
  int b_size_tmp;
  int b_data_tmp;
  b_size[1] = 2;
  b_size_tmp_tmp = (int)varargin_1;
  b_size_tmp = (signed char)b_size_tmp_tmp;
  b_size[0] = b_size_tmp;
  if (b_size_tmp != 0) {
    b_size_tmp_tmp--;
    for (b_size_tmp = 0; b_size_tmp <= b_size_tmp_tmp; b_size_tmp++) {
      b_data_tmp = b_size_tmp << 1;
      b_data[b_data_tmp] = a[0];
      b_data[1 + b_data_tmp] = a[1];
    }
  }
}

/*
 *
 */
static void b_round(double x_data[], int x_size[1])
{
  int i39;
  int k;
  i39 = x_size[0];
  for (k = 0; k < i39; k++) {
    x_data[k] = rt_roundd_snf(x_data[k]);
  }
}

/*
 *
 */
static void b_sort(SmartLoaderStackData *SD, double x_data[], int x_size[1], int
                   idx_data[], int idx_size[1])
{
  int dim;
  int vwork_size_idx_0;
  int vlen;
  int vstride;
  int k;
  emxArray_int32_T *iidx;
  emxArray_real_T *vwork;
  int j;
  int i61;
  dim = 0;
  if (x_size[0] != 1) {
    dim = -1;
  }

  if (dim + 2 <= 1) {
    vwork_size_idx_0 = x_size[0];
  } else {
    vwork_size_idx_0 = 1;
  }

  vlen = vwork_size_idx_0 - 1;
  idx_size[0] = x_size[0];
  vstride = 1;
  for (k = 0; k <= dim; k++) {
    vstride *= x_size[0];
  }

  emxInit_int32_T(&iidx, 1);
  emxInit_real_T(&vwork, 1);
  for (j = 0; j < vstride; j++) {
    for (k = 0; k <= vlen; k++) {
      SD->u1.f3.vwork_data[k] = x_data[j + k * vstride];
    }

    i61 = vwork->size[0];
    vwork->size[0] = vwork_size_idx_0;
    emxEnsureCapacity_real_T(vwork, i61);
    for (i61 = 0; i61 < vwork_size_idx_0; i61++) {
      vwork->data[i61] = SD->u1.f3.vwork_data[i61];
    }

    sortIdx(vwork, iidx);
    vwork_size_idx_0 = vwork->size[0];
    dim = vwork->size[0];
    for (i61 = 0; i61 < dim; i61++) {
      SD->u1.f3.vwork_data[i61] = vwork->data[i61];
    }

    for (k = 0; k <= vlen; k++) {
      i61 = j + k * vstride;
      x_data[i61] = vwork->data[k];
      idx_data[i61] = iidx->data[k];
    }
  }

  emxFree_real_T(&vwork);
  emxFree_int32_T(&iidx);
}

/*
 *
 */
static void b_sqrt(double x_data[], int x_size[1])
{
  int i63;
  int k;
  int i64;
  i63 = x_size[0];
  for (k = 0; k < i63; k++) {
    i64 = (signed char)(1 + k) - 1;
    x_data[i64] = std::sqrt(x_data[i64]);
  }
}

/*
 *
 */
static double b_sum(const emxArray_real_T *x)
{
  double y;
  if (x->size[0] == 0) {
    y = 0.0;
  } else {
    y = nestedIter(x, x->size[0]);
  }

  return y;
}

/*
 *
 */
static void batchUpdate(SmartLoaderStackData *SD, const double X_data[], const
  int X_size[2], int idx_data[], int idx_size[1], double C[6], double D_data[],
  int D_size[2], int counts[2], boolean_T *converged, int *iter)
{
  int n;
  int empties[2];
  int previdx_size_idx_0;
  int moved_size[1];
  int changed[2];
  int nchanged;
  double prevtotsumD;
  int exitg1;
  int nempty;
  int i;
  double maxd;
  int lonely;
  int nMoved;
  int from;
  double d10;
  boolean_T exitg2;
  int d_size[1];
  int nidx_size[1];
  n = X_size[0] - 1;
  empties[0] = 0;
  empties[1] = 0;
  previdx_size_idx_0 = X_size[0];
  if (0 <= X_size[0] - 1) {
    memset(&SD->u2.f8.previdx_data[0], 0, (unsigned int)(X_size[0] * (int)sizeof
            (int)));
  }

  moved_size[0] = X_size[0];
  if (0 <= X_size[0] - 1) {
    memset(&SD->u2.f8.moved_data[0], 0, (unsigned int)(X_size[0] * (int)sizeof
            (int)));
  }

  changed[0] = 1;
  changed[1] = 2;
  nchanged = 2;
  prevtotsumD = rtInf;
  *iter = 0;
  *converged = false;
  do {
    exitg1 = 0;
    (*iter)++;
    gcentroids(C, counts, X_data, X_size, idx_data, changed, nchanged);
    b_distfun(D_data, X_data, X_size, C, changed, nchanged);
    nempty = countEmpty(empties, counts, changed, nchanged);
    if (nempty > 0) {
      for (i = 0; i < nempty; i++) {
        maxd = D_data[idx_data[0] - 1];
        lonely = 0;
        for (nMoved = 0; nMoved <= n; nMoved++) {
          d10 = D_data[(idx_data[nMoved] + (nMoved << 1)) - 1];
          if (d10 > maxd) {
            maxd = d10;
            lonely = nMoved;
          }
        }

        from = idx_data[lonely];
        if (counts[idx_data[lonely] - 1] < 2) {
          nMoved = 0;
          exitg2 = false;
          while ((!exitg2) && (nMoved <= n)) {
            if (counts[nMoved] > 1) {
              from = nMoved + 1;
              exitg2 = true;
            } else {
              nMoved++;
            }
          }

          nMoved = 0;
          exitg2 = false;
          while ((!exitg2) && (nMoved <= n)) {
            if (idx_data[nMoved] == from) {
              lonely = nMoved;
              exitg2 = true;
            } else {
              nMoved++;
            }
          }
        }

        nMoved = 3 * (empties[i] - 1);
        C[nMoved] = X_data[3 * lonely];
        C[1 + nMoved] = X_data[1 + 3 * lonely];
        C[2 + nMoved] = X_data[2 + 3 * lonely];
        counts[empties[i] - 1] = 1;
        idx_data[lonely] = empties[i];
        distfun(D_data, X_data, X_size, C, empties[i]);
        b_gcentroids(C, counts, X_data, X_size, idx_data, from);
        distfun(D_data, X_data, X_size, C, from);
        if (nchanged < 2) {
          nMoved = 0;
          exitg2 = false;
          while ((!exitg2) && ((nMoved <= nchanged - 1) && (from != changed[0])))
          {
            if (from > changed[0]) {
              if (nchanged >= 1) {
                changed[1] = 1;
              }

              changed[0] = 2;
              nchanged++;
              exitg2 = true;
            } else {
              nMoved++;
            }
          }
        }
      }
    }

    maxd = 0.0;
    for (i = 0; i <= n; i++) {
      maxd += D_data[(idx_data[i] + (i << 1)) - 1];
    }

    if (prevtotsumD <= maxd) {
      idx_size[0] = previdx_size_idx_0;
      if (0 <= previdx_size_idx_0 - 1) {
        memcpy(&idx_data[0], &SD->u2.f8.previdx_data[0], (unsigned int)
               (previdx_size_idx_0 * (int)sizeof(int)));
      }

      gcentroids(C, counts, X_data, X_size, SD->u2.f8.previdx_data, changed,
                 nchanged);
      (*iter)--;
      exitg1 = 1;
    } else if (*iter >= 100) {
      exitg1 = 1;
    } else {
      previdx_size_idx_0 = idx_size[0];
      nMoved = idx_size[0];
      if (0 <= nMoved - 1) {
        memcpy(&SD->u2.f8.previdx_data[0], &idx_data[0], (unsigned int)(nMoved *
                (int)sizeof(int)));
      }

      prevtotsumD = maxd;
      mindim2(D_data, D_size, SD->u2.f8.d_data, d_size, SD->u2.f8.nidx_data,
              nidx_size);
      nMoved = 0;
      for (i = 0; i <= n; i++) {
        if ((SD->u2.f8.nidx_data[i] != SD->u2.f8.previdx_data[i]) && (D_data
             [(SD->u2.f8.previdx_data[i] + (i << 1)) - 1] > SD->u2.f8.d_data[i]))
        {
          nMoved++;
          SD->u2.f8.moved_data[nMoved - 1] = i + 1;
          idx_data[i] = SD->u2.f8.nidx_data[i];
        }
      }

      if (nMoved == 0) {
        *converged = true;
        exitg1 = 1;
      } else {
        nchanged = findchanged(SD, changed, idx_data, SD->u2.f8.previdx_data,
          SD->u2.f8.moved_data, moved_size, nMoved);
      }
    }
  } while (exitg1 == 0);
}

/*
 *
 */
static void bwlabel(const boolean_T varargin_1_data[], const int
                    varargin_1_size[2], double L_data[], int L_size[2])
{
  int numRuns;
  int firstRunOnThisColumn;
  emxArray_int16_T *startRow;
  int lastRunOnPreviousColumn;
  emxArray_int16_T *endRow;
  emxArray_int16_T *startCol;
  emxArray_int32_T *labelForEachRun;
  int runCounter;
  int k;
  int currentColumn;
  int row;
  emxArray_int32_T *labelsRenumbered;
  double numComponents;
  int p;
  int root_k;
  int root_p;
  numRuns = 0;
  if ((varargin_1_size[0] != 0) && (varargin_1_size[1] != 0)) {
    firstRunOnThisColumn = varargin_1_size[1];
    for (lastRunOnPreviousColumn = 0; lastRunOnPreviousColumn <
         firstRunOnThisColumn; lastRunOnPreviousColumn++) {
      if (varargin_1_data[lastRunOnPreviousColumn]) {
        numRuns++;
      }

      runCounter = varargin_1_size[0];
      for (k = 0; k <= runCounter - 2; k++) {
        if (varargin_1_data[lastRunOnPreviousColumn + varargin_1_size[1] * (1 +
             k)] && (!varargin_1_data[lastRunOnPreviousColumn + varargin_1_size
                     [1] * k])) {
          numRuns++;
        }
      }
    }
  }

  emxInit_int16_T(&startRow, 1);
  emxInit_int16_T(&endRow, 1);
  emxInit_int16_T(&startCol, 1);
  emxInit_int32_T(&labelForEachRun, 1);
  if (numRuns == 0) {
    startRow->size[0] = 0;
    endRow->size[0] = 0;
    startCol->size[0] = 0;
    labelForEachRun->size[0] = 0;
    L_size[1] = (short)varargin_1_size[1];
    L_size[0] = (short)varargin_1_size[0];
    currentColumn = (short)varargin_1_size[1] * (short)varargin_1_size[0];
    if (0 <= currentColumn - 1) {
      memset(&L_data[0], 0, (unsigned int)(currentColumn * (int)sizeof(double)));
    }
  } else {
    firstRunOnThisColumn = startRow->size[0];
    startRow->size[0] = numRuns;
    emxEnsureCapacity_int16_T(startRow, firstRunOnThisColumn);
    firstRunOnThisColumn = endRow->size[0];
    endRow->size[0] = numRuns;
    emxEnsureCapacity_int16_T(endRow, firstRunOnThisColumn);
    firstRunOnThisColumn = startCol->size[0];
    startCol->size[0] = numRuns;
    emxEnsureCapacity_int16_T(startCol, firstRunOnThisColumn);
    currentColumn = varargin_1_size[0];
    runCounter = 0;
    firstRunOnThisColumn = varargin_1_size[1];
    for (lastRunOnPreviousColumn = 0; lastRunOnPreviousColumn <
         firstRunOnThisColumn; lastRunOnPreviousColumn++) {
      row = 1;
      while (row <= currentColumn) {
        while ((row <= currentColumn) &&
               (!varargin_1_data[lastRunOnPreviousColumn + varargin_1_size[1] *
                (row - 1)])) {
          row++;
        }

        if ((row <= currentColumn) && varargin_1_data[lastRunOnPreviousColumn +
            varargin_1_size[1] * (row - 1)]) {
          startCol->data[runCounter] = (short)(lastRunOnPreviousColumn + 1);
          startRow->data[runCounter] = (short)row;
          while ((row <= currentColumn) &&
                 varargin_1_data[lastRunOnPreviousColumn + varargin_1_size[1] *
                 (row - 1)]) {
            row++;
          }

          endRow->data[runCounter] = (short)(row - 1);
          runCounter++;
        }
      }
    }

    firstRunOnThisColumn = labelForEachRun->size[0];
    labelForEachRun->size[0] = numRuns;
    emxEnsureCapacity_int32_T(labelForEachRun, firstRunOnThisColumn);
    for (firstRunOnThisColumn = 0; firstRunOnThisColumn < numRuns;
         firstRunOnThisColumn++) {
      labelForEachRun->data[firstRunOnThisColumn] = 0;
    }

    k = 0;
    currentColumn = 1;
    runCounter = 1;
    row = -1;
    lastRunOnPreviousColumn = -1;
    firstRunOnThisColumn = 0;
    while (k + 1 <= numRuns) {
      if (startCol->data[k] == currentColumn + 1) {
        row = firstRunOnThisColumn + 1;
        firstRunOnThisColumn = k;
        lastRunOnPreviousColumn = k;
        currentColumn = startCol->data[k];
      } else {
        if (startCol->data[k] > currentColumn + 1) {
          row = -1;
          lastRunOnPreviousColumn = -1;
          firstRunOnThisColumn = k;
          currentColumn = startCol->data[k];
        }
      }

      if (row >= 0) {
        for (p = row - 1; p < lastRunOnPreviousColumn; p++) {
          if ((endRow->data[k] >= startRow->data[p] - 1) && (startRow->data[k] <=
               endRow->data[p] + 1)) {
            if (labelForEachRun->data[k] == 0) {
              labelForEachRun->data[k] = labelForEachRun->data[p];
              runCounter++;
            } else {
              if (labelForEachRun->data[k] != labelForEachRun->data[p]) {
                for (root_k = k; root_k + 1 != labelForEachRun->data[root_k];
                     root_k = labelForEachRun->data[root_k] - 1) {
                  labelForEachRun->data[root_k] = labelForEachRun->
                    data[labelForEachRun->data[root_k] - 1];
                }

                for (root_p = p; root_p + 1 != labelForEachRun->data[root_p];
                     root_p = labelForEachRun->data[root_p] - 1) {
                  labelForEachRun->data[root_p] = labelForEachRun->
                    data[labelForEachRun->data[root_p] - 1];
                }

                if (root_k + 1 != root_p + 1) {
                  if (root_p + 1 < root_k + 1) {
                    labelForEachRun->data[root_k] = root_p + 1;
                    labelForEachRun->data[k] = root_p + 1;
                  } else {
                    labelForEachRun->data[root_p] = root_k + 1;
                    labelForEachRun->data[p] = root_k + 1;
                  }
                }
              }
            }
          }
        }
      }

      if (labelForEachRun->data[k] == 0) {
        labelForEachRun->data[k] = runCounter;
        runCounter++;
      }

      k++;
    }

    emxInit_int32_T(&labelsRenumbered, 1);
    firstRunOnThisColumn = labelsRenumbered->size[0];
    labelsRenumbered->size[0] = labelForEachRun->size[0];
    emxEnsureCapacity_int32_T(labelsRenumbered, firstRunOnThisColumn);
    numComponents = 0.0;
    L_size[1] = (short)varargin_1_size[1];
    L_size[0] = (short)varargin_1_size[0];
    currentColumn = (short)varargin_1_size[1] * (short)varargin_1_size[0];
    if (0 <= currentColumn - 1) {
      memset(&L_data[0], 0, (unsigned int)(currentColumn * (int)sizeof(double)));
    }

    for (k = 0; k < numRuns; k++) {
      if (labelForEachRun->data[k] == k + 1) {
        numComponents++;
        labelsRenumbered->data[k] = (int)numComponents;
      }

      labelsRenumbered->data[k] = labelsRenumbered->data[labelForEachRun->data[k]
        - 1];
      firstRunOnThisColumn = startRow->data[k];
      runCounter = endRow->data[k];
      for (currentColumn = firstRunOnThisColumn; currentColumn <= runCounter;
           currentColumn++) {
        L_data[(startCol->data[k] + L_size[1] * (currentColumn - 1)) - 1] =
          labelsRenumbered->data[k];
      }
    }

    emxFree_int32_T(&labelsRenumbered);
  }

  emxFree_int32_T(&labelForEachRun);
  emxFree_int16_T(&startCol);
  emxFree_int16_T(&endRow);
  emxFree_int16_T(&startRow);
}

/*
 *
 */
static void c_imdilate(const emxArray_real32_T *A, emxArray_real32_T *B)
{
  int i12;
  boolean_T is2DInput;
  double asizeT[2];
  boolean_T nhood[3];
  int tmp;
  double nsizeT[2];
  int end;
  boolean_T b4;
  boolean_T b5;
  int i13;
  int i;
  emxArray_int32_T *r11;
  int i14;
  int i15;
  emxArray_int32_T *r12;
  emxArray_int32_T *r13;
  i12 = B->size[0] * B->size[1];
  B->size[1] = A->size[1];
  B->size[0] = A->size[0];
  emxEnsureCapacity_real32_T(B, i12);
  is2DInput = ((A->size[0] != 0) && (A->size[1] != 0));
  if (is2DInput) {
    asizeT[0] = A->size[0];
    asizeT[1] = A->size[1];
    nhood[0] = true;
    nhood[1] = true;
    nhood[2] = true;
    tmp = (int)asizeT[0];
    asizeT[0] = asizeT[1];
    asizeT[1] = tmp;
    nsizeT[0] = 1.0;
    nsizeT[1] = 3.0;
    dilate_real32_ocv(&A->data[0], asizeT, nhood, nsizeT, &B->data[0]);
    end = B->size[0] * B->size[1] - 1;
    tmp = 0;
    b4 = true;
    b5 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i12 = B->size[1] * B->size[0];
    i13 = 0;
    for (i = 0; i <= end; i++) {
      if (b5 || (i >= i12)) {
        i13 = 0;
        b4 = true;
      } else if (b4) {
        b4 = false;
        i13 = B->size[1];
        i14 = B->size[0];
        i13 = i % i14 * i13 + i / i14;
      } else {
        i14 = B->size[1];
        i15 = i14 * B->size[0] - 1;
        if (i13 > MAX_int32_T - i14) {
          i13 = B->size[1];
          i14 = B->size[0];
          i13 = i % i14 * i13 + i / i14;
        } else {
          i13 += i14;
          if (i13 > i15) {
            i13 -= i15;
          }
        }
      }

      if (B->data[i13] >= 3.402823466E+38F) {
        tmp++;
      }
    }

    emxInit_int32_T(&r11, 1);
    i12 = r11->size[0];
    r11->size[0] = tmp;
    emxEnsureCapacity_int32_T(r11, i12);
    tmp = 0;
    b4 = true;
    b5 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i12 = B->size[1] * B->size[0];
    i13 = 0;
    for (i = 0; i <= end; i++) {
      if (b5 || (i >= i12)) {
        i13 = 0;
        b4 = true;
      } else if (b4) {
        b4 = false;
        i13 = B->size[1];
        i14 = B->size[0];
        i13 = i % i14 * i13 + i / i14;
      } else {
        i14 = B->size[1];
        i15 = i14 * B->size[0] - 1;
        if (i13 > MAX_int32_T - i14) {
          i13 = B->size[1];
          i14 = B->size[0];
          i13 = i % i14 * i13 + i / i14;
        } else {
          i13 += i14;
          if (i13 > i15) {
            i13 -= i15;
          }
        }
      }

      if (B->data[i13] >= 3.402823466E+38F) {
        r11->data[tmp] = i + 1;
        tmp++;
      }
    }

    emxInit_int32_T(&r12, 1);
    i12 = r12->size[0];
    r12->size[0] = r11->size[0];
    emxEnsureCapacity_int32_T(r12, i12);
    tmp = r11->size[0];
    for (i12 = 0; i12 < tmp; i12++) {
      r12->data[i12] = r11->data[i12] - 1;
    }

    i12 = B->size[1];
    i13 = B->size[0];
    tmp = r11->size[0] - 1;
    for (i14 = 0; i14 <= tmp; i14++) {
      B->data[r12->data[i14] % i13 * i12 + r12->data[i14] / i13] = rtInfF;
    }

    emxFree_int32_T(&r12);
    end = B->size[0] * B->size[1] - 1;
    tmp = 0;
    b4 = true;
    b5 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i12 = B->size[1] * B->size[0];
    i13 = 0;
    for (i = 0; i <= end; i++) {
      if (b5 || (i >= i12)) {
        i13 = 0;
        b4 = true;
      } else if (b4) {
        b4 = false;
        i13 = B->size[1];
        i14 = B->size[0];
        i13 = i % i14 * i13 + i / i14;
      } else {
        i14 = B->size[1];
        i15 = i14 * B->size[0] - 1;
        if (i13 > MAX_int32_T - i14) {
          i13 = B->size[1];
          i14 = B->size[0];
          i13 = i % i14 * i13 + i / i14;
        } else {
          i13 += i14;
          if (i13 > i15) {
            i13 -= i15;
          }
        }
      }

      if (B->data[i13] <= 1.17549435E-38F) {
        tmp++;
      }
    }

    emxInit_int32_T(&r13, 1);
    i12 = r13->size[0];
    r13->size[0] = tmp;
    emxEnsureCapacity_int32_T(r13, i12);
    tmp = 0;
    b4 = true;
    b5 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i12 = B->size[1] * B->size[0];
    i13 = 0;
    for (i = 0; i <= end; i++) {
      if (b5 || (i >= i12)) {
        i13 = 0;
        b4 = true;
      } else if (b4) {
        b4 = false;
        i13 = B->size[1];
        i14 = B->size[0];
        i13 = i % i14 * i13 + i / i14;
      } else {
        i14 = B->size[1];
        i15 = i14 * B->size[0] - 1;
        if (i13 > MAX_int32_T - i14) {
          i13 = B->size[1];
          i14 = B->size[0];
          i13 = i % i14 * i13 + i / i14;
        } else {
          i13 += i14;
          if (i13 > i15) {
            i13 -= i15;
          }
        }
      }

      if (B->data[i13] <= 1.17549435E-38F) {
        r13->data[tmp] = i + 1;
        tmp++;
      }
    }

    i12 = r11->size[0];
    r11->size[0] = r13->size[0];
    emxEnsureCapacity_int32_T(r11, i12);
    tmp = r13->size[0];
    for (i12 = 0; i12 < tmp; i12++) {
      r11->data[i12] = r13->data[i12] - 1;
    }

    i12 = B->size[1];
    i13 = B->size[0];
    tmp = r13->size[0] - 1;
    emxFree_int32_T(&r13);
    for (i14 = 0; i14 <= tmp; i14++) {
      B->data[r11->data[i14] % i13 * i12 + r11->data[i14] / i13] = rtMinusInfF;
    }

    emxFree_int32_T(&r11);
  } else {
    asizeT[0] = A->size[0];
    asizeT[1] = A->size[1];
    nhood[0] = true;
    nhood[1] = true;
    nhood[2] = true;
    tmp = (int)asizeT[0];
    asizeT[0] = asizeT[1];
    asizeT[1] = tmp;
    nsizeT[0] = 1.0;
    nsizeT[1] = 3.0;
    dilate_flat_real32_tbb(&A->data[0], asizeT, 2.0, nhood, nsizeT, 2.0,
      &B->data[0]);
  }
}

/*
 *
 */
static void c_sum(const double x_data[], const int x_size[2], double y_data[],
                  int y_size[1])
{
  int i36;
  int k;
  int y_data_tmp;
  int b_y_data_tmp;
  if (x_size[0] == 0) {
    y_size[0] = 0;
  } else {
    y_size[0] = (signed char)x_size[0];
    i36 = x_size[0];
    for (k = 0; k < i36; k++) {
      y_data_tmp = k << 1;
      y_data[k] = x_data[y_data_tmp];
      b_y_data_tmp = (signed char)(k + 1) - 1;
      y_data[b_y_data_tmp] += x_data[1 + y_data_tmp];
    }
  }
}

/*
 *
 */
static int countEmpty(int empties[2], const int counts[2], const int changed[2],
                      int nchanged)
{
  int nempty;
  int j;
  nempty = 0;
  for (j = 0; j < nchanged; j++) {
    if (counts[changed[j] - 1] == 0) {
      nempty++;
      empties[nempty - 1] = changed[j];
    }
  }

  return nempty;
}

/*
 *
 */
static void d_imdilate(const emxArray_real32_T *A, emxArray_real32_T *B)
{
  int i16;
  boolean_T is2DInput;
  double asizeT[2];
  boolean_T nhood[4];
  int tmp;
  double nsizeT[2];
  int end;
  boolean_T b6;
  boolean_T b7;
  int i17;
  int i;
  emxArray_int32_T *r14;
  int i18;
  int i19;
  emxArray_int32_T *r15;
  emxArray_int32_T *r16;
  i16 = B->size[0] * B->size[1];
  B->size[1] = A->size[1];
  B->size[0] = A->size[0];
  emxEnsureCapacity_real32_T(B, i16);
  is2DInput = ((A->size[0] != 0) && (A->size[1] != 0));
  if (is2DInput) {
    asizeT[0] = A->size[0];
    asizeT[1] = A->size[1];
    nhood[0] = true;
    nhood[1] = true;
    nhood[2] = true;
    nhood[3] = true;
    tmp = (int)asizeT[0];
    asizeT[0] = asizeT[1];
    asizeT[1] = tmp;
    nsizeT[0] = 2.0;
    nsizeT[1] = 2.0;
    dilate_real32_ocv(&A->data[0], asizeT, nhood, nsizeT, &B->data[0]);
    end = B->size[0] * B->size[1] - 1;
    tmp = 0;
    b6 = true;
    b7 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i16 = B->size[1] * B->size[0];
    i17 = 0;
    for (i = 0; i <= end; i++) {
      if (b7 || (i >= i16)) {
        i17 = 0;
        b6 = true;
      } else if (b6) {
        b6 = false;
        i17 = B->size[1];
        i18 = B->size[0];
        i17 = i % i18 * i17 + i / i18;
      } else {
        i18 = B->size[1];
        i19 = i18 * B->size[0] - 1;
        if (i17 > MAX_int32_T - i18) {
          i17 = B->size[1];
          i18 = B->size[0];
          i17 = i % i18 * i17 + i / i18;
        } else {
          i17 += i18;
          if (i17 > i19) {
            i17 -= i19;
          }
        }
      }

      if (B->data[i17] >= 3.402823466E+38F) {
        tmp++;
      }
    }

    emxInit_int32_T(&r14, 1);
    i16 = r14->size[0];
    r14->size[0] = tmp;
    emxEnsureCapacity_int32_T(r14, i16);
    tmp = 0;
    b6 = true;
    b7 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i16 = B->size[1] * B->size[0];
    i17 = 0;
    for (i = 0; i <= end; i++) {
      if (b7 || (i >= i16)) {
        i17 = 0;
        b6 = true;
      } else if (b6) {
        b6 = false;
        i17 = B->size[1];
        i18 = B->size[0];
        i17 = i % i18 * i17 + i / i18;
      } else {
        i18 = B->size[1];
        i19 = i18 * B->size[0] - 1;
        if (i17 > MAX_int32_T - i18) {
          i17 = B->size[1];
          i18 = B->size[0];
          i17 = i % i18 * i17 + i / i18;
        } else {
          i17 += i18;
          if (i17 > i19) {
            i17 -= i19;
          }
        }
      }

      if (B->data[i17] >= 3.402823466E+38F) {
        r14->data[tmp] = i + 1;
        tmp++;
      }
    }

    emxInit_int32_T(&r15, 1);
    i16 = r15->size[0];
    r15->size[0] = r14->size[0];
    emxEnsureCapacity_int32_T(r15, i16);
    tmp = r14->size[0];
    for (i16 = 0; i16 < tmp; i16++) {
      r15->data[i16] = r14->data[i16] - 1;
    }

    i16 = B->size[1];
    i17 = B->size[0];
    tmp = r14->size[0] - 1;
    for (i18 = 0; i18 <= tmp; i18++) {
      B->data[r15->data[i18] % i17 * i16 + r15->data[i18] / i17] = rtInfF;
    }

    emxFree_int32_T(&r15);
    end = B->size[0] * B->size[1] - 1;
    tmp = 0;
    b6 = true;
    b7 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i16 = B->size[1] * B->size[0];
    i17 = 0;
    for (i = 0; i <= end; i++) {
      if (b7 || (i >= i16)) {
        i17 = 0;
        b6 = true;
      } else if (b6) {
        b6 = false;
        i17 = B->size[1];
        i18 = B->size[0];
        i17 = i % i18 * i17 + i / i18;
      } else {
        i18 = B->size[1];
        i19 = i18 * B->size[0] - 1;
        if (i17 > MAX_int32_T - i18) {
          i17 = B->size[1];
          i18 = B->size[0];
          i17 = i % i18 * i17 + i / i18;
        } else {
          i17 += i18;
          if (i17 > i19) {
            i17 -= i19;
          }
        }
      }

      if (B->data[i17] <= 1.17549435E-38F) {
        tmp++;
      }
    }

    emxInit_int32_T(&r16, 1);
    i16 = r16->size[0];
    r16->size[0] = tmp;
    emxEnsureCapacity_int32_T(r16, i16);
    tmp = 0;
    b6 = true;
    b7 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i16 = B->size[1] * B->size[0];
    i17 = 0;
    for (i = 0; i <= end; i++) {
      if (b7 || (i >= i16)) {
        i17 = 0;
        b6 = true;
      } else if (b6) {
        b6 = false;
        i17 = B->size[1];
        i18 = B->size[0];
        i17 = i % i18 * i17 + i / i18;
      } else {
        i18 = B->size[1];
        i19 = i18 * B->size[0] - 1;
        if (i17 > MAX_int32_T - i18) {
          i17 = B->size[1];
          i18 = B->size[0];
          i17 = i % i18 * i17 + i / i18;
        } else {
          i17 += i18;
          if (i17 > i19) {
            i17 -= i19;
          }
        }
      }

      if (B->data[i17] <= 1.17549435E-38F) {
        r16->data[tmp] = i + 1;
        tmp++;
      }
    }

    i16 = r14->size[0];
    r14->size[0] = r16->size[0];
    emxEnsureCapacity_int32_T(r14, i16);
    tmp = r16->size[0];
    for (i16 = 0; i16 < tmp; i16++) {
      r14->data[i16] = r16->data[i16] - 1;
    }

    i16 = B->size[1];
    i17 = B->size[0];
    tmp = r16->size[0] - 1;
    emxFree_int32_T(&r16);
    for (i18 = 0; i18 <= tmp; i18++) {
      B->data[r14->data[i18] % i17 * i16 + r14->data[i18] / i17] = rtMinusInfF;
    }

    emxFree_int32_T(&r14);
  } else {
    asizeT[0] = A->size[0];
    asizeT[1] = A->size[1];
    nhood[0] = true;
    nhood[1] = true;
    nhood[2] = true;
    nhood[3] = true;
    tmp = (int)asizeT[0];
    asizeT[0] = asizeT[1];
    asizeT[1] = tmp;
    nsizeT[0] = 2.0;
    nsizeT[1] = 2.0;
    dilate_flat_real32_tbb(&A->data[0], asizeT, 2.0, nhood, nsizeT, 2.0,
      &B->data[0]);
  }
}

/*
 *
 */
static void distfun(double D_data[], const double X_data[], const int X_size[2],
                    const double C[6], int crows)
{
  int n;
  int r;
  int i51;
  n = X_size[0] - 1;
  for (r = 0; r <= n; r++) {
    D_data[(crows + (r << 1)) - 1] = rt_powd_snf(X_data[3 * r] - C[3 * (crows -
      1)], 2.0);
  }

  for (r = 0; r <= n; r++) {
    i51 = (crows + (r << 1)) - 1;
    D_data[i51] += rt_powd_snf(X_data[3 * r + 1] - C[3 * (crows - 1) + 1], 2.0);
  }

  for (r = 0; r <= n; r++) {
    D_data[(crows + (r << 1)) - 1] += rt_powd_snf(X_data[3 * r + 2] - C[3 *
      (crows - 1) + 2], 2.0);
  }
}

static int div_s32(int numerator, int denominator)
{
  int quotient;
  unsigned int b_numerator;
  unsigned int b_denominator;
  unsigned int tempAbsQuotient;
  if (denominator == 0) {
    if (numerator >= 0) {
      quotient = MAX_int32_T;
    } else {
      quotient = MIN_int32_T;
    }
  } else {
    if (numerator < 0) {
      b_numerator = ~(unsigned int)numerator + 1U;
    } else {
      b_numerator = (unsigned int)numerator;
    }

    if (denominator < 0) {
      b_denominator = ~(unsigned int)denominator + 1U;
    } else {
      b_denominator = (unsigned int)denominator;
    }

    tempAbsQuotient = b_numerator / b_denominator;
    if ((numerator < 0) != (denominator < 0)) {
      quotient = -(int)tempAbsQuotient;
    } else {
      quotient = (int)tempAbsQuotient;
    }
  }

  return quotient;
}

static void eml_rand_mt19937ar_stateful_init(SmartLoaderStackData *SD)
{
  unsigned int r;
  int mti;
  memset(&SD->pd->state[0], 0, 625U * sizeof(unsigned int));
  r = 5489U;
  SD->pd->state[0] = 5489U;
  for (mti = 0; mti < 623; mti++) {
    r = ((r ^ r >> 30U) * 1812433253U + mti) + 1U;
    SD->pd->state[mti + 1] = r;
  }

  SD->pd->state[624] = 624U;
}

/*
 *
 */
static int findchanged(SmartLoaderStackData *SD, int changed[2], const int
  idx_data[], const int previdx_data[], const int moved_data[], const int
  moved_size[1], int nmoved)
{
  int nchanged;
  int j;
  int i59;
  if (0 <= moved_size[0] - 1) {
    memset(&SD->u1.f4.b_data[0], 0, (unsigned int)(moved_size[0] * (int)sizeof
            (boolean_T)));
  }

  for (j = 0; j < nmoved; j++) {
    SD->u1.f4.b_data[idx_data[moved_data[j] - 1] - 1] = true;
    SD->u1.f4.b_data[previdx_data[moved_data[j] - 1] - 1] = true;
  }

  nchanged = 0;
  i59 = moved_size[0];
  for (j = 0; j < i59; j++) {
    if (SD->u1.f4.b_data[j]) {
      nchanged++;
      changed[nchanged - 1] = j + 1;
    }
  }

  return nchanged;
}

/*
 *
 */
static void gcentroids(double C[6], int counts[2], const double X_data[], const
  int X_size[2], const int idx_data[], const int clusters[2], int nclusters)
{
  int n;
  int ic;
  int i53;
  int cc;
  int i54;
  int i55;
  int i;
  n = X_size[0];
  for (ic = 0; ic < nclusters; ic++) {
    counts[clusters[ic] - 1] = 0;
    i53 = 3 * (clusters[ic] - 1);
    C[i53] = rtNaN;
    C[1 + i53] = rtNaN;
    C[2 + i53] = rtNaN;
  }

  for (ic = 0; ic < nclusters; ic++) {
    cc = 0;
    i53 = 3 * (clusters[ic] - 1);
    C[i53] = 0.0;
    i54 = 1 + i53;
    C[i54] = 0.0;
    i55 = 2 + i53;
    C[i55] = 0.0;
    for (i = 0; i < n; i++) {
      if (idx_data[i] == clusters[ic]) {
        cc++;
        C[i53] += X_data[3 * i];
        C[i54] += X_data[1 + 3 * i];
        C[i55] += X_data[2 + 3 * i];
      }
    }

    counts[clusters[ic] - 1] = cc;
    C[i53] /= (double)cc;
    C[i54] /= (double)cc;
    C[i55] /= (double)cc;
  }
}

/*
 *
 */
static boolean_T ifWhileCond(const boolean_T x_data[])
{
  boolean_T y;
  y = true;
  if (!x_data[0]) {
    y = false;
  }

  return y;
}

/*
 *
 */
static void imdilate(const emxArray_real32_T *A, emxArray_real32_T *B)
{
  int i4;
  boolean_T is2DInput;
  int tmp;
  int k;
  double asizeT[2];
  boolean_T nhood[20];
  double nsizeT[2];
  boolean_T b0;
  boolean_T b1;
  int i5;
  int i;
  emxArray_int32_T *r5;
  int i6;
  int i7;
  emxArray_int32_T *r6;
  emxArray_int32_T *r7;
  i4 = B->size[0] * B->size[1];
  B->size[1] = A->size[1];
  B->size[0] = A->size[0];
  emxEnsureCapacity_real32_T(B, i4);
  is2DInput = ((A->size[0] != 0) && (A->size[1] != 0));
  tmp = 0;
  for (k = 0; k < 20; k++) {
    tmp++;
  }

  if (is2DInput && (!((double)tmp / 20.0 < 0.05))) {
    asizeT[0] = A->size[0];
    asizeT[1] = A->size[1];
    for (i4 = 0; i4 < 20; i4++) {
      nhood[i4] = true;
    }

    tmp = (int)asizeT[0];
    asizeT[0] = asizeT[1];
    asizeT[1] = tmp;
    nsizeT[0] = 2.0;
    nsizeT[1] = 10.0;
    dilate_real32_ocv(&A->data[0], asizeT, nhood, nsizeT, &B->data[0]);
    k = B->size[0] * B->size[1] - 1;
    tmp = 0;
    b0 = true;
    b1 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i4 = B->size[1] * B->size[0];
    i5 = 0;
    for (i = 0; i <= k; i++) {
      if (b1 || (i >= i4)) {
        i5 = 0;
        b0 = true;
      } else if (b0) {
        b0 = false;
        i5 = B->size[1];
        i6 = B->size[0];
        i5 = i % i6 * i5 + i / i6;
      } else {
        i6 = B->size[1];
        i7 = i6 * B->size[0] - 1;
        if (i5 > MAX_int32_T - i6) {
          i5 = B->size[1];
          i6 = B->size[0];
          i5 = i % i6 * i5 + i / i6;
        } else {
          i5 += i6;
          if (i5 > i7) {
            i5 -= i7;
          }
        }
      }

      if (B->data[i5] >= 3.402823466E+38F) {
        tmp++;
      }
    }

    emxInit_int32_T(&r5, 1);
    i4 = r5->size[0];
    r5->size[0] = tmp;
    emxEnsureCapacity_int32_T(r5, i4);
    tmp = 0;
    b0 = true;
    b1 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i4 = B->size[1] * B->size[0];
    i5 = 0;
    for (i = 0; i <= k; i++) {
      if (b1 || (i >= i4)) {
        i5 = 0;
        b0 = true;
      } else if (b0) {
        b0 = false;
        i5 = B->size[1];
        i6 = B->size[0];
        i5 = i % i6 * i5 + i / i6;
      } else {
        i6 = B->size[1];
        i7 = i6 * B->size[0] - 1;
        if (i5 > MAX_int32_T - i6) {
          i5 = B->size[1];
          i6 = B->size[0];
          i5 = i % i6 * i5 + i / i6;
        } else {
          i5 += i6;
          if (i5 > i7) {
            i5 -= i7;
          }
        }
      }

      if (B->data[i5] >= 3.402823466E+38F) {
        r5->data[tmp] = i + 1;
        tmp++;
      }
    }

    emxInit_int32_T(&r6, 1);
    i4 = r6->size[0];
    r6->size[0] = r5->size[0];
    emxEnsureCapacity_int32_T(r6, i4);
    tmp = r5->size[0];
    for (i4 = 0; i4 < tmp; i4++) {
      r6->data[i4] = r5->data[i4] - 1;
    }

    i4 = B->size[1];
    i5 = B->size[0];
    tmp = r5->size[0] - 1;
    for (i6 = 0; i6 <= tmp; i6++) {
      B->data[r6->data[i6] % i5 * i4 + r6->data[i6] / i5] = rtInfF;
    }

    emxFree_int32_T(&r6);
    k = B->size[0] * B->size[1] - 1;
    tmp = 0;
    b0 = true;
    b1 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i4 = B->size[1] * B->size[0];
    i5 = 0;
    for (i = 0; i <= k; i++) {
      if (b1 || (i >= i4)) {
        i5 = 0;
        b0 = true;
      } else if (b0) {
        b0 = false;
        i5 = B->size[1];
        i6 = B->size[0];
        i5 = i % i6 * i5 + i / i6;
      } else {
        i6 = B->size[1];
        i7 = i6 * B->size[0] - 1;
        if (i5 > MAX_int32_T - i6) {
          i5 = B->size[1];
          i6 = B->size[0];
          i5 = i % i6 * i5 + i / i6;
        } else {
          i5 += i6;
          if (i5 > i7) {
            i5 -= i7;
          }
        }
      }

      if (B->data[i5] <= 1.17549435E-38F) {
        tmp++;
      }
    }

    emxInit_int32_T(&r7, 1);
    i4 = r7->size[0];
    r7->size[0] = tmp;
    emxEnsureCapacity_int32_T(r7, i4);
    tmp = 0;
    b0 = true;
    b1 = ((B->size[1] <= 0) || (B->size[0] <= 0));
    i4 = B->size[1] * B->size[0];
    i5 = 0;
    for (i = 0; i <= k; i++) {
      if (b1 || (i >= i4)) {
        i5 = 0;
        b0 = true;
      } else if (b0) {
        b0 = false;
        i5 = B->size[1];
        i6 = B->size[0];
        i5 = i % i6 * i5 + i / i6;
      } else {
        i6 = B->size[1];
        i7 = i6 * B->size[0] - 1;
        if (i5 > MAX_int32_T - i6) {
          i5 = B->size[1];
          i6 = B->size[0];
          i5 = i % i6 * i5 + i / i6;
        } else {
          i5 += i6;
          if (i5 > i7) {
            i5 -= i7;
          }
        }
      }

      if (B->data[i5] <= 1.17549435E-38F) {
        r7->data[tmp] = i + 1;
        tmp++;
      }
    }

    i4 = r5->size[0];
    r5->size[0] = r7->size[0];
    emxEnsureCapacity_int32_T(r5, i4);
    tmp = r7->size[0];
    for (i4 = 0; i4 < tmp; i4++) {
      r5->data[i4] = r7->data[i4] - 1;
    }

    i4 = B->size[1];
    i5 = B->size[0];
    tmp = r7->size[0] - 1;
    emxFree_int32_T(&r7);
    for (i6 = 0; i6 <= tmp; i6++) {
      B->data[r5->data[i6] % i5 * i4 + r5->data[i6] / i5] = rtMinusInfF;
    }

    emxFree_int32_T(&r5);
  } else {
    asizeT[0] = A->size[0];
    asizeT[1] = A->size[1];
    for (i4 = 0; i4 < 20; i4++) {
      nhood[i4] = true;
    }

    tmp = (int)asizeT[0];
    asizeT[0] = asizeT[1];
    asizeT[1] = tmp;
    nsizeT[0] = 2.0;
    nsizeT[1] = 10.0;
    dilate_flat_real32_tbb(&A->data[0], asizeT, 2.0, nhood, nsizeT, 2.0,
      &B->data[0]);
  }
}

/*
 *
 */
static void initializeStatsStruct(double numObjs, b_emxArray_struct_T *stats,
  struct_T *statsAlreadyComputed)
{
  d_struct_T statsOneObj;
  int i20;
  int loop_ub;
  emxInitStruct_struct_T1(&statsOneObj);
  statsAlreadyComputed->Area = false;
  statsOneObj.Area = 0.0;
  statsAlreadyComputed->Centroid = false;
  statsOneObj.Centroid[0] = 0.0;
  statsOneObj.Centroid[1] = 0.0;
  statsAlreadyComputed->BoundingBox = false;
  statsOneObj.BoundingBox[0] = 0.0;
  statsOneObj.BoundingBox[1] = 0.0;
  statsOneObj.BoundingBox[2] = 0.0;
  statsOneObj.BoundingBox[3] = 0.0;
  statsAlreadyComputed->MajorAxisLength = false;
  statsOneObj.MajorAxisLength = 0.0;
  statsAlreadyComputed->MinorAxisLength = false;
  statsOneObj.MinorAxisLength = 0.0;
  statsAlreadyComputed->Eccentricity = false;
  statsOneObj.Eccentricity = 0.0;
  statsAlreadyComputed->Orientation = false;
  statsOneObj.Orientation = 0.0;
  statsAlreadyComputed->Image = false;
  statsOneObj.Image->size[1] = 0;
  statsOneObj.Image->size[0] = 0;
  statsAlreadyComputed->FilledImage = false;
  statsOneObj.FilledImage->size[1] = 0;
  statsOneObj.FilledImage->size[0] = 0;
  statsAlreadyComputed->FilledArea = false;
  statsOneObj.FilledArea = 0.0;
  statsAlreadyComputed->EulerNumber = false;
  statsOneObj.EulerNumber = 0.0;
  statsAlreadyComputed->Extrema = false;
  memset(&statsOneObj.Extrema[0], 0, sizeof(double) << 4);
  statsAlreadyComputed->EquivDiameter = false;
  statsOneObj.EquivDiameter = 0.0;
  statsAlreadyComputed->Extent = false;
  statsOneObj.Extent = 0.0;
  statsAlreadyComputed->PixelIdxList = false;
  statsOneObj.PixelIdxList->size[0] = 0;
  statsAlreadyComputed->PixelList = false;
  statsOneObj.PixelList->size[1] = 2;
  statsOneObj.PixelList->size[0] = 0;
  statsAlreadyComputed->Perimeter = false;
  statsOneObj.Perimeter = 0.0;
  statsAlreadyComputed->PixelValues = false;
  statsOneObj.PixelValues->size[0] = 0;
  statsAlreadyComputed->WeightedCentroid = false;
  statsOneObj.WeightedCentroid[0] = 0.0;
  statsOneObj.WeightedCentroid[1] = 0.0;
  statsAlreadyComputed->MeanIntensity = false;
  statsOneObj.MeanIntensity = 0.0;
  statsAlreadyComputed->MinIntensity = false;
  statsOneObj.MinIntensity = 0.0;
  statsAlreadyComputed->MaxIntensity = false;
  statsOneObj.MaxIntensity = 0.0;
  statsAlreadyComputed->SubarrayIdx = false;
  statsOneObj.SubarrayIdx->size[1] = 0;
  statsOneObj.SubarrayIdx->size[0] = 1;
  statsOneObj.SubarrayIdxLengths[0] = 0.0;
  statsOneObj.SubarrayIdxLengths[1] = 0.0;
  i20 = stats->size[0];
  loop_ub = (int)numObjs;
  stats->size[0] = loop_ub;
  emxEnsureCapacity_struct_T1(stats, i20);
  for (i20 = 0; i20 < loop_ub; i20++) {
    emxCopyStruct_struct_T(&stats->data[i20], &statsOneObj);
  }

  emxFreeStruct_struct_T(&statsOneObj);
}

/*
 *
 */
static void inv(const double x[4], double y[4])
{
  double r;
  double t;
  if (std::abs(x[2]) > std::abs(x[0])) {
    r = x[0] / x[2];
    t = 1.0 / (r * x[3] - x[1]);
    y[0] = x[3] / x[2] * t;
    y[2] = -t;
    y[1] = -x[1] / x[2] * t;
    y[3] = r * t;
  } else {
    r = x[2] / x[0];
    t = 1.0 / (x[3] - r * x[1]);
    y[0] = x[3] / x[0] * t;
    y[2] = -r * t;
    y[1] = -x[1] / x[0] * t;
    y[3] = t;
  }
}

/*
 *
 */
static void kmeans(SmartLoaderStackData *SD, double X_data[], int X_size[2],
                   double idxbest_data[], int idxbest_size[1], double Cbest[6],
                   double varargout_1[2], double varargout_2_data[], int
                   varargout_2_size[2])
{
  int n;
  boolean_T hadnans;
  int i;
  int j;
  boolean_T exitg1;
  int idx_size[1];
  int trueCount;
  int partialTrueCount;
  n = X_size[0];
  if (0 <= X_size[0] - 1) {
    memset(&SD->f12.wasnan_data[0], 0, (unsigned int)(X_size[0] * (int)sizeof
            (boolean_T)));
  }

  hadnans = false;
  for (i = 0; i < n; i++) {
    j = 0;
    exitg1 = false;
    while ((!exitg1) && (j < 3)) {
      if (rtIsNaN(X_data[j + 3 * i])) {
        hadnans = true;
        SD->f12.wasnan_data[i] = true;
        exitg1 = true;
      } else {
        j++;
      }
    }
  }

  if (hadnans) {
    j = X_size[0] - 1;
    trueCount = 0;
    for (i = 0; i <= j; i++) {
      if (!SD->f12.wasnan_data[i]) {
        trueCount++;
      }
    }

    partialTrueCount = 0;
    for (i = 0; i <= j; i++) {
      if (!SD->f12.wasnan_data[i]) {
        SD->f12.idx_data[partialTrueCount] = i + 1;
        partialTrueCount++;
      }
    }

    for (partialTrueCount = 0; partialTrueCount < trueCount; partialTrueCount++)
    {
      j = 3 * (SD->f12.idx_data[partialTrueCount] - 1);
      SD->f12.X_data[3 * partialTrueCount] = X_data[j];
      SD->f12.X_data[1 + 3 * partialTrueCount] = X_data[1 + j];
      SD->f12.X_data[2 + 3 * partialTrueCount] = X_data[2 + j];
    }

    X_size[1] = 3;
    X_size[0] = trueCount;
    j = 3 * trueCount;
    if (0 <= j - 1) {
      memcpy(&X_data[0], &SD->f12.X_data[0], (unsigned int)(j * (int)sizeof
              (double)));
    }
  }

  local_kmeans(SD, X_data, X_size, SD->f12.idx_data, idx_size, Cbest,
               varargout_1, varargout_2_data, varargout_2_size);
  if (hadnans) {
    j = -1;
    idxbest_size[0] = n;
    for (i = 0; i < n; i++) {
      if (SD->f12.wasnan_data[i]) {
        idxbest_data[i] = rtNaN;
      } else {
        j++;
        idxbest_data[i] = SD->f12.idx_data[j];
      }
    }
  } else {
    idxbest_size[0] = idx_size[0];
    j = idx_size[0];
    for (partialTrueCount = 0; partialTrueCount < j; partialTrueCount++) {
      idxbest_data[partialTrueCount] = SD->f12.idx_data[partialTrueCount];
    }
  }
}

/*
 *
 */
static void local_kmeans(SmartLoaderStackData *SD, const double X_data[], const
  int X_size[2], int idxbest_data[], int idxbest_size[1], double Cbest[6],
  double varargout_1[2], double varargout_2_data[], int varargout_2_size[2])
{
  double totsumDbest;
  int rep;
  double totsumD;
  int idx_size[1];
  double C[6];
  double sumD[2];
  int D_size[2];
  int loop_ub;
  loopBody(SD, X_data, X_size, &totsumDbest, idxbest_data, idxbest_size, Cbest,
           varargout_1, varargout_2_data, varargout_2_size);
  for (rep = 0; rep < 4; rep++) {
    loopBody(SD, X_data, X_size, &totsumD, SD->f11.idx_data, idx_size, C, sumD,
             SD->f11.D_data, D_size);
    if (totsumD < totsumDbest) {
      totsumDbest = totsumD;
      idxbest_size[0] = idx_size[0];
      if (0 <= idx_size[0] - 1) {
        memcpy(&idxbest_data[0], &SD->f11.idx_data[0], (unsigned int)(idx_size[0]
                * (int)sizeof(int)));
      }

      for (loop_ub = 0; loop_ub < 6; loop_ub++) {
        Cbest[loop_ub] = C[loop_ub];
      }

      varargout_1[0] = sumD[0];
      varargout_1[1] = sumD[1];
      varargout_2_size[1] = 2;
      varargout_2_size[0] = D_size[0];
      loop_ub = D_size[1] * D_size[0];
      if (0 <= loop_ub - 1) {
        memcpy(&varargout_2_data[0], &SD->f11.D_data[0], (unsigned int)(loop_ub *
                (int)sizeof(double)));
      }
    }
  }
}

/*
 *
 */
static void loopBody(SmartLoaderStackData *SD, const double X_data[], const int
                     X_size[2], double *totsumD, int idx_data[], int idx_size[1],
                     double C[6], double sumD[2], double D_data[], int D_size[2])
{
  int n;
  double b_index;
  int pidx;
  int nNonEmpty;
  int sampleDist_size[1];
  boolean_T DNeedsComputing;
  double d0;
  int crows[2];
  int nonEmpties[2];
  n = X_size[0] - 1;
  b_index = b_rand(SD);
  for (pidx = 0; pidx < 6; pidx++) {
    C[pidx] = 0.0;
  }

  pidx = (int)(1.0 + std::floor(b_index * (double)X_size[0]));
  C[0] = X_data[3 * (pidx - 1)];
  C[1] = X_data[1 + 3 * (pidx - 1)];
  C[2] = X_data[2 + 3 * (pidx - 1)];
  D_size[1] = 2;
  D_size[0] = X_size[0];
  nNonEmpty = X_size[0] << 1;
  if (0 <= nNonEmpty - 1) {
    memset(&D_data[0], 0, (unsigned int)(nNonEmpty * (int)sizeof(double)));
  }

  distfun(D_data, X_data, X_size, C, 1);
  nNonEmpty = X_size[0];
  for (pidx = 0; pidx < nNonEmpty; pidx++) {
    SD->u3.f10.d_data[pidx] = D_data[pidx << 1];
  }

  idx_size[0] = X_size[0];
  nNonEmpty = X_size[0];
  for (pidx = 0; pidx < nNonEmpty; pidx++) {
    idx_data[pidx] = 1;
  }

  sampleDist_size[0] = X_size[0] + 1;
  if (0 <= X_size[0]) {
    memset(&SD->u3.f10.sampleDist_data[0], 0, (unsigned int)((X_size[0] + 1) *
            (int)sizeof(double)));
  }

  DNeedsComputing = false;
  b_index = 0.0;
  SD->u3.f10.sampleDist_data[0] = 0.0;
  for (pidx = 0; pidx <= n; pidx++) {
    d0 = D_data[pidx << 1];
    SD->u3.f10.sampleDist_data[pidx + 1] = SD->u3.f10.sampleDist_data[pidx] + d0;
    b_index += d0;
  }

  if ((b_index == 0.0) || (!b_isfinite(b_index))) {
    simpleRandperm(SD, X_size[0], idx_data, idx_size);
    pidx = 3 * (idx_data[0] - 1);
    C[3] = X_data[pidx];
    C[4] = X_data[1 + pidx];
    C[5] = X_data[2 + pidx];
    DNeedsComputing = true;
  } else {
    nNonEmpty = X_size[0] + 1;
    for (pidx = 0; pidx < nNonEmpty; pidx++) {
      SD->u3.f10.sampleDist_data[pidx] /= b_index;
    }

    pidx = b_bsearch(SD->u3.f10.sampleDist_data, sampleDist_size, b_rand(SD));
    b_index = SD->u3.f10.sampleDist_data[pidx - 1];
    if (SD->u3.f10.sampleDist_data[pidx - 1] < 1.0) {
      while ((pidx <= n + 1) && (SD->u3.f10.sampleDist_data[pidx] <= b_index)) {
        pidx++;
      }
    } else {
      while ((pidx >= 2) && (SD->u3.f10.sampleDist_data[pidx - 2] >= b_index)) {
        pidx--;
      }
    }

    pidx = 3 * (pidx - 1);
    C[3] = X_data[pidx];
    C[4] = X_data[1 + pidx];
    C[5] = X_data[2 + pidx];
    distfun(D_data, X_data, X_size, C, 2);
    for (pidx = 0; pidx <= n; pidx++) {
      d0 = D_data[1 + (pidx << 1)];
      if (d0 < SD->u3.f10.d_data[pidx]) {
        SD->u3.f10.d_data[pidx] = d0;
        idx_data[pidx] = 2;
      }
    }
  }

  if (DNeedsComputing) {
    crows[0] = 1;
    crows[1] = 2;
    b_distfun(D_data, X_data, X_size, C, crows, 2);
    mindim2(D_data, D_size, SD->u3.f10.d_data, sampleDist_size, idx_data,
            idx_size);
  }

  crows[0] = 0;
  crows[1] = 0;
  for (pidx = 0; pidx <= n; pidx++) {
    crows[idx_data[pidx] - 1]++;
  }

  nonEmpties[0] = 0;
  nonEmpties[1] = 0;
  batchUpdate(SD, X_data, X_size, idx_data, idx_size, C, D_data, D_size, crows,
              &DNeedsComputing, &pidx);
  nNonEmpty = -1;
  if (crows[0] > 0) {
    nNonEmpty = 0;
    nonEmpties[0] = 1;
  }

  if (crows[1] > 0) {
    nNonEmpty++;
    nonEmpties[nNonEmpty] = 2;
  }

  b_distfun(D_data, X_data, X_size, C, nonEmpties, nNonEmpty + 1);
  for (pidx = 0; pidx <= n; pidx++) {
    SD->u3.f10.d_data[pidx] = D_data[(idx_data[pidx] + (pidx << 1)) - 1];
  }

  sumD[0] = 0.0;
  sumD[1] = 0.0;
  for (pidx = 0; pidx <= n; pidx++) {
    sumD[idx_data[pidx] - 1] += SD->u3.f10.d_data[pidx];
  }

  *totsumD = 0.0;
  for (pidx = 0; pidx <= nNonEmpty; pidx++) {
    *totsumD += sumD[nonEmpties[pidx] - 1];
  }
}

/*
 *
 */
static void mean(const double x_data[], const int x_size[2], double y[3])
{
  int vlen;
  int k;
  vlen = x_size[0];
  if (x_size[0] == 0) {
    y[0] = 0.0;
    y[1] = 0.0;
    y[2] = 0.0;
  } else {
    y[0] = x_data[0];
    y[1] = x_data[1];
    y[2] = x_data[2];
    for (k = 2; k <= vlen; k++) {
      if (vlen >= 2) {
        y[0] += x_data[3 * (k - 1)];
        y[1] += x_data[1 + 3 * (k - 1)];
        y[2] += x_data[2 + 3 * (k - 1)];
      }
    }
  }

  y[0] /= (double)x_size[0];
  y[1] /= (double)x_size[0];
  y[2] /= (double)x_size[0];
}

/*
 *
 */
static void med3(double v_data[], int nv, int ia, int ib)
{
  int ic;
  double tmp;
  if (nv >= 3) {
    ic = ia + (nv - 1) / 2;
    if (v_data[ia - 1] < v_data[ic - 1]) {
      if (v_data[ic - 1] < v_data[ib - 1]) {
      } else if (v_data[ia - 1] < v_data[ib - 1]) {
        ic = ib;
      } else {
        ic = ia;
      }
    } else if (v_data[ia - 1] < v_data[ib - 1]) {
      ic = ia;
    } else {
      if (v_data[ic - 1] < v_data[ib - 1]) {
        ic = ib;
      }
    }

    if (ic > ia) {
      tmp = v_data[ia - 1];
      v_data[ia - 1] = v_data[ic - 1];
      v_data[ic - 1] = tmp;
    }
  }
}

/*
 *
 */
static double median(SmartLoaderStackData *SD, const double x_data[], const int
                     x_size[1])
{
  double y;
  int b_x_size[1];
  if (x_size[0] == 0) {
    y = rtNaN;
  } else {
    b_x_size[0] = x_size[0];
    if (0 <= x_size[0] - 1) {
      memcpy(&SD->u2.f7.x_data[0], &x_data[0], (unsigned int)(x_size[0] * (int)
              sizeof(double)));
    }

    y = vmedian(SD, SD->u2.f7.x_data, b_x_size, x_size[0]);
  }

  return y;
}

/*
 *
 */
static void medmed(double v_data[], int nv, int ia)
{
  int ngroupsof5;
  int nlast;
  int k;
  int i1;
  int destidx;
  double tmp;
  while (nv > 1) {
    ngroupsof5 = nv / 5;
    nlast = nv - ngroupsof5 * 5;
    nv = ngroupsof5;
    for (k = 0; k < ngroupsof5; k++) {
      i1 = ia + k * 5;
      i1 = thirdOfFive(v_data, i1, i1 + 4) - 1;
      destidx = (ia + k) - 1;
      tmp = v_data[destidx];
      v_data[destidx] = v_data[i1];
      v_data[i1] = tmp;
    }

    if (nlast > 0) {
      i1 = ia + ngroupsof5 * 5;
      i1 = thirdOfFive(v_data, i1, (i1 + nlast) - 1) - 1;
      destidx = (ia + ngroupsof5) - 1;
      tmp = v_data[destidx];
      v_data[destidx] = v_data[i1];
      v_data[i1] = tmp;
      nv = ngroupsof5 + 1;
    }
  }
}

/*
 *
 */
static void merge(emxArray_int32_T *idx, emxArray_real_T *x, int offset, int np,
                  int nq, emxArray_int32_T *iwork, emxArray_real_T *xwork)
{
  int n_tmp;
  int iout;
  int p;
  int i43;
  int q;
  int exitg1;
  if (nq != 0) {
    n_tmp = np + nq;
    for (iout = 0; iout < n_tmp; iout++) {
      i43 = offset + iout;
      iwork->data[iout] = idx->data[i43];
      xwork->data[iout] = x->data[i43];
    }

    p = 0;
    q = np;
    iout = offset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork->data[p] <= xwork->data[q]) {
        idx->data[iout] = iwork->data[p];
        x->data[iout] = xwork->data[p];
        if (p + 1 < np) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx->data[iout] = iwork->data[q];
        x->data[iout] = xwork->data[q];
        if (q + 1 < n_tmp) {
          q++;
        } else {
          q = iout - p;
          for (iout = p + 1; iout <= np; iout++) {
            i43 = q + iout;
            idx->data[i43] = iwork->data[iout - 1];
            x->data[i43] = xwork->data[iout - 1];
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

/*
 *
 */
static void merge_block(emxArray_int32_T *idx, emxArray_real_T *x, int offset,
  int n, int preSortLevel, emxArray_int32_T *iwork, emxArray_real_T *xwork)
{
  int nPairs;
  int bLen;
  int tailOffset;
  int nTail;
  nPairs = n >> preSortLevel;
  bLen = 1 << preSortLevel;
  while (nPairs > 1) {
    if ((nPairs & 1) != 0) {
      nPairs--;
      tailOffset = bLen * nPairs;
      nTail = n - tailOffset;
      if (nTail > bLen) {
        merge(idx, x, offset + tailOffset, bLen, nTail - bLen, iwork, xwork);
      }
    }

    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 0; nTail < nPairs; nTail++) {
      merge(idx, x, offset + nTail * tailOffset, bLen, bLen, iwork, xwork);
    }

    bLen = tailOffset;
  }

  if (n > bLen) {
    merge(idx, x, offset, bLen, n - bLen, iwork, xwork);
  }
}

/*
 *
 */
static double minOrMaxRealFloatVector(const emxArray_real_T *x)
{
  double ex;
  int n;
  int idx;
  int k;
  boolean_T exitg1;
  n = x->size[0];
  if (x->size[0] <= 2) {
    if (x->size[0] == 1) {
      ex = x->data[0];
    } else if ((x->data[0] < x->data[1]) || (rtIsNaN(x->data[0]) && (!rtIsNaN
                 (x->data[1])))) {
      ex = x->data[1];
    } else {
      ex = x->data[0];
    }
  } else {
    if (!rtIsNaN(x->data[0])) {
      idx = 1;
    } else {
      idx = 0;
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= x->size[0])) {
        if (!rtIsNaN(x->data[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }

    if (idx == 0) {
      ex = x->data[0];
    } else {
      ex = x->data[idx - 1];
      idx++;
      for (k = idx; k <= n; k++) {
        if (ex < x->data[k - 1]) {
          ex = x->data[k - 1];
        }
      }
    }
  }

  return ex;
}

/*
 *
 */
static void mindim2(const double D_data[], const int D_size[2], double d_data[],
                    int d_size[1], int idx_data[], int idx_size[1])
{
  int n;
  int loop_ub;
  int i21;
  double d1;
  n = D_size[0];
  repmat(D_size[0], d_data, d_size);
  idx_size[0] = D_size[0];
  loop_ub = D_size[0];
  for (i21 = 0; i21 < loop_ub; i21++) {
    idx_data[i21] = 1;
  }

  for (loop_ub = 0; loop_ub < n; loop_ub++) {
    i21 = loop_ub << 1;
    if (D_data[i21] < d_data[loop_ub]) {
      idx_data[loop_ub] = 1;
      d_data[loop_ub] = D_data[i21];
    }
  }

  for (loop_ub = 0; loop_ub < n; loop_ub++) {
    d1 = D_data[1 + (loop_ub << 1)];
    if (d1 < d_data[loop_ub]) {
      idx_data[loop_ub] = 2;
      d_data[loop_ub] = d1;
    }
  }
}

/*
 *
 */
static double nCk(double n, double k)
{
  double y;
  double nmk;
  int i35;
  int j;
  if ((!rtIsInf(n)) && (!rtIsNaN(n))) {
    if (2.0 > n / 2.0) {
      k = n - 2.0;
    }

    y = 1.0;
    nmk = n - k;
    i35 = (int)k;
    for (j = 0; j < i35; j++) {
      y *= ((1.0 + (double)j) + nmk) / (1.0 + (double)j);
    }

    y = rt_roundd_snf(y);
  } else {
    y = rtNaN;
  }

  return y;
}

/*
 *
 */
static void nchoosek(const double x_data[], const int x_size[2], emxArray_real_T
                     *y)
{
  int nrows;
  int icomb;
  int comb[2];
  int nmkpi;
  int row;
  int a;
  int combj;
  if (x_size[1] == 1) {
    if (!(2.0 > x_data[0])) {
      icomb = y->size[0] * y->size[1];
      y->size[1] = 1;
      y->size[0] = 1;
      emxEnsureCapacity_real_T(y, icomb);
      y->data[0] = nCk(x_data[0], 2.0);
    }
  } else if (2 > x_size[1]) {
    y->size[1] = 2;
    y->size[0] = 0;
  } else {
    nrows = (int)std::floor(nCk((double)x_size[1], 2.0));
    icomb = y->size[0] * y->size[1];
    y->size[1] = 2;
    y->size[0] = nrows;
    emxEnsureCapacity_real_T(y, icomb);
    comb[0] = 1;
    comb[1] = 2;
    icomb = 1;
    nmkpi = x_size[1];
    for (row = 0; row < nrows; row++) {
      y->data[y->size[1] * row] = x_data[comb[0] - 1];
      y->data[1 + y->size[1] * row] = x_data[comb[1] - 1];
      if (icomb + 1 > 0) {
        a = comb[icomb];
        combj = comb[icomb] + 1;
        comb[icomb]++;
        if (a + 1 < nmkpi) {
          icomb += 2;
          for (nmkpi = icomb; nmkpi < 3; nmkpi++) {
            combj++;
            comb[1] = combj;
          }

          icomb = 1;
          nmkpi = x_size[1];
        } else {
          icomb--;
          nmkpi--;
        }
      }
    }
  }
}

/*
 *
 */
static double nestedIter(const emxArray_real_T *x, int vlen)
{
  double y;
  int k;
  y = x->data[0];
  for (k = 2; k <= vlen; k++) {
    if (vlen >= 2) {
      y += x->data[k - 1];
    }
  }

  return y;
}

/*
 *
 */
static void nullAssignment(const cell_wrap_0 x_data[], const boolean_T idx_data[],
  cell_wrap_0 b_x_data[], int x_size[2])
{
  int n;
  int k;
  int i30;
  int bidx;
  int loop_ub;
  n = 0;
  for (k = 0; k < 64; k++) {
    n += idx_data[k];
  }

  i30 = x_size[0] * x_size[1];
  x_size[1] = 64 - n;
  x_size[0] = 1;
  emxEnsureCapacity_cell_wrap_0(b_x_data, x_size, i30);
  bidx = 0;
  i30 = 63 - n;
  for (k = 0; k <= i30; k++) {
    while ((bidx + 1 <= 64) && idx_data[bidx]) {
      bidx++;
    }

    n = b_x_data[k].f1->size[0] * b_x_data[k].f1->size[1];
    b_x_data[k].f1->size[1] = x_data[bidx].f1->size[1];
    b_x_data[k].f1->size[0] = x_data[bidx].f1->size[0];
    emxEnsureCapacity_real_T(b_x_data[k].f1, n);
    loop_ub = x_data[bidx].f1->size[1] * x_data[bidx].f1->size[0];
    for (n = 0; n < loop_ub; n++) {
      b_x_data[k].f1->data[n] = x_data[bidx].f1->data[n];
    }

    bidx++;
  }
}

/*
 *
 */
static void pdist(const emxArray_real_T *Xin, emxArray_real_T *Y)
{
  emxArray_real_T *X;
  int px;
  int nx;
  int nd;
  int ub_loop;
  int loop_ub;
  int b_loop_ub;
  emxArray_boolean_T *logIndX;
  int i31;
  int ii;
  int jj;
  boolean_T nanflag;
  int kk;
  boolean_T exitg1;
  double qq;
  double b_ii;
  double tempSum;
  int i32;
  emxInit_real_T(&X, 2);
  px = Xin->size[1];
  nx = Xin->size[0];
  nd = Xin->size[0] * (Xin->size[0] - 1) / 2;
  ub_loop = X->size[0] * X->size[1];
  X->size[1] = Xin->size[0];
  X->size[0] = Xin->size[1];
  emxEnsureCapacity_real_T(X, ub_loop);
  loop_ub = Xin->size[1];
  for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
    b_loop_ub = Xin->size[0];
    for (i31 = 0; i31 < b_loop_ub; i31++) {
      X->data[i31 + X->size[1] * ub_loop] = Xin->data[ub_loop + Xin->size[1] *
        i31];
    }
  }

  if (Xin->size[0] == 0) {
    Y->size[1] = 0;
    Y->size[0] = 1;
  } else {
    emxInit_boolean_T(&logIndX, 2);
    ub_loop = logIndX->size[0] * logIndX->size[1];
    logIndX->size[1] = Xin->size[0];
    logIndX->size[0] = 1;
    emxEnsureCapacity_boolean_T(logIndX, ub_loop);
    loop_ub = Xin->size[0];
    for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
      logIndX->data[ub_loop] = true;
    }

    ub_loop = Xin->size[0];
    ub_loop--;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(jj,nanflag,exitg1)

    for (ii = 0; ii <= ub_loop; ii++) {
      nanflag = false;
      jj = 0;
      exitg1 = false;
      while ((!exitg1) && (jj <= px - 1)) {
        if (rtIsNaN(X->data[ii + X->size[1] * jj])) {
          nanflag = true;
          exitg1 = true;
        } else {
          jj++;
        }
      }

      if (nanflag) {
        logIndX->data[ii] = false;
      }
    }

    ub_loop = Y->size[0] * Y->size[1];
    Y->size[1] = nd;
    Y->size[0] = 1;
    emxEnsureCapacity_real_T(Y, ub_loop);
    for (ub_loop = 0; ub_loop < nd; ub_loop++) {
      Y->data[ub_loop] = rtNaN;
    }

    ub_loop = (int)((double)nx * ((double)nx - 1.0) / 2.0) - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(jj,qq,b_ii,tempSum,ii,i32)

    for (kk = 0; kk <= ub_loop; kk++) {
      tempSum = 0.0;
      b_ii = (((double)nx - 2.0) - std::floor(std::sqrt((-8.0 * ((1.0 + (double)
        kk) - 1.0) + 4.0 * (double)nx * ((double)nx - 1.0)) - 7.0) / 2.0 - 0.5))
        + 1.0;
      qq = (double)nx - b_ii;
      qq = (((1.0 + (double)kk) + b_ii) - (double)nx * ((double)nx - 1.0) / 2.0)
        + qq * (qq + 1.0) / 2.0;
      ii = (int)b_ii;
      if (logIndX->data[ii - 1]) {
        i32 = (int)qq;
        if (logIndX->data[i32 - 1]) {
          for (jj = 0; jj < px; jj++) {
            tempSum += (X->data[(i32 + X->size[1] * jj) - 1] - X->data[(ii +
              X->size[1] * jj) - 1]) * (X->data[((int)qq + X->size[1] * jj) - 1]
              - X->data[((int)b_ii + X->size[1] * jj) - 1]);
          }

          Y->data[kk] = std::sqrt(tempSum);
        }
      }
    }

    emxFree_boolean_T(&logIndX);
  }

  emxFree_real_T(&X);
}

/*
 *
 */
static void pdist2(SmartLoaderStackData *SD, const emxArray_real_T *Xin, const
                   double Yin_data[], const int Yin_size[2], emxArray_real_T *D)
{
  emxArray_real_T *X;
  int nx;
  int ny;
  int ub_loop;
  int loop_ub;
  int Y_size[2];
  int b_loop_ub;
  int i29;
  emxArray_boolean_T *logIndX;
  int ii;
  int qq;
  int jj;
  boolean_T nanflag;
  boolean_T exitg1;
  double tempSum;
  emxInit_real_T(&X, 2);
  nx = Xin->size[0];
  ny = Yin_size[0] - 1;
  ub_loop = X->size[0] * X->size[1];
  X->size[1] = Xin->size[0];
  X->size[0] = Xin->size[1];
  emxEnsureCapacity_real_T(X, ub_loop);
  loop_ub = Xin->size[1];
  for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
    b_loop_ub = Xin->size[0];
    for (i29 = 0; i29 < b_loop_ub; i29++) {
      X->data[i29 + X->size[1] * ub_loop] = Xin->data[ub_loop + Xin->size[1] *
        i29];
    }
  }

  Y_size[1] = Yin_size[0];
  loop_ub = Yin_size[0];
  for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
    SD->u1.f1.Y_data[ub_loop] = Yin_data[ub_loop << 1];
  }

  for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
    SD->u1.f1.Y_data[ub_loop + Y_size[1]] = Yin_data[1 + (ub_loop << 1)];
  }

  emxInit_boolean_T(&logIndX, 2);
  if ((Xin->size[0] == 0) || (Yin_size[0] == 0)) {
    ub_loop = D->size[0] * D->size[1];
    D->size[1] = Yin_size[0];
    D->size[0] = Xin->size[0];
    emxEnsureCapacity_real_T(D, ub_loop);
    loop_ub = Yin_size[0] * Xin->size[0];
    for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
      D->data[ub_loop] = 0.0;
    }
  } else {
    ub_loop = D->size[0] * D->size[1];
    D->size[1] = Yin_size[0];
    D->size[0] = Xin->size[0];
    emxEnsureCapacity_real_T(D, ub_loop);
    loop_ub = Yin_size[0] * Xin->size[0];
    for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
      D->data[ub_loop] = rtNaN;
    }

    ub_loop = logIndX->size[0] * logIndX->size[1];
    logIndX->size[1] = Xin->size[0];
    logIndX->size[0] = 1;
    emxEnsureCapacity_boolean_T(logIndX, ub_loop);
    loop_ub = Xin->size[0];
    for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
      logIndX->data[ub_loop] = true;
    }

    loop_ub = Yin_size[0];
    for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
      SD->u1.f1.logIndY_data[ub_loop] = true;
    }

    ub_loop = Yin_size[0] - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(jj,nanflag,exitg1)

    for (ii = 0; ii <= ub_loop; ii++) {
      nanflag = false;
      jj = 0;
      exitg1 = false;
      while ((!exitg1) && (jj < 2)) {
        if (rtIsNaN(SD->u1.f1.Y_data[ii + Y_size[1] * jj])) {
          nanflag = true;
          exitg1 = true;
        } else {
          jj++;
        }
      }

      if (nanflag) {
        SD->u1.f1.logIndY_data[ii] = false;
      }
    }

    ub_loop = nx - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(jj,nanflag,exitg1)

    for (qq = 0; qq <= ub_loop; qq++) {
      nanflag = false;
      jj = 0;
      exitg1 = false;
      while ((!exitg1) && (jj < 2)) {
        if (rtIsNaN(X->data[qq + X->size[1] * jj])) {
          nanflag = true;
          exitg1 = true;
        } else {
          jj++;
        }
      }

      if (nanflag) {
        logIndX->data[qq] = false;
      }
    }

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(tempSum,qq)

    for (ii = 0; ii <= ny; ii++) {
      if (SD->u1.f1.logIndY_data[ii]) {
        for (qq = 0; qq < nx; qq++) {
          if (logIndX->data[qq]) {
            tempSum = rt_powd_snf(X->data[qq] - SD->u1.f1.Y_data[ii], 2.0);
            tempSum += rt_powd_snf(X->data[qq + X->size[1]] - SD->
              u1.f1.Y_data[ii + Y_size[1]], 2.0);
            D->data[ii + D->size[1] * qq] = std::sqrt(tempSum);
          }
        }
      }
    }
  }

  emxFree_boolean_T(&logIndX);
  emxFree_real_T(&X);
}

/*
 *
 */
static int pivot(double v_data[], int *ip, int ia, int ib)
{
  int reps;
  double vref;
  int i60;
  int k;
  double vk_tmp;
  vref = v_data[*ip - 1];
  v_data[*ip - 1] = v_data[ib - 1];
  v_data[ib - 1] = vref;
  *ip = ia;
  reps = 0;
  i60 = ib - 1;
  for (k = ia; k <= i60; k++) {
    vk_tmp = v_data[k - 1];
    if (vk_tmp == vref) {
      v_data[k - 1] = v_data[*ip - 1];
      v_data[*ip - 1] = vk_tmp;
      reps++;
      (*ip)++;
    } else {
      if (vk_tmp < vref) {
        v_data[k - 1] = v_data[*ip - 1];
        v_data[*ip - 1] = vk_tmp;
        (*ip)++;
      }
    }
  }

  v_data[ib - 1] = v_data[*ip - 1];
  v_data[*ip - 1] = vref;
  return reps;
}

/*
 *
 */
static void populateOutputStatsStructure(emxArray_struct_T *outstats, const
  b_emxArray_struct_T *stats)
{
  int i50;
  int k;
  i50 = stats->size[0];
  for (k = 0; k < i50; k++) {
    outstats->data[k].Orientation = stats->data[k].Orientation;
    outstats->data[k].Area = stats->data[k].Area;
    outstats->data[k].MajorAxisLength = stats->data[k].MajorAxisLength;
    outstats->data[k].MinorAxisLength = stats->data[k].MinorAxisLength;
    outstats->data[k].Centroid[0] = stats->data[k].Centroid[0];
    outstats->data[k].Centroid[1] = stats->data[k].Centroid[1];
  }
}

/*
 *
 */
static void power(const emxArray_real_T *a, emxArray_real_T *y)
{
  unsigned int a_idx_0;
  int N;
  int k;
  a_idx_0 = (unsigned int)a->size[0];
  N = y->size[0];
  y->size[0] = (int)a_idx_0;
  emxEnsureCapacity_real_T(y, N);
  a_idx_0 = (unsigned int)a->size[0];
  N = (int)a_idx_0;
  for (k = 0; k < N; k++) {
    y->data[k] = rt_powd_snf(a->data[k], 2.0);
  }
}

/*
 *
 */
static void quickselect(double v_data[], int n, int vlen, double *vn, int
  *nfirst, int *nlast)
{
  int ipiv;
  int ia;
  int ib;
  int oldnv;
  boolean_T checkspeed;
  boolean_T isslow;
  boolean_T exitg1;
  int reps;
  boolean_T guard1 = false;
  if (n > vlen) {
    *vn = rtNaN;
    *nfirst = 0;
    *nlast = 0;
  } else {
    ipiv = n;
    ia = 1;
    ib = vlen;
    *nfirst = 1;
    *nlast = vlen;
    oldnv = vlen;
    checkspeed = false;
    isslow = false;
    exitg1 = false;
    while ((!exitg1) && (ia < ib)) {
      reps = pivot(v_data, &ipiv, ia, ib);
      *nlast = ipiv;
      guard1 = false;
      if (n <= ipiv) {
        *nfirst = ipiv - reps;
        if (n >= *nfirst) {
          exitg1 = true;
        } else {
          ib = ipiv - 1;
          guard1 = true;
        }
      } else {
        ia = ipiv + 1;
        guard1 = true;
      }

      if (guard1) {
        reps = (ib - ia) + 1;
        if (checkspeed) {
          isslow = (reps > oldnv / 2);
          oldnv = reps;
        }

        checkspeed = !checkspeed;
        if (isslow) {
          medmed(v_data, reps, ia);
        } else {
          med3(v_data, reps, ia, ib);
        }

        ipiv = ia;
        *nfirst = ia;
        *nlast = ib;
      }
    }

    *vn = v_data[*nlast - 1];
  }
}

/*
 *
 */
static void regionprops(const double varargin_1_data[], const int
  varargin_1_size[2], emxArray_struct_T *outstats)
{
  double imageSize[2];
  int m;
  double numObjs;
  int n;
  int maxval_size[1];
  b_emxArray_struct_T *stats;
  int i;
  struct_T statsAlreadyComputed;
  emxArray_real_T maxval_data;
  double b_maxval_data[1024];
  int maxval_data_tmp;
  int j;
  boolean_T p;
  static const c_struct_T r17 = { 0.0, /* Orientation */
    0.0,                               /* Area */
    0.0,                               /* MajorAxisLength */
    0.0,                               /* MinorAxisLength */
    { 0.0, 0.0 }                       /* Centroid */
  };

  imageSize[0] = varargin_1_size[0];
  imageSize[1] = varargin_1_size[1];
  if ((varargin_1_size[0] == 0) || (varargin_1_size[1] == 0)) {
    numObjs = 0.0;
  } else {
    m = varargin_1_size[0];
    n = varargin_1_size[1];
    maxval_size[0] = (short)varargin_1_size[0];
    for (i = 0; i < m; i++) {
      maxval_data_tmp = varargin_1_size[1] * i;
      b_maxval_data[i] = varargin_1_data[maxval_data_tmp];
      for (j = 2; j <= n; j++) {
        numObjs = varargin_1_data[(j + maxval_data_tmp) - 1];
        p = ((!rtIsNaN(numObjs)) && (rtIsNaN(b_maxval_data[i]) ||
              (b_maxval_data[i] < numObjs)));
        if (p) {
          b_maxval_data[i] = numObjs;
        }
      }
    }

    maxval_data.data = &b_maxval_data[0];
    maxval_data.size = &maxval_size[0];
    maxval_data.allocatedSize = 1024;
    maxval_data.numDimensions = 1;
    maxval_data.canFreeData = false;
    numObjs = std::floor(minOrMaxRealFloatVector(&maxval_data));
    if ((0.0 > numObjs) || rtIsNaN(numObjs)) {
      numObjs = 0.0;
    }
  }

  emxInit_struct_T(&stats, 1);
  initializeStatsStruct(numObjs, stats, &statsAlreadyComputed);
  ComputePixelIdxList(varargin_1_data, varargin_1_size, numObjs, stats,
                      &statsAlreadyComputed);
  ComputeEllipseParams(imageSize, stats, &statsAlreadyComputed);
  ComputeArea(stats, &statsAlreadyComputed);
  ComputeEllipseParams(imageSize, stats, &statsAlreadyComputed);
  ComputeEllipseParams(imageSize, stats, &statsAlreadyComputed);
  ComputeCentroid(imageSize, stats, &statsAlreadyComputed);
  m = outstats->size[0];
  n = (int)numObjs;
  outstats->size[0] = n;
  emxEnsureCapacity_struct_T(outstats, m);
  for (m = 0; m < n; m++) {
    outstats->data[m] = r17;
  }

  populateOutputStatsStructure(outstats, stats);
  emxFree_struct_T(&stats);
}

/*
 *
 */
static void repmat(int varargin_1, double b_data[], int b_size[1])
{
  int i22;
  b_size[0] = varargin_1;
  for (i22 = 0; i22 < varargin_1; i22++) {
    b_data[i22] = rtInf;
  }
}

static double rt_powd_snf(double u0, double u1)
{
  double y;
  double d3;
  double d4;
  if (rtIsNaN(u0) || rtIsNaN(u1)) {
    y = rtNaN;
  } else {
    d3 = std::abs(u0);
    d4 = std::abs(u1);
    if (rtIsInf(u1)) {
      if (d3 == 1.0) {
        y = 1.0;
      } else if (d3 > 1.0) {
        if (u1 > 0.0) {
          y = rtInf;
        } else {
          y = 0.0;
        }
      } else if (u1 > 0.0) {
        y = 0.0;
      } else {
        y = rtInf;
      }
    } else if (d4 == 0.0) {
      y = 1.0;
    } else if (d4 == 1.0) {
      if (u1 > 0.0) {
        y = u0;
      } else {
        y = 1.0 / u0;
      }
    } else if (u1 == 2.0) {
      y = u0 * u0;
    } else if ((u1 == 0.5) && (u0 >= 0.0)) {
      y = std::sqrt(u0);
    } else if ((u0 < 0.0) && (u1 > std::floor(u1))) {
      y = rtNaN;
    } else {
      y = pow(u0, u1);
    }
  }

  return y;
}

static double rt_roundd_snf(double u)
{
  double y;
  if (std::abs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = std::floor(u + 0.5);
    } else if (u > -0.5) {
      y = u * 0.0;
    } else {
      y = std::ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

/*
 *
 */
static void simpleRandperm(SmartLoaderStackData *SD, int n, int idx_data[], int
  idx_size[1])
{
  int t;
  double denom;
  double pt;
  double u;
  t = 1;
  idx_size[0] = n;
  if (0 <= n - 1) {
    memset(&idx_data[0], 0, (unsigned int)(n * (int)sizeof(int)));
  }

  denom = n;
  pt = 1.0 / (double)n;
  u = b_rand(SD);
  while (u > pt) {
    t++;
    denom--;
    pt += (1.0 - pt) * (1.0 / denom);
  }

  idx_data[0] = t;
  b_rand(SD);
}

/*
 *
 */
static void sort(emxArray_real_T *x)
{
  int dim;
  int j;
  emxArray_real_T *vwork;
  int vlen;
  int vstride;
  int k;
  emxArray_int32_T *b_vwork;
  dim = 0;
  if (x->size[0] != 1) {
    dim = -1;
  }

  if (dim + 2 <= 1) {
    j = x->size[0];
  } else {
    j = 1;
  }

  emxInit_real_T(&vwork, 1);
  vlen = j - 1;
  vstride = vwork->size[0];
  vwork->size[0] = j;
  emxEnsureCapacity_real_T(vwork, vstride);
  vstride = 1;
  for (k = 0; k <= dim; k++) {
    vstride *= x->size[0];
  }

  emxInit_int32_T(&b_vwork, 1);
  for (j = 0; j < vstride; j++) {
    for (k = 0; k <= vlen; k++) {
      vwork->data[k] = x->data[j + k * vstride];
    }

    sortIdx(vwork, b_vwork);
    for (k = 0; k <= vlen; k++) {
      x->data[j + k * vstride] = vwork->data[k];
    }
  }

  emxFree_int32_T(&b_vwork);
  emxFree_real_T(&vwork);
}

/*
 *
 */
static void sortIdx(emxArray_real_T *x, emxArray_int32_T *idx)
{
  int ib;
  int i42;
  emxArray_int32_T *b_idx;
  emxArray_real_T *b_x;
  int i1;
  int bLen;
  double x4[4];
  int idx4[4];
  emxArray_int32_T *iwork;
  int offset;
  emxArray_real_T *xwork;
  int nNaNs;
  int k;
  signed char perm[4];
  int quartetOffset;
  int i3;
  int nNonNaN;
  int i4;
  int nBlocks;
  double d7;
  double d8;
  int bLen2;
  int nPairs;
  int b_iwork[256];
  double b_xwork[256];
  int exitg1;
  ib = x->size[0];
  i42 = idx->size[0];
  idx->size[0] = ib;
  emxEnsureCapacity_int32_T(idx, i42);
  for (i42 = 0; i42 < ib; i42++) {
    idx->data[i42] = 0;
  }

  if (x->size[0] != 0) {
    emxInit_int32_T(&b_idx, 1);
    i42 = b_idx->size[0];
    b_idx->size[0] = ib;
    emxEnsureCapacity_int32_T(b_idx, i42);
    for (i42 = 0; i42 < ib; i42++) {
      b_idx->data[i42] = 0;
    }

    emxInit_real_T(&b_x, 1);
    i1 = x->size[0];
    i42 = b_x->size[0];
    b_x->size[0] = i1;
    emxEnsureCapacity_real_T(b_x, i42);
    for (i42 = 0; i42 < i1; i42++) {
      b_x->data[i42] = x->data[i42];
    }

    i42 = x->size[0];
    bLen = x->size[0];
    x4[0] = 0.0;
    idx4[0] = 0;
    x4[1] = 0.0;
    idx4[1] = 0;
    x4[2] = 0.0;
    idx4[2] = 0;
    x4[3] = 0.0;
    idx4[3] = 0;
    emxInit_int32_T(&iwork, 1);
    offset = iwork->size[0];
    iwork->size[0] = ib;
    emxEnsureCapacity_int32_T(iwork, offset);
    for (offset = 0; offset < ib; offset++) {
      iwork->data[offset] = 0;
    }

    emxInit_real_T(&xwork, 1);
    i1 = x->size[0];
    offset = xwork->size[0];
    xwork->size[0] = i1;
    emxEnsureCapacity_real_T(xwork, offset);
    for (offset = 0; offset < i1; offset++) {
      xwork->data[offset] = 0.0;
    }

    nNaNs = 0;
    ib = -1;
    for (k = 0; k < bLen; k++) {
      if (rtIsNaN(b_x->data[k])) {
        offset = (bLen - nNaNs) - 1;
        b_idx->data[offset] = k + 1;
        xwork->data[offset] = b_x->data[k];
        nNaNs++;
      } else {
        ib++;
        idx4[ib] = k + 1;
        x4[ib] = b_x->data[k];
        if (ib + 1 == 4) {
          quartetOffset = k - nNaNs;
          if (x4[0] <= x4[1]) {
            i1 = 1;
            ib = 2;
          } else {
            i1 = 2;
            ib = 1;
          }

          if (x4[2] <= x4[3]) {
            i3 = 3;
            i4 = 4;
          } else {
            i3 = 4;
            i4 = 3;
          }

          d7 = x4[i1 - 1];
          d8 = x4[i3 - 1];
          if (d7 <= d8) {
            if (x4[ib - 1] <= d8) {
              perm[0] = (signed char)i1;
              perm[1] = (signed char)ib;
              perm[2] = (signed char)i3;
              perm[3] = (signed char)i4;
            } else if (x4[ib - 1] <= x4[i4 - 1]) {
              perm[0] = (signed char)i1;
              perm[1] = (signed char)i3;
              perm[2] = (signed char)ib;
              perm[3] = (signed char)i4;
            } else {
              perm[0] = (signed char)i1;
              perm[1] = (signed char)i3;
              perm[2] = (signed char)i4;
              perm[3] = (signed char)ib;
            }
          } else {
            d8 = x4[i4 - 1];
            if (d7 <= d8) {
              if (x4[ib - 1] <= d8) {
                perm[0] = (signed char)i3;
                perm[1] = (signed char)i1;
                perm[2] = (signed char)ib;
                perm[3] = (signed char)i4;
              } else {
                perm[0] = (signed char)i3;
                perm[1] = (signed char)i1;
                perm[2] = (signed char)i4;
                perm[3] = (signed char)ib;
              }
            } else {
              perm[0] = (signed char)i3;
              perm[1] = (signed char)i4;
              perm[2] = (signed char)i1;
              perm[3] = (signed char)ib;
            }
          }

          offset = perm[0] - 1;
          b_idx->data[quartetOffset - 3] = idx4[offset];
          i4 = perm[1] - 1;
          b_idx->data[quartetOffset - 2] = idx4[i4];
          i1 = perm[2] - 1;
          b_idx->data[quartetOffset - 1] = idx4[i1];
          ib = perm[3] - 1;
          b_idx->data[quartetOffset] = idx4[ib];
          b_x->data[quartetOffset - 3] = x4[offset];
          b_x->data[quartetOffset - 2] = x4[i4];
          b_x->data[quartetOffset - 1] = x4[i1];
          b_x->data[quartetOffset] = x4[ib];
          ib = -1;
        }
      }
    }

    offset = (bLen - nNaNs) - 1;
    if (ib + 1 > 0) {
      perm[1] = 0;
      perm[2] = 0;
      perm[3] = 0;
      if (ib + 1 == 1) {
        perm[0] = 1;
      } else if (ib + 1 == 2) {
        if (x4[0] <= x4[1]) {
          perm[0] = 1;
          perm[1] = 2;
        } else {
          perm[0] = 2;
          perm[1] = 1;
        }
      } else if (x4[0] <= x4[1]) {
        if (x4[1] <= x4[2]) {
          perm[0] = 1;
          perm[1] = 2;
          perm[2] = 3;
        } else if (x4[0] <= x4[2]) {
          perm[0] = 1;
          perm[1] = 3;
          perm[2] = 2;
        } else {
          perm[0] = 3;
          perm[1] = 1;
          perm[2] = 2;
        }
      } else if (x4[0] <= x4[2]) {
        perm[0] = 2;
        perm[1] = 1;
        perm[2] = 3;
      } else if (x4[1] <= x4[2]) {
        perm[0] = 2;
        perm[1] = 3;
        perm[2] = 1;
      } else {
        perm[0] = 3;
        perm[1] = 2;
        perm[2] = 1;
      }

      for (k = 0; k <= ib; k++) {
        i4 = perm[k] - 1;
        i1 = (offset - ib) + k;
        b_idx->data[i1] = idx4[i4];
        b_x->data[i1] = x4[i4];
      }
    }

    ib = (nNaNs >> 1) + 1;
    for (k = 0; k <= ib - 2; k++) {
      i1 = (offset + k) + 1;
      i3 = b_idx->data[i1];
      i4 = (bLen - k) - 1;
      b_idx->data[i1] = b_idx->data[i4];
      b_idx->data[i4] = i3;
      b_x->data[i1] = xwork->data[i4];
      b_x->data[i4] = xwork->data[i1];
    }

    if ((nNaNs & 1) != 0) {
      bLen = offset + ib;
      b_x->data[bLen] = xwork->data[bLen];
    }

    nNonNaN = i42 - nNaNs;
    i1 = 2;
    if (nNonNaN > 1) {
      if (i42 >= 256) {
        nBlocks = nNonNaN >> 8;
        if (nBlocks > 0) {
          for (quartetOffset = 0; quartetOffset < nBlocks; quartetOffset++) {
            offset = (quartetOffset << 8) - 1;
            for (nNaNs = 0; nNaNs < 6; nNaNs++) {
              bLen = 1 << (nNaNs + 2);
              bLen2 = bLen << 1;
              nPairs = 256 >> (nNaNs + 3);
              for (k = 0; k < nPairs; k++) {
                i3 = (offset + k * bLen2) + 1;
                for (i1 = 0; i1 < bLen2; i1++) {
                  ib = i3 + i1;
                  b_iwork[i1] = b_idx->data[ib];
                  b_xwork[i1] = b_x->data[ib];
                }

                i4 = 0;
                i1 = bLen;
                ib = i3 - 1;
                do {
                  exitg1 = 0;
                  ib++;
                  if (b_xwork[i4] <= b_xwork[i1]) {
                    b_idx->data[ib] = b_iwork[i4];
                    b_x->data[ib] = b_xwork[i4];
                    if (i4 + 1 < bLen) {
                      i4++;
                    } else {
                      exitg1 = 1;
                    }
                  } else {
                    b_idx->data[ib] = b_iwork[i1];
                    b_x->data[ib] = b_xwork[i1];
                    if (i1 + 1 < bLen2) {
                      i1++;
                    } else {
                      ib -= i4;
                      for (i1 = i4 + 1; i1 <= bLen; i1++) {
                        i42 = ib + i1;
                        b_idx->data[i42] = b_iwork[i1 - 1];
                        b_x->data[i42] = b_xwork[i1 - 1];
                      }

                      exitg1 = 1;
                    }
                  }
                } while (exitg1 == 0);
              }
            }
          }

          i1 = nBlocks << 8;
          ib = nNonNaN - i1;
          if (ib > 0) {
            merge_block(b_idx, b_x, i1, ib, 2, iwork, xwork);
          }

          i1 = 8;
        }
      }

      merge_block(b_idx, b_x, 0, nNonNaN, i1, iwork, xwork);
    }

    emxFree_real_T(&xwork);
    emxFree_int32_T(&iwork);
    i1 = b_idx->size[0];
    for (i42 = 0; i42 < i1; i42++) {
      idx->data[i42] = b_idx->data[i42];
    }

    emxFree_int32_T(&b_idx);
    i1 = b_x->size[0];
    for (i42 = 0; i42 < i1; i42++) {
      x->data[i42] = b_x->data[i42];
    }

    emxFree_real_T(&b_x);
  }
}

/*
 *
 */
static void squareform(const emxArray_real_T *Y, emxArray_real_T *Z)
{
  int m;
  int i33;
  int loop_ub;
  unsigned int k;
  int i;
  int b_i;
  int i34;
  m = (int)std::ceil(std::sqrt(2.0 * (double)Y->size[1]));
  i33 = Z->size[0] * Z->size[1];
  Z->size[1] = m;
  Z->size[0] = m;
  emxEnsureCapacity_real_T(Z, i33);
  loop_ub = m * m;
  for (i33 = 0; i33 < loop_ub; i33++) {
    Z->data[i33] = 0.0;
  }

  if (m > 1) {
    k = 1U;
    for (loop_ub = 0; loop_ub <= m - 2; loop_ub++) {
      i33 = m - loop_ub;
      for (i = 0; i <= i33 - 2; i++) {
        b_i = (loop_ub + i) + 1;
        i34 = (int)k - 1;
        Z->data[loop_ub + Z->size[1] * b_i] = Y->data[i34];
        Z->data[b_i + Z->size[1] * loop_ub] = Y->data[i34];
        k++;
      }
    }
  }
}

/*
 *
 */
static double sum(const emxArray_real_T *x)
{
  return nestedIter(x, x->size[0]);
}

/*
 *
 */
static int thirdOfFive(const double v_data[], int ia, int ib)
{
  int im;
  double v4;
  double v5_tmp;
  int b_j1;
  int j2;
  int j3;
  int j4;
  int j5;
  double v5;
  if ((ia == ib) || (ia + 1 == ib)) {
    im = ia;
  } else if ((ia + 2 == ib) || (ia + 3 == ib)) {
    if (v_data[ia - 1] < v_data[ia]) {
      if (v_data[ia] < v_data[ia + 1]) {
        im = ia + 1;
      } else if (v_data[ia - 1] < v_data[ia + 1]) {
        im = ia + 2;
      } else {
        im = ia;
      }
    } else if (v_data[ia - 1] < v_data[ia + 1]) {
      im = ia;
    } else if (v_data[ia] < v_data[ia + 1]) {
      im = ia + 2;
    } else {
      im = ia + 1;
    }
  } else {
    v4 = v_data[ia - 1];
    if (v4 < v_data[ia]) {
      if (v_data[ia] < v_data[ia + 1]) {
        b_j1 = ia;
        j2 = ia;
        j3 = ia + 2;
      } else if (v4 < v_data[ia + 1]) {
        b_j1 = ia;
        j2 = ia + 1;
        j3 = ia + 1;
      } else {
        b_j1 = ia + 2;
        j2 = ia - 1;
        j3 = ia + 1;
      }
    } else {
      v5_tmp = v_data[ia + 1];
      if (v4 < v5_tmp) {
        b_j1 = ia + 1;
        j2 = ia - 1;
        j3 = ia + 2;
      } else if (v_data[ia] < v5_tmp) {
        b_j1 = ia + 1;
        j2 = ia + 1;
        j3 = ia;
      } else {
        b_j1 = ia + 2;
        j2 = ia;
        j3 = ia;
      }
    }

    j4 = ia;
    j5 = ia + 1;
    v4 = v_data[ia + 2];
    v5_tmp = v_data[ia + 3];
    v5 = v5_tmp;
    if (v5_tmp < v4) {
      j4 = ia + 1;
      j5 = ia;
      v5 = v4;
      v4 = v5_tmp;
    }

    if (v5 < v_data[b_j1 - 1]) {
      im = b_j1;
    } else if (v5 < v_data[j2]) {
      im = j5 + 3;
    } else if (v4 < v_data[j2]) {
      im = j2 + 1;
    } else if (v4 < v_data[j3 - 1]) {
      im = j4 + 3;
    } else {
      im = j3;
    }
  }

  return im;
}

/*
 *
 */
static void vecnorm(const double x[6], double y[2])
{
  int k;
  double scale;
  double absxk;
  double t;
  double yv;
  y[0] = 0.0;
  y[1] = 0.0;
  for (k = 0; k < 2; k++) {
    scale = 3.3121686421112381E-170;
    absxk = std::abs(x[k]);
    if (absxk > 3.3121686421112381E-170) {
      yv = 1.0;
      scale = absxk;
    } else {
      t = absxk / 3.3121686421112381E-170;
      yv = t * t;
    }

    absxk = std::abs(x[k + 2]);
    if (absxk > scale) {
      t = scale / absxk;
      yv = 1.0 + yv * t * t;
      scale = absxk;
    } else {
      t = absxk / scale;
      yv += t * t;
    }

    absxk = std::abs(x[k + 4]);
    if (absxk > scale) {
      t = scale / absxk;
      yv = 1.0 + yv * t * t;
      scale = absxk;
    } else {
      t = absxk / scale;
      yv += t * t;
    }

    y[k] = scale * std::sqrt(yv);
  }
}

/*
 *
 */
static double vmedian(SmartLoaderStackData *SD, double v_data[], int v_size[1],
                      int n)
{
  double m;
  int k;
  int exitg1;
  int midm1;
  int ilast;
  int unusedU5;
  boolean_T b8;
  boolean_T b9;
  double b;
  boolean_T b10;
  boolean_T b11;
  boolean_T b12;
  boolean_T b13;
  boolean_T b14;
  boolean_T b15;
  boolean_T b16;
  boolean_T b17;
  k = 0;
  do {
    exitg1 = 0;
    if (k <= n - 1) {
      if (rtIsNaN(v_data[k])) {
        m = rtNaN;
        exitg1 = 1;
      } else {
        k++;
      }
    } else {
      if (n <= 4) {
        if (n == 0) {
          m = rtNaN;
        } else if (n == 1) {
          m = v_data[0];
        } else if (n == 2) {
          if (rtIsInf(v_data[0]) || rtIsInf(v_data[1])) {
            m = (v_data[0] + v_data[1]) / 2.0;
          } else {
            m = v_data[0] + (v_data[1] - v_data[0]) / 2.0;
          }
        } else if (n == 3) {
          if (rtIsNaN(v_data[1])) {
            b9 = !rtIsNaN(v_data[0]);
          } else {
            b9 = ((!rtIsNaN(v_data[0])) && (v_data[0] < v_data[1]));
          }

          if (b9) {
            if (rtIsNaN(v_data[2])) {
              b13 = !rtIsNaN(v_data[1]);
            } else {
              b13 = ((!rtIsNaN(v_data[1])) && (v_data[1] < v_data[2]));
            }

            if (b13) {
              unusedU5 = 1;
            } else {
              if (rtIsNaN(v_data[2])) {
                b17 = !rtIsNaN(v_data[0]);
              } else {
                b17 = ((!rtIsNaN(v_data[0])) && (v_data[0] < v_data[2]));
              }

              if (b17) {
                unusedU5 = 2;
              } else {
                unusedU5 = 0;
              }
            }
          } else {
            if (rtIsNaN(v_data[2])) {
              b12 = !rtIsNaN(v_data[0]);
            } else {
              b12 = ((!rtIsNaN(v_data[0])) && (v_data[0] < v_data[2]));
            }

            if (b12) {
              unusedU5 = 0;
            } else {
              if (rtIsNaN(v_data[2])) {
                b16 = !rtIsNaN(v_data[1]);
              } else {
                b16 = ((!rtIsNaN(v_data[1])) && (v_data[1] < v_data[2]));
              }

              if (b16) {
                unusedU5 = 2;
              } else {
                unusedU5 = 1;
              }
            }
          }

          m = v_data[unusedU5];
        } else {
          if (rtIsNaN(v_data[1])) {
            b8 = !rtIsNaN(v_data[0]);
          } else {
            b8 = ((!rtIsNaN(v_data[0])) && (v_data[0] < v_data[1]));
          }

          if (b8) {
            if (rtIsNaN(v_data[2])) {
              b11 = !rtIsNaN(v_data[1]);
            } else {
              b11 = ((!rtIsNaN(v_data[1])) && (v_data[1] < v_data[2]));
            }

            if (b11) {
              k = 0;
              unusedU5 = 1;
              ilast = 2;
            } else {
              if (rtIsNaN(v_data[2])) {
                b15 = !rtIsNaN(v_data[0]);
              } else {
                b15 = ((!rtIsNaN(v_data[0])) && (v_data[0] < v_data[2]));
              }

              if (b15) {
                k = 0;
                unusedU5 = 2;
                ilast = 1;
              } else {
                k = 2;
                unusedU5 = 0;
                ilast = 1;
              }
            }
          } else {
            if (rtIsNaN(v_data[2])) {
              b10 = !rtIsNaN(v_data[0]);
            } else {
              b10 = ((!rtIsNaN(v_data[0])) && (v_data[0] < v_data[2]));
            }

            if (b10) {
              k = 1;
              unusedU5 = 0;
              ilast = 2;
            } else {
              if (rtIsNaN(v_data[2])) {
                b14 = !rtIsNaN(v_data[1]);
              } else {
                b14 = ((!rtIsNaN(v_data[1])) && (v_data[1] < v_data[2]));
              }

              if (b14) {
                k = 1;
                unusedU5 = 2;
                ilast = 0;
              } else {
                k = 2;
                unusedU5 = 1;
                ilast = 0;
              }
            }
          }

          if (v_data[k] < v_data[3]) {
            if (v_data[3] < v_data[ilast]) {
              if (rtIsInf(v_data[unusedU5]) || rtIsInf(v_data[3])) {
                m = (v_data[unusedU5] + v_data[3]) / 2.0;
              } else {
                m = v_data[unusedU5] + (v_data[3] - v_data[unusedU5]) / 2.0;
              }
            } else if (rtIsInf(v_data[unusedU5]) || rtIsInf(v_data[ilast])) {
              m = (v_data[unusedU5] + v_data[ilast]) / 2.0;
            } else {
              m = v_data[unusedU5] + (v_data[ilast] - v_data[unusedU5]) / 2.0;
            }
          } else if (rtIsInf(v_data[k]) || rtIsInf(v_data[unusedU5])) {
            m = (v_data[k] + v_data[unusedU5]) / 2.0;
          } else {
            m = v_data[k] + (v_data[unusedU5] - v_data[k]) / 2.0;
          }
        }
      } else {
        midm1 = n >> 1;
        if ((n & 1) == 0) {
          quickselect(v_data, midm1 + 1, n, &m, &k, &ilast);
          if (midm1 < k) {
            k = v_size[0];
            if (0 <= k - 1) {
              memcpy(&SD->u1.f2.unusedU3_data[0], &v_data[0], (unsigned int)(k *
                      (int)sizeof(double)));
            }

            quickselect(SD->u1.f2.unusedU3_data, midm1, ilast - 1, &b, &k,
                        &unusedU5);
            if (rtIsInf(m) || rtIsInf(b)) {
              m = (m + b) / 2.0;
            } else {
              m += (b - m) / 2.0;
            }
          }
        } else {
          k = v_size[0];
          if (0 <= k - 1) {
            memcpy(&SD->u1.f2.unusedU3_data[0], &v_data[0], (unsigned int)(k *
                    (int)sizeof(double)));
          }

          quickselect(SD->u1.f2.unusedU3_data, midm1 + 1, n, &m, &k, &unusedU5);
        }
      }

      exitg1 = 1;
    }
  } while (exitg1 == 0);

  return m;
}

/*
 * function [smartLoaderStruct, heightMap_res, debugPtCloudSenceXyz, debugPtCloudSenceIntensity] = SmartLoader(configParams, xyz, intensity)
 */
void SmartLoader(SmartLoaderStackData *SD, const SmartLoaderConfigParam
                 *configParams, const double xyz_data[], const int xyz_size[2],
                 const double intensity_data[], const int [1], SmartLoaderStruct
                 *smartLoaderStruct, float heightMap_res_data[], int
                 heightMap_res_size[2])
{
  int i1;
  double b_configParams[16];
  int n;
  int loop_ub;
  double affineTrans[12];
  int m;
  int i;
  int j;
  int pcTrans_size[2];
  static const double B[12] = { 6.123233995736766E-17, 0.026176948307873153,
    0.99965732497555726, 0.0, 0.99965732497555726, -0.026176948307873153, -1.0,
    1.6028757978341289E-18, 6.1211357163776083E-17, 0.0, 0.0, 0.0 };

  double zMedian;
  double singleReflectorRangeLimitMeter;
  int trueCount;
  int ptCloudSenceXyz_size[2];
  double v[3];
  double minval_idx_0;
  double b_tmp;
  boolean_T b_v[2];
  int ptCloudSenceReflectorsXyz_size[2];
  int kmeansIdx_size[1];
  double kmeansC[6];
  double distanceToKmeansClusterMeter[2];
  int iv0[1];
  int tmp_size[1];
  int b_ptCloudSenceReflectorsXyz_size[2];
  double pcFirstRange_idx_0;
  double pcFirstRange_idx_1;
  int c_ptCloudSenceReflectorsXyz_size[2];
  emxArray_cell_wrap_0_64x1 clustersXs;
  emxArray_cell_wrap_0_64x1 clustersYs;
  int d_ptCloudSenceReflectorsXyz_size[2];
  emxArray_cell_wrap_0_64x1 r0;
  boolean_T isInvalid_data[64];
  double pcSecondRange_idx_0;
  double pcSecondRange_idx_1;
  int extremePoints_size_idx_1;
  int extremePoints_size_idx_0;
  emxArray_real_T *pdistOutput;
  emxArray_real_T *Z;
  emxArray_real_T *extremePointsInds;
  emxArray_real_T *b;
  cell_wrap_0 reshapes[2];
  cell_wrap_0 b_reshapes[2];
  cell_wrap_0 c_reshapes[2];
  emxArray_real_T *d_reshapes;
  int ptCloudShovelReflectorsXyz_size[2];
  int q;
  boolean_T isFoundLoaderPc;
  boolean_T empty_non_axis_sizes;
  signed char input_sizes_idx_1;
  int y_size[2];
  boolean_T guard1 = false;
  int i2;
  emxArray_real_T *r1;
  int i3;
  unsigned long long u0;
  double modelErr1_data[64];
  unsigned long long u1;
  unsigned long long u2;
  int iidx_size[1];
  double b_kmeansC[6];
  double c_v[6];
  emxArray_real_T *r2;
  double extremePoints_data[128];
  emxArray_real_T *r3;
  emxArray_real_T *r4;
  emxArray_real_T *y;
  boolean_T exitg1;
  emxArray_real_T *A;
  int vk;
  double tempXs_data[2];
  signed char input_sizes_idx_0;
  emxArray_real_T *Atranspose;
  double b_y[4];
  emxArray_real_T *c_y;
  double a[4];
  unsigned int unnamed_idx_1;
  double b_extremePoints_data[130];
  double b_modelErr1_data[128];
  int modelErr1_size[1];
  emxArray_real_T c_modelErr1_data;
  double d_modelErr1_data[64];
  double tmp_data[99];
  unsigned long long b_tmp_data[33];
  signed char c_tmp_data[32];
  unsigned long long d_tmp_data[32];
  double e_tmp_data[96];

  /* 'SmartLoader:4' coder.cstructname(configParams, 'SmartLoaderConfigParam'); */
  /*  Parameters  */
  /* 'SmartLoader:7' percisionMode = 'double'; */
  /*   */
  /* 'SmartLoader:12' if ~SmartLoaderGlobal.isInitialized */
  if (!SD->pd->SmartLoaderGlobal.isInitialized) {
    /* 'SmartLoader:13' SmartLoaderGlobalInit(); */
    SmartLoaderGlobalInit(SD);
  }

  /* 'SmartLoader:16' if ~coder.target('Matlab') */
  /* 'SmartLoader:17' debugPtCloudSenceXyz = zeros(0,0,'uint8'); */
  /* 'SmartLoader:18' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); */
  /* 'SmartLoader:21' ptCloudObjXyz = xyz; */
  /* 'SmartLoader:22' ptCloudObjIntensity = intensity; */
  /* 'SmartLoader:24' smartLoaderStruct = GetSmartLoaderStruct(); */
  smartLoaderStruct->loaderLoc[0] = 0.0;
  smartLoaderStruct->shvelLoc[0] = 0.0;
  smartLoaderStruct->shovelHeadingVec[0] = 0.0;
  smartLoaderStruct->loaderLoc[1] = 0.0;
  smartLoaderStruct->shvelLoc[1] = 0.0;
  smartLoaderStruct->shovelHeadingVec[1] = 0.0;
  smartLoaderStruct->loaderLoc[2] = 0.0;
  smartLoaderStruct->shvelLoc[2] = 0.0;
  smartLoaderStruct->shovelHeadingVec[2] = 0.0;
  smartLoaderStruct->shovelHeadingVec2D[0] = 0.0;
  smartLoaderStruct->shovelHeadingVec2D[1] = 0.0;
  smartLoaderStruct->loaderYawAngleDeg = 0.0;
  smartLoaderStruct->loaderYawAngleStatus = false;
  smartLoaderStruct->status = false;

  /* 'SmartLoader:25' coder.cstructname(smartLoaderStruct, 'SmartLoaderStruct'); */
  /*  Align the point cloud to the sensor */
  /* 'SmartLoader:28' if configParams.useExternalProjectionMatrix */
  if (configParams->useExternalProjectionMatrix) {
    /* 'SmartLoader:29' trans = [configParams.externalProjectionMatrix, [0 0 0 1]']; */
    /* 'SmartLoader:30' pcTrans = cast(TransformPointsForward3DAffineCompiledVersion(trans, ptCloudObjXyz), percisionMode); */
    /*  The function transform 3d points using affine transformation  */
    /*  This implementation is special impelmentation for coder that supposes to run faster then projective transformation */
    /* 'TransformPointsForward3DAffineCompiledVersion:5' coder.inline('always'); */
    /*  Assersions - make sure the data is right */
    /* 'TransformPointsForward3DAffineCompiledVersion:8' assert(isequal(trans(end,:), [0 0 0 1])); */
    /* 'TransformPointsForward3DAffineCompiledVersion:9' assert(isequal(size(trans), [4 4])); */
    /* 'TransformPointsForward3DAffineCompiledVersion:10' assert(isequal(size(src,2), 3)); */
    /*  Comment for coder varsize command - updated on 2/2020 - varsize is non needed for this function because the size is determined from the parent function */
    /* coder.varsize('srcHomogenious', [PcClassificationCompilationConstants.MaxPointCloudSize, 4], [1 0]); */
    /* coder.varsize('dst', [PcClassificationCompilationConstants.MaxPointCloudSize, 3], [1 0]); */
    /*  We shall remove the last line from the affine transformation - the last element is one, therefor there is no need to divide the result by 1 */
    /* 'TransformPointsForward3DAffineCompiledVersion:17' affineTrans = trans(1:3,:)'; */
    for (i1 = 0; i1 < 3; i1++) {
      n = i1 << 2;
      b_configParams[n] = configParams->externalProjectionMatrix[i1];
      b_configParams[1 + n] = configParams->externalProjectionMatrix[i1 + 3];
      b_configParams[2 + n] = configParams->externalProjectionMatrix[i1 + 6];
      b_configParams[3 + n] = configParams->externalProjectionMatrix[i1 + 9];
    }

    b_configParams[12] = 0.0;
    b_configParams[13] = 0.0;
    b_configParams[14] = 0.0;
    b_configParams[15] = 1.0;
    for (i1 = 0; i1 < 4; i1++) {
      n = i1 << 2;
      affineTrans[3 * i1] = b_configParams[n];
      affineTrans[1 + 3 * i1] = b_configParams[1 + n];
      affineTrans[2 + 3 * i1] = b_configParams[2 + n];
    }

    /* 'TransformPointsForward3DAffineCompiledVersion:19' srcHomogenious = coder.nullcopy(zeros(size(src,1), 4)); */
    /* 'TransformPointsForward3DAffineCompiledVersion:20' srcHomogenious(:, 1:3) = src; */
    loop_ub = xyz_size[0];
    for (i1 = 0; i1 < loop_ub; i1++) {
      n = i1 << 2;
      SD->f13.srcHomogenious_data[n] = xyz_data[3 * i1];
      SD->f13.srcHomogenious_data[1 + n] = xyz_data[1 + 3 * i1];
      SD->f13.srcHomogenious_data[2 + n] = xyz_data[2 + 3 * i1];
    }

    /* 'TransformPointsForward3DAffineCompiledVersion:21' srcHomogenious(:, 4) = 1; */
    loop_ub = xyz_size[0];
    for (i1 = 0; i1 < loop_ub; i1++) {
      SD->f13.srcHomogenious_data[3 + (i1 << 2)] = 1.0;
    }

    /* 'TransformPointsForward3DAffineCompiledVersion:23' dst = srcHomogenious * affineTrans; */
    m = xyz_size[0];
    pcTrans_size[0] = xyz_size[0];
    for (i = 0; i < m; i++) {
      for (j = 0; j < 3; j++) {
        n = i << 2;
        SD->f13.pcTrans_data[j + 3 * i] = ((SD->f13.srcHomogenious_data[n] *
          affineTrans[j] + SD->f13.srcHomogenious_data[1 + n] * affineTrans[j +
          3]) + SD->f13.srcHomogenious_data[2 + n] * affineTrans[j + 6]) +
          SD->f13.srcHomogenious_data[3 + n] * affineTrans[j + 9];
      }
    }
  } else {
    /* 'SmartLoader:31' else */
    /* 'SmartLoader:32' transRot3D = CreateRotMat_YawZ(deg2rad(0)) * CreateRotMat_PitchY(deg2rad(0)) * CreateRotMat_RollX(deg2rad(0)); */
    /* 'SmartLoader:33' transRot3D = CreateRotMat_RollX(deg2rad(-1.5)) * CreateRotMat_PitchY(deg2rad(-90)) * transRot3D; */
    /* 'SmartLoader:34' trans = [[transRot3D;0 0 0], [0 0 0 1]']; */
    /* 'SmartLoader:36' if false */
    /* 'SmartLoader:40' else */
    /* 'SmartLoader:41' pcTrans = cast(TransformPointsForward3DAffineCompiledVersion(trans, ptCloudObjXyz), percisionMode); */
    /*  The function transform 3d points using affine transformation  */
    /*  This implementation is special impelmentation for coder that supposes to run faster then projective transformation */
    /* 'TransformPointsForward3DAffineCompiledVersion:5' coder.inline('always'); */
    /*  Assersions - make sure the data is right */
    /* 'TransformPointsForward3DAffineCompiledVersion:8' assert(isequal(trans(end,:), [0 0 0 1])); */
    /* 'TransformPointsForward3DAffineCompiledVersion:9' assert(isequal(size(trans), [4 4])); */
    /* 'TransformPointsForward3DAffineCompiledVersion:10' assert(isequal(size(src,2), 3)); */
    /*  Comment for coder varsize command - updated on 2/2020 - varsize is non needed for this function because the size is determined from the parent function */
    /* coder.varsize('srcHomogenious', [PcClassificationCompilationConstants.MaxPointCloudSize, 4], [1 0]); */
    /* coder.varsize('dst', [PcClassificationCompilationConstants.MaxPointCloudSize, 3], [1 0]); */
    /*  We shall remove the last line from the affine transformation - the last element is one, therefor there is no need to divide the result by 1 */
    /* 'TransformPointsForward3DAffineCompiledVersion:17' affineTrans = trans(1:3,:)'; */
    /* 'TransformPointsForward3DAffineCompiledVersion:19' srcHomogenious = coder.nullcopy(zeros(size(src,1), 4)); */
    /* 'TransformPointsForward3DAffineCompiledVersion:20' srcHomogenious(:, 1:3) = src; */
    loop_ub = xyz_size[0];
    for (i1 = 0; i1 < loop_ub; i1++) {
      n = i1 << 2;
      SD->f13.srcHomogenious_data[n] = xyz_data[3 * i1];
      SD->f13.srcHomogenious_data[1 + n] = xyz_data[1 + 3 * i1];
      SD->f13.srcHomogenious_data[2 + n] = xyz_data[2 + 3 * i1];
    }

    /* 'TransformPointsForward3DAffineCompiledVersion:21' srcHomogenious(:, 4) = 1; */
    loop_ub = xyz_size[0];
    for (i1 = 0; i1 < loop_ub; i1++) {
      SD->f13.srcHomogenious_data[3 + (i1 << 2)] = 1.0;
    }

    /* 'TransformPointsForward3DAffineCompiledVersion:23' dst = srcHomogenious * affineTrans; */
    m = xyz_size[0];
    for (i = 0; i < m; i++) {
      for (j = 0; j < 3; j++) {
        n = i << 2;
        SD->f13.pcTrans_data[j + 3 * i] = ((SD->f13.srcHomogenious_data[n] * B[j]
          + SD->f13.srcHomogenious_data[1 + n] * B[j + 3]) +
          SD->f13.srcHomogenious_data[2 + n] * B[j + 6]) +
          SD->f13.srcHomogenious_data[3 + n] * B[j + 9];
      }
    }

    /*  switch between the x,y coordiantes */
    /* 'SmartLoader:45' pcTrans = [pcTrans(:,2), pcTrans(:,1), pcTrans(:,3)]; */
    loop_ub = xyz_size[0];
    for (i1 = 0; i1 < loop_ub; i1++) {
      SD->f13.ptCloudSenceXyz_data[3 * i1] = SD->f13.pcTrans_data[1 + 3 * i1];
    }

    loop_ub = xyz_size[0];
    for (i1 = 0; i1 < loop_ub; i1++) {
      SD->f13.ptCloudSenceXyz_data[1 + 3 * i1] = SD->f13.pcTrans_data[3 * i1];
    }

    loop_ub = xyz_size[0];
    for (i1 = 0; i1 < loop_ub; i1++) {
      n = 2 + 3 * i1;
      SD->f13.ptCloudSenceXyz_data[n] = SD->f13.pcTrans_data[n];
    }

    pcTrans_size[0] = xyz_size[0];
    loop_ub = 3 * xyz_size[0];
    if (0 <= loop_ub - 1) {
      memcpy(&SD->f13.pcTrans_data[0], &SD->f13.ptCloudSenceXyz_data[0],
             (unsigned int)(loop_ub * (int)sizeof(double)));
    }

    /*  figure, PlotPointCloud(pcTrans); title('pc aligned') */
  }

  /* 'SmartLoader:49' indices = pcTrans(:,1) >= configParams.xyzLimits(1,1) & pcTrans(:,1) <= configParams.xyzLimits(1,2) & ... */
  /* 'SmartLoader:50'     pcTrans(:,2) >= configParams.xyzLimits(2,1) & pcTrans(:,2) <= configParams.xyzLimits(2,2) & ... */
  /* 'SmartLoader:51'     pcTrans(:,3) >= configParams.xyzLimits(3,1) & pcTrans(:,3) <= configParams.xyzLimits(3,2); */
  loop_ub = pcTrans_size[0];
  for (i1 = 0; i1 < loop_ub; i1++) {
    zMedian = SD->f13.pcTrans_data[3 * i1];
    singleReflectorRangeLimitMeter = SD->f13.pcTrans_data[1 + 3 * i1];
    SD->f13.temp30_data[i1] = ((zMedian >= configParams->xyzLimits[0]) &&
      (zMedian <= configParams->xyzLimits[1]) && (singleReflectorRangeLimitMeter
      >= configParams->xyzLimits[2]) && (singleReflectorRangeLimitMeter <=
      configParams->xyzLimits[3]) && (SD->f13.pcTrans_data[2 + 3 * i1] >=
      configParams->xyzLimits[4]));
  }

  loop_ub = pcTrans_size[0];
  for (i1 = 0; i1 < loop_ub; i1++) {
    SD->f13.h_tmp_data[i1] = (SD->f13.pcTrans_data[2 + 3 * i1] <=
      configParams->xyzLimits[5]);
  }

  /* 'SmartLoader:53' ptCloudSenceXyz = pcTrans(indices,:); */
  n = pcTrans_size[0] - 1;
  trueCount = 0;
  for (i = 0; i <= n; i++) {
    if (SD->f13.temp30_data[i] && SD->f13.h_tmp_data[i]) {
      trueCount++;
    }
  }

  m = 0;
  for (i = 0; i <= n; i++) {
    if (SD->f13.temp30_data[i] && SD->f13.h_tmp_data[i]) {
      SD->f13.iidx_data[m] = i + 1;
      m++;
    }
  }

  ptCloudSenceXyz_size[1] = 3;
  ptCloudSenceXyz_size[0] = trueCount;
  for (i1 = 0; i1 < trueCount; i1++) {
    n = 3 * (SD->f13.iidx_data[i1] - 1);
    SD->f13.ptCloudSenceXyz_data[3 * i1] = SD->f13.pcTrans_data[n];
    SD->f13.ptCloudSenceXyz_data[1 + 3 * i1] = SD->f13.pcTrans_data[1 + n];
    SD->f13.ptCloudSenceXyz_data[2 + 3 * i1] = SD->f13.pcTrans_data[2 + n];
  }

  /* 'SmartLoader:54' ptCloudSenceIntensity = ptCloudObjIntensity(indices,:); */
  n = pcTrans_size[0] - 1;
  j = 0;
  for (i = 0; i <= n; i++) {
    if (SD->f13.temp30_data[i] && SD->f13.h_tmp_data[i]) {
      j++;
    }
  }

  m = 0;
  for (i = 0; i <= n; i++) {
    if (SD->f13.temp30_data[i] && SD->f13.h_tmp_data[i]) {
      SD->f13.c_tmp_data[m] = i + 1;
      m++;
    }
  }

  /*  figure, PlotPointCloud([ptCloudSenceXyz double(ptCloudSenceIntensity)]); */
  /* 'SmartLoader:57' if size(ptCloudSenceXyz,1) < configParams.minNumPointsInPc */
  if (trueCount < configParams->minNumPointsInPc) {
    /* 'SmartLoader:58' if coder.target('Matlab') */
    /* 'SmartLoader:61' heightMap_res = zeros(0,0,'single'); */
    heightMap_res_size[1] = 0;
    heightMap_res_size[0] = 0;

    /* 'SmartLoader:62' if coder.target('Matlab') */
    /* 'SmartLoader:65' else */
    /* 'SmartLoader:66' debugPtCloudSenceXyz = zeros(0,0,'uint8'); */
    /* 'SmartLoader:67' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); */
  } else {
    /* 'SmartLoader:72' if configParams.debugIsPlayerMode */
    /*  Create height map  */
    /* 'SmartLoader:85' [smartLoaderStruct, heightMap_res, deg_loaderProperties] = CreateHeightMap(smartLoaderStruct, ptCloudSenceXyz); */
    CreateHeightMap(SD, smartLoaderStruct, SD->f13.ptCloudSenceXyz_data,
                    ptCloudSenceXyz_size, heightMap_res_data, heightMap_res_size);

    /*  Estimate loader by circled reflector */
    /* 'SmartLoader:88' temp30 = ptCloudSenceIntensity > configParams.minimumIntensityReflectorValue; */
    for (i1 = 0; i1 < j; i1++) {
      SD->f13.temp30_data[i1] = (intensity_data[SD->f13.c_tmp_data[i1] - 1] >
        configParams->minimumIntensityReflectorValue);
    }

    /* 'SmartLoader:89' ptCloudSenceReflectorsXyz = ptCloudSenceXyz(temp30,:); */
    n = j - 1;
    trueCount = 0;
    for (i = 0; i <= n; i++) {
      if (SD->f13.temp30_data[i]) {
        trueCount++;
      }
    }

    m = 0;
    for (i = 0; i <= n; i++) {
      if (SD->f13.temp30_data[i]) {
        SD->f13.d_tmp_data[m] = i + 1;
        m++;
      }
    }

    for (i1 = 0; i1 < trueCount; i1++) {
      n = 3 * (SD->f13.d_tmp_data[i1] - 1);
      SD->f13.ptCloudSenceReflectorsXyz_data[3 * i1] =
        SD->f13.ptCloudSenceXyz_data[n];
      SD->f13.ptCloudSenceReflectorsXyz_data[1 + 3 * i1] =
        SD->f13.ptCloudSenceXyz_data[1 + n];
      SD->f13.ptCloudSenceReflectorsXyz_data[2 + 3 * i1] =
        SD->f13.ptCloudSenceXyz_data[2 + n];
    }

    /* 'SmartLoader:90' ptCloudSenceReflectorsIntensity = ptCloudSenceIntensity(temp30,:); */
    /*  figure, PlotPointCloud(ptCloudSenceReflectors); */
    /*  figure, PlotPointCloud([ptCloudSenceReflectorsXyz ptCloudSenceReflectorsIntensity]); */
    /* 'SmartLoader:94' if size(ptCloudSenceReflectorsXyz,1) < configParams.minPointsForReflector */
    if (!(trueCount < configParams->minPointsForReflector)) {
      /*  Determine whether or not the reflectors are good enough for estimating two representing ellipses */
      /*  first we have to determine whehter the reflector reside on the (loader) or (loader and shovel) */
      /*  TODO - align the points according to the major and minor axis !!! */
      /*  I arbitrary set twich the reflector range as a limit - this number determine whether the sensor detected both the shovel and the loader. */
      /* 'SmartLoader:112' singleReflectorRangeLimitMeter = 2 * configParams.loaderReflectorDiameterMeter; */
      singleReflectorRangeLimitMeter = 2.0 *
        configParams->loaderReflectorDiameterMeter;

      /* r = RangeCompiledVersion(ptCloudSenceReflectors.Location); */
      /* 'SmartLoader:115' r = RangeCompiledVersion(ptCloudSenceReflectorsXyz); */
      /* 'RangeCompiledVersion:4' coder.inline('always'); */
      /* 'RangeCompiledVersion:7' output = max(input) - min(input); */
      n = 3 * (SD->f13.d_tmp_data[0] - 1);
      v[0] = SD->f13.ptCloudSenceXyz_data[n];
      zMedian = SD->f13.ptCloudSenceXyz_data[1 + n];
      v[1] = zMedian;
      for (i = 2; i <= trueCount; i++) {
        n = 3 * (SD->f13.d_tmp_data[i - 1] - 1);
        if ((!rtIsNaN(SD->f13.ptCloudSenceXyz_data[n])) && (rtIsNaN(v[0]) || (v
              [0] < SD->f13.ptCloudSenceXyz_data[n]))) {
          v[0] = SD->f13.ptCloudSenceXyz_data[n];
        }

        b_tmp = SD->f13.ptCloudSenceXyz_data[1 + n];
        if ((!rtIsNaN(b_tmp)) && (rtIsNaN(v[1]) || (v[1] < b_tmp))) {
          v[1] = b_tmp;
        }
      }

      minval_idx_0 = SD->f13.ptCloudSenceXyz_data[3 * (SD->f13.d_tmp_data[0] - 1)];
      for (i = 2; i <= trueCount; i++) {
        n = 3 * (SD->f13.d_tmp_data[i - 1] - 1);
        if ((!rtIsNaN(SD->f13.ptCloudSenceXyz_data[n])) && (rtIsNaN(minval_idx_0)
             || (minval_idx_0 > SD->f13.ptCloudSenceXyz_data[n]))) {
          minval_idx_0 = SD->f13.ptCloudSenceXyz_data[n];
        }

        b_tmp = SD->f13.ptCloudSenceXyz_data[1 + n];
        if ((!rtIsNaN(b_tmp)) && (rtIsNaN(zMedian) || (zMedian > b_tmp))) {
          zMedian = b_tmp;
        }
      }

      /* 'SmartLoader:116' if all(r(1:2) < singleReflectorRangeLimitMeter) */
      b_v[0] = (v[0] - minval_idx_0 < singleReflectorRangeLimitMeter);
      b_v[1] = (v[1] - zMedian < singleReflectorRangeLimitMeter);
      if (!all(b_v)) {
        /* 'SmartLoader:129' else */
        /*  Found both shovel and loader reflectors */
        /* [kmeansIdx,kmeansC,kmeanssumd,kmeansDistanceMat] = kmeans(ptCloudSenceReflectors.Location, 2, 'Replicates', 5); */
        /* 'SmartLoader:132' [kmeansIdx,kmeansC,kmeanssumd,kmeansDistanceMat] = kmeans(ptCloudSenceReflectorsXyz, 2, 'Replicates', 5); */
        ptCloudSenceReflectorsXyz_size[1] = 3;
        ptCloudSenceReflectorsXyz_size[0] = trueCount;
        loop_ub = 3 * trueCount;
        if (0 <= loop_ub - 1) {
          memcpy(&SD->f13.b_ptCloudSenceReflectorsXyz_data[0],
                 &SD->f13.ptCloudSenceReflectorsXyz_data[0], (unsigned int)
                 (loop_ub * (int)sizeof(double)));
        }

        kmeans(SD, SD->f13.b_ptCloudSenceReflectorsXyz_data,
               ptCloudSenceReflectorsXyz_size, SD->f13.kmeansIdx_data,
               kmeansIdx_size, kmeansC, distanceToKmeansClusterMeter,
               SD->f13.kmeansDistanceMat_data, pcTrans_size);

        /* 'SmartLoader:133' [~, minInd] = min(kmeansDistanceMat,[],1); */
        /* 'SmartLoader:133' ~ */
        /* 'SmartLoader:135' if coder.target('Matlab') && false */
        /*  Get the first and the second point cloud */
        /*  Clean the first and the second point cloud with everything larger or smaller than this threashold configParams.reflectorMaxZaxisDistanceForOutlierMeter */
        /* pcFirstOrg = select(ptCloudSenceReflectors, find(kmeansIdx == 1)); */
        /* 'SmartLoader:152' pcFirstOrgXyz = ptCloudSenceReflectorsXyz(kmeansIdx == 1,:); */
        /*  figure, PlotPointCloud(pcFirst); */
        /* 'SmartLoader:155' [pcFirstXyz] = FilterPointCloudAccordingToZdifferences(pcFirstOrgXyz, configParams.reflectorMaxZaxisDistanceForOutlierMeter); */
        n = kmeansIdx_size[0] - 1;
        trueCount = 0;
        for (i = 0; i <= n; i++) {
          if (SD->f13.kmeansIdx_data[i] == 1.0) {
            trueCount++;
          }
        }

        m = 0;
        for (i = 0; i <= n; i++) {
          if (SD->f13.kmeansIdx_data[i] == 1.0) {
            SD->f13.e_tmp_data[m] = i + 1;
            m++;
          }
        }

        b_ptCloudSenceReflectorsXyz_size[1] = 3;
        b_ptCloudSenceReflectorsXyz_size[0] = trueCount;
        for (i1 = 0; i1 < trueCount; i1++) {
          n = 3 * (SD->f13.e_tmp_data[i1] - 1);
          SD->f13.b_ptCloudSenceReflectorsXyz_data[3 * i1] =
            SD->f13.ptCloudSenceReflectorsXyz_data[n];
          SD->f13.b_ptCloudSenceReflectorsXyz_data[1 + 3 * i1] =
            SD->f13.ptCloudSenceReflectorsXyz_data[1 + n];
          SD->f13.b_ptCloudSenceReflectorsXyz_data[2 + 3 * i1] =
            SD->f13.ptCloudSenceReflectorsXyz_data[2 + n];
        }

        FilterPointCloudAccordingToZdifferences(SD,
          SD->f13.b_ptCloudSenceReflectorsXyz_data,
          b_ptCloudSenceReflectorsXyz_size,
          configParams->reflectorMaxZaxisDistanceForOutlierMeter,
          SD->f13.pcTrans_data, pcTrans_size);

        /*  figure, PlotPointCloud(pcFirst); */
        /* pcFirstRange = RangeCompiledVersion(pcFirst.Location); */
        /* 'SmartLoader:159' pcFirstRange = RangeCompiledVersion(pcFirstXyz); */
        /* 'RangeCompiledVersion:4' coder.inline('always'); */
        /* 'RangeCompiledVersion:7' output = max(input) - min(input); */
        m = pcTrans_size[0];
        pcFirstRange_idx_0 = SD->f13.pcTrans_data[0];
        pcFirstRange_idx_1 = SD->f13.pcTrans_data[1];
        for (i = 2; i <= m; i++) {
          n = 3 * (i - 1);
          if ((!rtIsNaN(SD->f13.pcTrans_data[n])) && (rtIsNaN(pcFirstRange_idx_0)
               || (pcFirstRange_idx_0 < SD->f13.pcTrans_data[n]))) {
            pcFirstRange_idx_0 = SD->f13.pcTrans_data[n];
          }

          b_tmp = SD->f13.pcTrans_data[1 + n];
          if ((!rtIsNaN(b_tmp)) && (rtIsNaN(pcFirstRange_idx_1) ||
               (pcFirstRange_idx_1 < b_tmp))) {
            pcFirstRange_idx_1 = b_tmp;
          }
        }

        m = pcTrans_size[0];
        minval_idx_0 = SD->f13.pcTrans_data[0];
        zMedian = SD->f13.pcTrans_data[1];
        for (i = 2; i <= m; i++) {
          n = 3 * (i - 1);
          if ((!rtIsNaN(SD->f13.pcTrans_data[n])) && (rtIsNaN(minval_idx_0) ||
               (minval_idx_0 > SD->f13.pcTrans_data[n]))) {
            minval_idx_0 = SD->f13.pcTrans_data[n];
          }

          b_tmp = SD->f13.pcTrans_data[1 + n];
          if ((!rtIsNaN(b_tmp)) && (rtIsNaN(zMedian) || (zMedian > b_tmp))) {
            zMedian = b_tmp;
          }
        }

        pcFirstRange_idx_0 -= minval_idx_0;
        pcFirstRange_idx_1 -= zMedian;

        /* pcSecondOrg = select(ptCloudSenceReflectors, find(kmeansIdx == 2)); */
        /* 'SmartLoader:162' pcSecondOrgXyz = ptCloudSenceReflectorsXyz(kmeansIdx == 2,:); */
        /*  figure, PlotPointCloud(pcSecondOrg); */
        /* 'SmartLoader:165' [pcSecondXyz] = FilterPointCloudAccordingToZdifferences(pcSecondOrgXyz, configParams.reflectorMaxZaxisDistanceForOutlierMeter); */
        n = kmeansIdx_size[0] - 1;
        trueCount = 0;
        for (i = 0; i <= n; i++) {
          if (SD->f13.kmeansIdx_data[i] == 2.0) {
            trueCount++;
          }
        }

        m = 0;
        for (i = 0; i <= n; i++) {
          if (SD->f13.kmeansIdx_data[i] == 2.0) {
            SD->f13.g_tmp_data[m] = i + 1;
            m++;
          }
        }

        d_ptCloudSenceReflectorsXyz_size[1] = 3;
        d_ptCloudSenceReflectorsXyz_size[0] = trueCount;
        for (i1 = 0; i1 < trueCount; i1++) {
          n = 3 * (SD->f13.g_tmp_data[i1] - 1);
          SD->f13.b_ptCloudSenceReflectorsXyz_data[3 * i1] =
            SD->f13.ptCloudSenceReflectorsXyz_data[n];
          SD->f13.b_ptCloudSenceReflectorsXyz_data[1 + 3 * i1] =
            SD->f13.ptCloudSenceReflectorsXyz_data[1 + n];
          SD->f13.b_ptCloudSenceReflectorsXyz_data[2 + 3 * i1] =
            SD->f13.ptCloudSenceReflectorsXyz_data[2 + n];
        }

        FilterPointCloudAccordingToZdifferences(SD,
          SD->f13.b_ptCloudSenceReflectorsXyz_data,
          d_ptCloudSenceReflectorsXyz_size,
          configParams->reflectorMaxZaxisDistanceForOutlierMeter,
          SD->f13.ptCloudSenceXyz_data, ptCloudSenceXyz_size);

        /*  figure, PlotPointCloud(pcSecond); */
        /* pcSecondRange = RangeCompiledVersion(pcSecond.Location); */
        /* 'SmartLoader:168' pcSecondRange = RangeCompiledVersion(pcSecondXyz); */
        /* 'RangeCompiledVersion:4' coder.inline('always'); */
        /* 'RangeCompiledVersion:7' output = max(input) - min(input); */
        m = ptCloudSenceXyz_size[0];
        pcSecondRange_idx_0 = SD->f13.ptCloudSenceXyz_data[0];
        pcSecondRange_idx_1 = SD->f13.ptCloudSenceXyz_data[1];
        for (i = 2; i <= m; i++) {
          n = 3 * (i - 1);
          if ((!rtIsNaN(SD->f13.ptCloudSenceXyz_data[n])) && (rtIsNaN
               (pcSecondRange_idx_0) || (pcSecondRange_idx_0 <
                SD->f13.ptCloudSenceXyz_data[n]))) {
            pcSecondRange_idx_0 = SD->f13.ptCloudSenceXyz_data[n];
          }

          b_tmp = SD->f13.ptCloudSenceXyz_data[1 + n];
          if ((!rtIsNaN(b_tmp)) && (rtIsNaN(pcSecondRange_idx_1) ||
               (pcSecondRange_idx_1 < b_tmp))) {
            pcSecondRange_idx_1 = b_tmp;
          }
        }

        m = ptCloudSenceXyz_size[0];
        minval_idx_0 = SD->f13.ptCloudSenceXyz_data[0];
        zMedian = SD->f13.ptCloudSenceXyz_data[1];
        for (i = 2; i <= m; i++) {
          n = 3 * (i - 1);
          if ((!rtIsNaN(SD->f13.ptCloudSenceXyz_data[n])) && (rtIsNaN
               (minval_idx_0) || (minval_idx_0 > SD->f13.ptCloudSenceXyz_data[n])))
          {
            minval_idx_0 = SD->f13.ptCloudSenceXyz_data[n];
          }

          b_tmp = SD->f13.ptCloudSenceXyz_data[1 + n];
          if ((!rtIsNaN(b_tmp)) && (rtIsNaN(zMedian) || (zMedian > b_tmp))) {
            zMedian = b_tmp;
          }
        }

        pcSecondRange_idx_0 -= minval_idx_0;
        pcSecondRange_idx_1 -= zMedian;

        /*     %% Determine which cluster is the loader or the shovel */
        /*  First stradegy - the loader point cloud reflector is the cluster center closest to the previous loader location */
        /*  This is simply for letting matlab coder know that ptCloudLoaderReflectors and  ptCloudShovelReflectors will eventually set to a certain value */
        /*  we give them an initial value that will change during the code  */
        /* ptCloudLoaderReflectors = pcSecond; */
        /* 'SmartLoader:176' ptCloudLoaderReflectorsXyz = pcSecondXyz; */
        trueCount = ptCloudSenceXyz_size[0];
        j = ptCloudSenceXyz_size[1] * ptCloudSenceXyz_size[0];
        if (0 <= j - 1) {
          memcpy(&SD->f13.ptCloudSenceReflectorsXyz_data[0],
                 &SD->f13.ptCloudSenceXyz_data[0], (unsigned int)(j * (int)
                  sizeof(double)));
        }

        /* ptCloudShovelReflectors = pcFirst; */
        /* 'SmartLoader:178' ptCloudShovelReflectorsXyz = pcFirstXyz; */
        ptCloudShovelReflectorsXyz_size[1] = 3;
        ptCloudShovelReflectorsXyz_size[0] = pcTrans_size[0];
        i = pcTrans_size[1] * pcTrans_size[0];
        if (0 <= i - 1) {
          memcpy(&SD->f13.ptCloudShovelReflectorsXyz_data[0],
                 &SD->f13.pcTrans_data[0], (unsigned int)(i * (int)sizeof(double)));
        }

        /* 'SmartLoader:180' isFoundLoaderPc = false; */
        isFoundLoaderPc = false;

        /* 'SmartLoader:181' if ~isempty(SmartLoaderGlobal.loaderLocHistory) */
        if (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] != 0) {
          /* 'SmartLoader:182' if size(SmartLoaderGlobal.loaderLocHistory,1) >= 2 && ... */
          /* 'SmartLoader:183'                 configParams.timeTagMs - SmartLoaderGlobal.loaderTimeTatHistoryMs(end) < configParams.maximumTimeTagDiffMs && ... */
          /* 'SmartLoader:184'                 configParams.timeTagMs - SmartLoaderGlobal.loaderTimeTatHistoryMs(end-1) < configParams.maximumTimeTagDiffMs */
          guard1 = false;
          if (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] >= 2) {
            u0 = SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[SD->
              pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] - 1];
            u1 = configParams->timeTagMs - u0;
            if (u1 < configParams->maximumTimeTagDiffMs) {
              u2 = SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[SD->
                pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] - 2];
              if (configParams->timeTagMs - u2 <
                  configParams->maximumTimeTagDiffMs) {
                /*  Estiamte where loader location should be according to the previous locations of the loader */
                /* 'SmartLoader:187' v = (SmartLoaderGlobal.loaderLocHistory(end,:) - SmartLoaderGlobal.loaderLocHistory(end-1,:)) / ... */
                /* 'SmartLoader:188'                 double(SmartLoaderGlobal.loaderTimeTatHistoryMs(end) - SmartLoaderGlobal.loaderTimeTatHistoryMs(end-1)); */
                zMedian = (double)(u0 - u2);

                /* . */
                /* 'SmartLoader:190' estimatedLoaderLoc = SmartLoaderGlobal.loaderLocHistory(end,:) + .... */
                /* 'SmartLoader:191'                 v * double(configParams.timeTagMs - SmartLoaderGlobal.loaderTimeTatHistoryMs(end)); */
                n = 3 * (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] - 2);
                v[0] = SD->pd->SmartLoaderGlobal.loaderLocHistory.data[3 *
                  (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] - 1)] +
                  (SD->pd->SmartLoaderGlobal.loaderLocHistory.data[3 * (SD->
                    pd->SmartLoaderGlobal.loaderLocHistory.size[0] - 1)] -
                   SD->pd->SmartLoaderGlobal.loaderLocHistory.data[n]) / zMedian
                  * (double)u1;
                v[1] = SD->pd->SmartLoaderGlobal.loaderLocHistory.data[1 + 3 *
                  (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] - 1)] +
                  (SD->pd->SmartLoaderGlobal.loaderLocHistory.data[1 + 3 *
                   (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] - 1)] -
                   SD->pd->SmartLoaderGlobal.loaderLocHistory.data[1 + n]) /
                  zMedian * (double)u1;
                v[2] = SD->pd->SmartLoaderGlobal.loaderLocHistory.data[2 + 3 *
                  (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] - 1)] +
                  (SD->pd->SmartLoaderGlobal.loaderLocHistory.data[2 + 3 *
                   (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] - 1)] -
                   SD->pd->SmartLoaderGlobal.loaderLocHistory.data[2 + n]) /
                  zMedian * (double)u1;
              } else {
                guard1 = true;
              }
            } else {
              guard1 = true;
            }
          } else {
            guard1 = true;
          }

          if (guard1) {
            /* 'SmartLoader:192' else */
            /* 'SmartLoader:193' estimatedLoaderLoc = SmartLoaderGlobal.loaderLocHistory(end,:); */
            n = 3 * (SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] - 1);
            v[0] = SD->pd->SmartLoaderGlobal.loaderLocHistory.data[n];
            v[1] = SD->pd->SmartLoaderGlobal.loaderLocHistory.data[1 + n];
            v[2] = SD->pd->SmartLoaderGlobal.loaderLocHistory.data[2 + n];
          }

          /* 'SmartLoader:196' if ~isempty(estimatedLoaderLoc) */
          /* 'SmartLoader:198' distanceToKmeansClusterMeter = vecnorm((kmeansC - [estimatedLoaderLoc; estimatedLoaderLoc])'); */
          for (i1 = 0; i1 < 3; i1++) {
            n = i1 << 1;
            c_v[n] = v[i1];
            m = 1 + n;
            c_v[m] = v[i1];
            b_kmeansC[n] = kmeansC[i1] - c_v[n];
            b_kmeansC[m] = kmeansC[i1 + 3] - c_v[m];
          }

          vecnorm(b_kmeansC, distanceToKmeansClusterMeter);

          /*  There are 38 cm distance between the two reflectors, */
          /*  we'd like ensure the previous location of the loader reside within a 10cm margin             */
          /* 'SmartLoader:202' if any(distanceToKmeansClusterMeter - configParams.previousLoaderLocationToCurrentLocationMaximumDistanceMeter < 0) */
          b_v[0] = (distanceToKmeansClusterMeter[0] -
                    configParams->previousLoaderLocationToCurrentLocationMaximumDistanceMeter
                    < 0.0);
          b_v[1] = (distanceToKmeansClusterMeter[1] -
                    configParams->previousLoaderLocationToCurrentLocationMaximumDistanceMeter
                    < 0.0);
          if (any(b_v)) {
            /* 'SmartLoader:204' if distanceToKmeansClusterMeter(1) < distanceToKmeansClusterMeter(2) */
            if (distanceToKmeansClusterMeter[0] < distanceToKmeansClusterMeter[1])
            {
              /* ptCloudLoaderReflectors = pcFirst; ptCloudShovelReflectors = pcSecond; */
              /* 'SmartLoader:206' ptCloudLoaderReflectorsXyz = pcFirstXyz; */
              trueCount = pcTrans_size[0];
              if (0 <= i - 1) {
                memcpy(&SD->f13.ptCloudSenceReflectorsXyz_data[0],
                       &SD->f13.pcTrans_data[0], (unsigned int)(i * (int)sizeof
                        (double)));
              }

              /* 'SmartLoader:206' ptCloudShovelReflectorsXyz = pcSecondXyz; */
              ptCloudShovelReflectorsXyz_size[1] = 3;
              ptCloudShovelReflectorsXyz_size[0] = ptCloudSenceXyz_size[0];
              if (0 <= j - 1) {
                memcpy(&SD->f13.ptCloudShovelReflectorsXyz_data[0],
                       &SD->f13.ptCloudSenceXyz_data[0], (unsigned int)(j * (int)
                        sizeof(double)));
              }
            } else {
              /* 'SmartLoader:207' else */
              /* ptCloudLoaderReflectors = pcSecond; ptCloudShovelReflectors = pcFirst; */
              /* 'SmartLoader:209' ptCloudLoaderReflectorsXyz = pcSecondXyz; */
              /* 'SmartLoader:209' ptCloudShovelReflectorsXyz = pcFirstXyz; */
            }

            /* 'SmartLoader:212' isFoundLoaderPc = true; */
            isFoundLoaderPc = true;
          }
        }

        /* 'SmartLoader:217' if ~isFoundLoaderPc */
        if (!isFoundLoaderPc) {
          /*  Stradegy - we know that the loader height from the ground plane is fix number, however the shovel is mostly */
          /*  reside below this height - therefor we'll determine the loader and the shovel according to the loader minimum height */
          /*  Find the point with the median z coordiante */
          /* [~, I] = sort(pcFirst.Location(:,3)); */
          /* 'SmartLoader:223' [~, I] = sort(pcFirstXyz(:,3)); */
          kmeansIdx_size[0] = pcTrans_size[0];
          loop_ub = pcTrans_size[0];
          for (i1 = 0; i1 < loop_ub; i1++) {
            SD->f13.kmeansIdx_data[i1] = SD->f13.pcTrans_data[2 + 3 * i1];
          }

          b_sort(SD, SD->f13.kmeansIdx_data, kmeansIdx_size, SD->f13.iidx_data,
                 iidx_size);

          /* 'SmartLoader:223' ~ */
          /* 'SmartLoader:224' zMedianPointFirst = pcFirstXyz(floor(size(I,1)/2),:); */
          /* 'SmartLoader:225' [pcFirstDistanceToPlane, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, zMedianPointFirst); */
          CalcPlaneToPointDistance(configParams->planeModelParameters, *(double
            (*)[3])&SD->f13.pcTrans_data[3 * ((int)std::floor((double)iidx_size
            [0] / 2.0) - 1)], &singleReflectorRangeLimitMeter,
            &empty_non_axis_sizes);

          /* 'SmartLoader:225' ~ */
          /* 'SmartLoader:227' [~, I] = sort(pcSecondXyz(:,3)); */
          kmeansIdx_size[0] = ptCloudSenceXyz_size[0];
          loop_ub = ptCloudSenceXyz_size[0];
          for (i1 = 0; i1 < loop_ub; i1++) {
            SD->f13.kmeansIdx_data[i1] = SD->f13.ptCloudSenceXyz_data[2 + 3 * i1];
          }

          b_sort(SD, SD->f13.kmeansIdx_data, kmeansIdx_size, SD->f13.iidx_data,
                 iidx_size);

          /* 'SmartLoader:227' ~ */
          /* 'SmartLoader:228' zMedianPointSecond = pcSecondXyz(floor(size(I,1)/2),:); */
          /* 'SmartLoader:229' [pcSecondDistanceToPlane, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, zMedianPointSecond); */
          CalcPlaneToPointDistance(configParams->planeModelParameters, *(double
            (*)[3])&SD->f13.ptCloudSenceXyz_data[3 * ((int)std::floor((double)
            iidx_size[0] / 2.0) - 1)], &zMedian, &empty_non_axis_sizes);

          /* 'SmartLoader:229' ~ */
          /* 'SmartLoader:231' if pcFirstDistanceToPlane > configParams.minimumDistanceFromLoaderToPlaneMeter &&... */
          /* 'SmartLoader:232'                 pcSecondDistanceToPlane < configParams.minimumDistanceFromLoaderToPlaneMeter */
          if ((singleReflectorRangeLimitMeter >
               configParams->minimumDistanceFromLoaderToPlaneMeter) && (zMedian <
               configParams->minimumDistanceFromLoaderToPlaneMeter)) {
            /* 'SmartLoader:233' ptCloudLoaderReflectorsXyz = pcFirstXyz; */
            trueCount = pcTrans_size[0];
            if (0 <= i - 1) {
              memcpy(&SD->f13.ptCloudSenceReflectorsXyz_data[0],
                     &SD->f13.pcTrans_data[0], (unsigned int)(i * (int)sizeof
                      (double)));
            }

            /* 'SmartLoader:233' ptCloudShovelReflectorsXyz = pcSecondXyz; */
            ptCloudShovelReflectorsXyz_size[1] = 3;
            ptCloudShovelReflectorsXyz_size[0] = ptCloudSenceXyz_size[0];
            if (0 <= j - 1) {
              memcpy(&SD->f13.ptCloudShovelReflectorsXyz_data[0],
                     &SD->f13.ptCloudSenceXyz_data[0], (unsigned int)(j * (int)
                      sizeof(double)));
            }

            /* 'SmartLoader:234' isFoundLoaderPc = true; */
            isFoundLoaderPc = true;
          } else {
            if ((singleReflectorRangeLimitMeter <
                 configParams->minimumDistanceFromLoaderToPlaneMeter) &&
                (zMedian > configParams->minimumDistanceFromLoaderToPlaneMeter))
            {
              /* 'SmartLoader:236' elseif pcFirstDistanceToPlane < configParams.minimumDistanceFromLoaderToPlaneMeter &&... */
              /* 'SmartLoader:237'                 pcSecondDistanceToPlane > configParams.minimumDistanceFromLoaderToPlaneMeter */
              /* 'SmartLoader:238' ptCloudLoaderReflectorsXyz = pcSecondXyz; */
              trueCount = ptCloudSenceXyz_size[0];
              if (0 <= j - 1) {
                memcpy(&SD->f13.ptCloudSenceReflectorsXyz_data[0],
                       &SD->f13.ptCloudSenceXyz_data[0], (unsigned int)(j * (int)
                        sizeof(double)));
              }

              /* 'SmartLoader:238' ptCloudShovelReflectorsXyz = pcFirstXyz; */
              ptCloudShovelReflectorsXyz_size[1] = 3;
              ptCloudShovelReflectorsXyz_size[0] = pcTrans_size[0];
              if (0 <= i - 1) {
                memcpy(&SD->f13.ptCloudShovelReflectorsXyz_data[0],
                       &SD->f13.pcTrans_data[0], (unsigned int)(i * (int)sizeof
                        (double)));
              }

              /* 'SmartLoader:239' isFoundLoaderPc = true; */
              isFoundLoaderPc = true;
            }
          }
        }

        /* 'SmartLoader:244' if ~isFoundLoaderPc */
        /* 'SmartLoader:253' if ~isFoundLoaderPc */
        if (!isFoundLoaderPc) {
          /*  Third stradegy - determine the range for both clusters, the loader reflector suppose to be circle shaped */
          /*  and the shovel reflector supposes to be much more rectangular shaped */
          /* 'SmartLoader:256' pcFirstMinorToMajorRation = min(pcFirstRange(1:2)) / max(pcFirstRange(1:2)); */
          if ((pcFirstRange_idx_0 > pcFirstRange_idx_1) || (rtIsNaN
               (pcFirstRange_idx_0) && (!rtIsNaN(pcFirstRange_idx_1)))) {
            zMedian = pcFirstRange_idx_1;
          } else {
            zMedian = pcFirstRange_idx_0;
          }

          if ((pcFirstRange_idx_0 < pcFirstRange_idx_1) || (rtIsNaN
               (pcFirstRange_idx_0) && (!rtIsNaN(pcFirstRange_idx_1)))) {
            pcFirstRange_idx_0 = pcFirstRange_idx_1;
          }

          /* 'SmartLoader:257' pcSecondMinorToMajorRation = min(pcSecondRange(1:2)) / max(pcSecondRange(1:2)); */
          if ((pcSecondRange_idx_0 > pcSecondRange_idx_1) || (rtIsNaN
               (pcSecondRange_idx_0) && (!rtIsNaN(pcSecondRange_idx_1)))) {
            singleReflectorRangeLimitMeter = pcSecondRange_idx_1;
          } else {
            singleReflectorRangeLimitMeter = pcSecondRange_idx_0;
          }

          if ((pcSecondRange_idx_0 < pcSecondRange_idx_1) || (rtIsNaN
               (pcSecondRange_idx_0) && (!rtIsNaN(pcSecondRange_idx_1)))) {
            pcSecondRange_idx_0 = pcSecondRange_idx_1;
          }

          /*  TODO : add a limit to the differenct between pcFirstMinorToMajorRation to pcSecondMinorToMajorRation */
          /* 'SmartLoader:260' if pcSecondMinorToMajorRation < pcFirstMinorToMajorRation */
          if (singleReflectorRangeLimitMeter / pcSecondRange_idx_0 < zMedian /
              pcFirstRange_idx_0) {
            /* 'SmartLoader:261' ptCloudLoaderReflectorsXyz = pcFirstXyz; */
            trueCount = pcTrans_size[0];
            if (0 <= i - 1) {
              memcpy(&SD->f13.ptCloudSenceReflectorsXyz_data[0],
                     &SD->f13.pcTrans_data[0], (unsigned int)(i * (int)sizeof
                      (double)));
            }

            /* 'SmartLoader:261' ptCloudShovelReflectorsXyz = pcSecondXyz; */
            ptCloudShovelReflectorsXyz_size[1] = 3;
            ptCloudShovelReflectorsXyz_size[0] = ptCloudSenceXyz_size[0];
            if (0 <= j - 1) {
              memcpy(&SD->f13.ptCloudShovelReflectorsXyz_data[0],
                     &SD->f13.ptCloudSenceXyz_data[0], (unsigned int)(j * (int)
                      sizeof(double)));
            }
          } else {
            /* 'SmartLoader:262' else */
            /* 'SmartLoader:263' ptCloudLoaderReflectorsXyz = pcSecondXyz; */
            trueCount = ptCloudSenceXyz_size[0];
            if (0 <= j - 1) {
              memcpy(&SD->f13.ptCloudSenceReflectorsXyz_data[0],
                     &SD->f13.ptCloudSenceXyz_data[0], (unsigned int)(j * (int)
                      sizeof(double)));
            }

            /* 'SmartLoader:263' ptCloudShovelReflectorsXyz = pcFirstXyz; */
            ptCloudShovelReflectorsXyz_size[1] = 3;
            ptCloudShovelReflectorsXyz_size[0] = pcTrans_size[0];
            if (0 <= i - 1) {
              memcpy(&SD->f13.ptCloudShovelReflectorsXyz_data[0],
                     &SD->f13.pcTrans_data[0], (unsigned int)(i * (int)sizeof
                      (double)));
            }
          }
        }

        /*  figure, PlotPointCloud(ptCloudLoaderReflectorsXyz); */
        /*  figure, PlotPointCloud(ptCloudShovelReflectorsXyz); */
        /*     %% Estimate the shovel loc */
        /* 'SmartLoader:271' if size(ptCloudShovelReflectorsXyz,1) >= configParams.minPointsForReflector */
        if (ptCloudShovelReflectorsXyz_size[0] >=
            configParams->minPointsForReflector) {
          /* 'SmartLoader:272' smartLoaderStruct.shvelLoc = mean(ptCloudShovelReflectorsXyz)'; */
          mean(SD->f13.ptCloudShovelReflectorsXyz_data,
               ptCloudShovelReflectorsXyz_size, v);
          smartLoaderStruct->shvelLoc[0] = v[0];
          smartLoaderStruct->shvelLoc[1] = v[1];
          smartLoaderStruct->shvelLoc[2] = v[2];
        }

        /* %%%%%%%%%%%%%%%%%%%%%%% */
        /*  determine which cluster is the loader or the shovel - */
        /*  the loader reflector must have more points from the shovel reflector */
        /*  TODO - estimate the ellipse on top of the cluster - and calcualte the ration between the major and minor axis - this will help you determine better which one of the reflectos is the loader and */
        /*  the shovel */
        /*  You cannot determine which is which according to the number of points! */
        /*     %{ */
        /*     kmeansClusterNumPoints = [sum(kmeansIdx == 1); sum(kmeansIdx == 2)]; */
        /*     clusterNumPointsMinRatio = 1.25; */
        /*     if kmeansClusterNumPoints(1) > kmeansClusterNumPoints(2) * clusterNumPointsMinRatio */
        /*         ptCloudShovelReflectors = select(ptCloudSenceReflectors, kmeansIdx == 1); */
        /*         ptCloudLoaderReflectors = select(ptCloudSenceReflectors, kmeansIdx == 2); */
        /*     elseif kmeansClusterNumPoints(2) > kmeansClusterNumPoints(1) * clusterNumPointsMinRatio */
        /*         ptCloudLoaderReflectors = select(ptCloudSenceReflectors, kmeansIdx == 1); */
        /*         ptCloudShovelReflectors = select(ptCloudSenceReflectors, kmeansIdx == 2); */
        /*     else */
        /*         % unable to determine in a certainty which reflector is which !!! */
        /*         assert(false); */
        /*     end */
        /*     %} */
        /* %%%%%%%%%%%%%%%%%%%%%%% */
      } else {
        /*     %% */
        /*  We assume the reflectors hold only the loader! */
        /* 'SmartLoader:119' if coder.target('Matlab') */
        /*  TODO - handle the reflectors here !!! */
        /* ptCloudLoaderReflectors = ptCloudSenceReflectors; */
        /* 'SmartLoader:124' ptCloudLoaderReflectorsXyz = ptCloudSenceReflectorsXyz; */
        /* 'SmartLoader:125' ptCloudLoaderReflectorsIntensity = ptCloudSenceReflectorsIntensity; */
        /*  figure, PlotPointCloud(ptCloudSenceReflectors); */
        /*  figure, PlotPointCloud(ptCloudLoaderReflectors); */
      }

      /*  Remove outliers points in the loader point cloud. these points z's coordinate difference is larger than the following threshold */
      /*  Remove points which are outliers in the z axis */
      /*  figure, PlotPointCloud(ptCloudLoaderReflectors); */
      /* 'SmartLoader:301' zCor = ptCloudLoaderReflectorsXyz(:,3); */
      for (i1 = 0; i1 < trueCount; i1++) {
        SD->f13.tmp_data[i1] = SD->f13.ptCloudSenceReflectorsXyz_data[2 + 3 * i1];
      }

      /* 'SmartLoader:302' zMedian = median(zCor); */
      iv0[0] = trueCount;
      zMedian = median(SD, SD->f13.tmp_data, iv0);

      /* 'SmartLoader:303' assert(numel(zMedian) == 1); */
      /* 'SmartLoader:304' loaderReflectorPtrInd = abs(zCor - zMedian) <= configParams.loaderReflectorMaxZaxisDistanceForOutlierMeter; */
      tmp_size[0] = trueCount;
      for (i1 = 0; i1 < trueCount; i1++) {
        SD->f13.b_tmp_data[i1] = SD->f13.tmp_data[i1] - zMedian;
      }

      b_abs(SD->f13.b_tmp_data, tmp_size, SD->f13.kmeansIdx_data, kmeansIdx_size);
      loop_ub = kmeansIdx_size[0];
      for (i1 = 0; i1 < loop_ub; i1++) {
        SD->f13.temp30_data[i1] = (SD->f13.kmeansIdx_data[i1] <=
          configParams->loaderReflectorMaxZaxisDistanceForOutlierMeter);
      }

      /*  sum(loaderReflectorPtrInd) */
      /* ptCloudLoaderReflectorsFilterd = select(ptCloudLoaderReflectors, find(loaderReflectorPtrInd)); */
      /* 'SmartLoader:307' ptCloudLoaderReflectorsFilterdXyz = ptCloudLoaderReflectorsXyz(loaderReflectorPtrInd,:); */
      /*  figure, PlotPointCloud(ptCloudLoaderReflectorsXyz); */
      /*  figure, PlotPointCloud(ptCloudLoaderReflectorsFilterdXyz); */
      /*  Determine the number of lines, determine the cluster of lines. */
      /*  Determine Find the number of lines */
      /* 'SmartLoader:314' coder.varsize('ptr', [SmartLoaderCompilationConstants.MaxPointCloudSize 3], [1 0]); */
      /* 'SmartLoader:315' ptr = ptCloudLoaderReflectorsFilterdXyz; */
      n = kmeansIdx_size[0] - 1;
      trueCount = 0;
      for (i = 0; i <= n; i++) {
        if (SD->f13.temp30_data[i]) {
          trueCount++;
        }
      }

      m = 0;
      for (i = 0; i <= n; i++) {
        if (SD->f13.temp30_data[i]) {
          SD->f13.f_tmp_data[m] = i + 1;
          m++;
        }
      }

      /* 'SmartLoader:317' if coder.target('Matlab') && false */
      /* 'SmartLoader:329' else */
      /*  Cluster in 2D */
      /* 'SmartLoader:331' coder.varsize('clustersYs', 'clustersXs', [1 64], [0 1]); */
      /* 'SmartLoader:332' [clustersXs, clustersYs] = ClusterPoints2D(ptr(:,1:2), configParams.maxDistanceBetweenEachRayMeter); */
      for (i1 = 0; i1 < trueCount; i1++) {
        n = 3 * (SD->f13.f_tmp_data[i1] - 1);
        SD->f13.ptCloudSenceXyz_data[3 * i1] =
          SD->f13.ptCloudSenceReflectorsXyz_data[n];
        SD->f13.ptCloudSenceXyz_data[1 + 3 * i1] =
          SD->f13.ptCloudSenceReflectorsXyz_data[1 + n];
        SD->f13.ptCloudSenceXyz_data[2 + 3 * i1] =
          SD->f13.ptCloudSenceReflectorsXyz_data[2 + n];
      }

      c_ptCloudSenceReflectorsXyz_size[1] = 2;
      c_ptCloudSenceReflectorsXyz_size[0] = trueCount;
      for (i1 = 0; i1 < trueCount; i1++) {
        n = i1 << 1;
        SD->f13.kmeansDistanceMat_data[n] = SD->f13.ptCloudSenceXyz_data[3 * i1];
        SD->f13.kmeansDistanceMat_data[1 + n] = SD->f13.ptCloudSenceXyz_data[1 +
          3 * i1];
      }

      emxInit_cell_wrap_0_64x1(&clustersXs);
      emxInit_cell_wrap_0_64x1(&clustersYs);
      ClusterPoints2D(SD, SD->f13.kmeansDistanceMat_data,
                      c_ptCloudSenceReflectorsXyz_size,
                      configParams->maxDistanceBetweenEachRayMeter,
                      clustersXs.data, clustersXs.size, clustersYs.data,
                      clustersYs.size);

      /* 'SmartLoader:333' if isempty(clustersXs) */
      /*  remove small clusters with less than 3 points */
      /* 'SmartLoader:348' minimumNumPointsInCluster = 3; */
      /*  cellfun doens't works with matlab coder */
      /*  Previous code: clustersXs(cellfun('length',clustersXs)<minimumNumPointsInCluster) = []; */
      /*  Previous code: clustersYs(cellfun('length',clustersYs)<minimumNumPointsInCluster) = []; */
      /*  In order to solve this issue - I have coded the filter function by my own */
      /* 'SmartLoader:354' isInvalid = zeros(1,size(clustersXs,2),'logical'); */
      /* 'SmartLoader:355' for i = 1:size(clustersXs,2) */
      for (i = 0; i < 64; i++) {
        isInvalid_data[i] = false;

        /* 'SmartLoader:356' if size(clustersXs{i},1) < minimumNumPointsInCluster */
        if (clustersXs.data[i].f1->size[0] < 3) {
          /* 'SmartLoader:357' isInvalid(i) = true; */
          isInvalid_data[i] = true;
        }
      }

      emxInit_cell_wrap_0_64x1(&r0);

      /* 'SmartLoader:360' clustersXs(isInvalid) = []; */
      nullAssignment(clustersXs.data, isInvalid_data, r0.data, r0.size);

      /* 'SmartLoader:361' clustersYs(isInvalid) = []; */
      nullAssignment(clustersYs.data, isInvalid_data, clustersXs.data,
                     clustersXs.size);

      /* 'SmartLoader:364' if isempty(clustersXs) */
      emxFree_cell_wrap_0_64x1(&clustersYs);
      if ((r0.size[1] != 0) && (r0.size[1] != 1)) {
        /* 'SmartLoader:377' if numel(clustersXs) == 1 */
        /* 'SmartLoader:391' if coder.target('Matlab') && false */
        /*     %% Get the extreme points for each cluster - these points will be use for circle center estimation */
        /* 'SmartLoader:404' coder.varsize('extremePoints', [SmartLoaderCompilationConstants.MaxNumClusters 2], [1 0]); */
        /* 'SmartLoader:406' extremePoints = zeros(0,2,percisionMode); */
        extremePoints_size_idx_1 = 2;
        extremePoints_size_idx_0 = 0;

        /* 'SmartLoader:407' for q = 1:numel(clustersXs) */
        i1 = r0.size[1];
        emxInit_real_T(&pdistOutput, 2);
        emxInit_real_T(&Z, 2);
        emxInit_real_T(&extremePointsInds, 2);
        emxInit_real_T(&b, 1);
        emxInitMatrix_cell_wrap_0(reshapes);
        emxInitMatrix_cell_wrap_0(b_reshapes);
        emxInitMatrix_cell_wrap_0(c_reshapes);
        emxInit_real_T(&d_reshapes, 2);
        for (q = 0; q < i1; q++) {
          /* 'SmartLoader:408' pdistOutput = pdist([clustersXs{q} clustersYs{q}]); */
          if ((r0.data[q].f1->size[0] != 0) && (r0.data[q].f1->size[1] != 0)) {
            m = r0.data[q].f1->size[0];
          } else if ((clustersXs.data[q].f1->size[0] != 0) && (clustersXs.data[q]
                      .f1->size[1] != 0)) {
            m = clustersXs.data[q].f1->size[0];
          } else {
            m = r0.data[q].f1->size[0];
            if (m <= 0) {
              m = 0;
            }

            if (clustersXs.data[q].f1->size[0] > m) {
              m = clustersXs.data[q].f1->size[0];
            }
          }

          empty_non_axis_sizes = (m == 0);
          if (empty_non_axis_sizes || ((r0.data[q].f1->size[0] != 0) &&
               (r0.data[q].f1->size[1] != 0))) {
            input_sizes_idx_1 = (signed char)r0.data[q].f1->size[1];
          } else {
            input_sizes_idx_1 = 0;
          }

          j = input_sizes_idx_1;
          if ((input_sizes_idx_1 == r0.data[q].f1->size[1]) && (m == r0.data[q].
               f1->size[0])) {
            i2 = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
            reshapes[0].f1->size[1] = input_sizes_idx_1;
            reshapes[0].f1->size[0] = m;
            emxEnsureCapacity_real_T(reshapes[0].f1, i2);
            loop_ub = input_sizes_idx_1 * m;
            for (i2 = 0; i2 < loop_ub; i2++) {
              reshapes[0].f1->data[i2] = r0.data[q].f1->data[i2];
            }
          } else {
            i2 = 0;
            i3 = 0;
            trueCount = 0;
            loop_ub = 0;
            i = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
            reshapes[0].f1->size[1] = input_sizes_idx_1;
            reshapes[0].f1->size[0] = m;
            emxEnsureCapacity_real_T(reshapes[0].f1, i);
            for (i = 0; i < m * j; i++) {
              reshapes[0].f1->data[i3 + reshapes[0].f1->size[1] * i2] =
                r0.data[q].f1->data[loop_ub + r0.data[q].f1->size[1] * trueCount];
              i2++;
              trueCount++;
              if (i2 > reshapes[0].f1->size[0] - 1) {
                i2 = 0;
                i3++;
              }

              if (trueCount > r0.data[q].f1->size[0] - 1) {
                trueCount = 0;
                loop_ub++;
              }
            }
          }

          if (empty_non_axis_sizes || ((clustersXs.data[q].f1->size[0] != 0) &&
               (clustersXs.data[q].f1->size[1] != 0))) {
            input_sizes_idx_1 = (signed char)clustersXs.data[q].f1->size[1];
          } else {
            input_sizes_idx_1 = 0;
          }

          j = input_sizes_idx_1;
          if ((input_sizes_idx_1 == clustersXs.data[q].f1->size[1]) && (m ==
               clustersXs.data[q].f1->size[0])) {
            i2 = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
            reshapes[1].f1->size[1] = input_sizes_idx_1;
            reshapes[1].f1->size[0] = m;
            emxEnsureCapacity_real_T(reshapes[1].f1, i2);
            loop_ub = input_sizes_idx_1 * m;
            for (i2 = 0; i2 < loop_ub; i2++) {
              reshapes[1].f1->data[i2] = clustersXs.data[q].f1->data[i2];
            }
          } else {
            i2 = 0;
            i3 = 0;
            trueCount = 0;
            loop_ub = 0;
            i = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
            reshapes[1].f1->size[1] = input_sizes_idx_1;
            reshapes[1].f1->size[0] = m;
            emxEnsureCapacity_real_T(reshapes[1].f1, i);
            for (i = 0; i < m * j; i++) {
              reshapes[1].f1->data[i3 + reshapes[1].f1->size[1] * i2] =
                clustersXs.data[q].f1->data[loop_ub + clustersXs.data[q]
                .f1->size[1] * trueCount];
              i2++;
              trueCount++;
              if (i2 > reshapes[1].f1->size[0] - 1) {
                i2 = 0;
                i3++;
              }

              if (trueCount > clustersXs.data[q].f1->size[0] - 1) {
                trueCount = 0;
                loop_ub++;
              }
            }
          }

          i2 = d_reshapes->size[0] * d_reshapes->size[1];
          d_reshapes->size[1] = reshapes[0].f1->size[1] + reshapes[1].f1->size[1];
          d_reshapes->size[0] = reshapes[0].f1->size[0];
          emxEnsureCapacity_real_T(d_reshapes, i2);
          loop_ub = reshapes[0].f1->size[0];
          for (i2 = 0; i2 < loop_ub; i2++) {
            j = reshapes[0].f1->size[1];
            for (i3 = 0; i3 < j; i3++) {
              d_reshapes->data[i3 + d_reshapes->size[1] * i2] = reshapes[0]
                .f1->data[i3 + reshapes[0].f1->size[1] * i2];
            }
          }

          loop_ub = reshapes[1].f1->size[0];
          for (i2 = 0; i2 < loop_ub; i2++) {
            j = reshapes[1].f1->size[1];
            for (i3 = 0; i3 < j; i3++) {
              d_reshapes->data[(i3 + reshapes[0].f1->size[1]) + d_reshapes->
                size[1] * i2] = reshapes[1].f1->data[i3 + reshapes[1].f1->size[1]
                * i2];
            }
          }

          pdist(d_reshapes, pdistOutput);

          /* 'SmartLoader:409' Z = squareform(pdistOutput); */
          squareform(pdistOutput, Z);

          /*  Find the cooridnate of the most distanced points */
          /* 'SmartLoader:411' [~, maxInd] = max(Z(:)); */
          n = Z->size[0] * Z->size[1];
          i2 = 0;
          i3 = 0;
          trueCount = 0;
          loop_ub = b->size[0];
          b->size[0] = n;
          emxEnsureCapacity_real_T(b, loop_ub);
          for (loop_ub = 0; loop_ub < n; loop_ub++) {
            b->data[i2] = Z->data[trueCount + Z->size[1] * i3];
            i2++;
            i3++;
            if (i3 > Z->size[0] - 1) {
              i3 = 0;
              trueCount++;
            }
          }

          n = b->size[0];
          if (b->size[0] <= 2) {
            if (b->size[0] == 1) {
              m = 1;
            } else if ((b->data[0] < b->data[1]) || (rtIsNaN(b->data[0]) &&
                        (!rtIsNaN(b->data[1])))) {
              m = 2;
            } else {
              m = 1;
            }
          } else {
            if (!rtIsNaN(b->data[0])) {
              m = 1;
            } else {
              m = 0;
              j = 2;
              exitg1 = false;
              while ((!exitg1) && (j <= b->size[0])) {
                if (!rtIsNaN(b->data[j - 1])) {
                  m = j;
                  exitg1 = true;
                } else {
                  j++;
                }
              }
            }

            if (m == 0) {
              m = 1;
            } else {
              singleReflectorRangeLimitMeter = b->data[m - 1];
              i2 = m + 1;
              for (j = i2; j <= n; j++) {
                if (singleReflectorRangeLimitMeter < b->data[j - 1]) {
                  singleReflectorRangeLimitMeter = b->data[j - 1];
                  m = j;
                }
              }
            }
          }

          /* 'SmartLoader:411' ~ */
          /* 'SmartLoader:413' [row,col] = ind2sub(size(Z), maxInd); */
          distanceToKmeansClusterMeter[0] = Z->size[0];
          n = (int)(unsigned int)distanceToKmeansClusterMeter[0];
          vk = (m - 1) / n;
          m = (m - vk * n) - 1;

          /* 'SmartLoader:415' tempXs = [clustersXs{q}(row,:) clustersYs{q}(row,:)]; */
          loop_ub = r0.data[q].f1->size[1];
          j = clustersXs.data[q].f1->size[1];
          n = loop_ub + j;
          for (i2 = 0; i2 < loop_ub; i2++) {
            tempXs_data[i2] = r0.data[q].f1->data[i2 + r0.data[q].f1->size[1] *
              m];
          }

          for (i2 = 0; i2 < j; i2++) {
            tempXs_data[i2 + loop_ub] = clustersXs.data[q].f1->data[i2 +
              clustersXs.data[q].f1->size[1] * m];
          }

          /* 'SmartLoader:416' extremePoints = [extremePoints; tempXs]; */
          if (extremePoints_size_idx_0 != 0) {
            input_sizes_idx_1 = 2;
          } else if (n != 0) {
            input_sizes_idx_1 = (signed char)n;
          } else {
            input_sizes_idx_1 = 2;
          }

          if (extremePoints_size_idx_0 != 0) {
            input_sizes_idx_0 = (signed char)extremePoints_size_idx_0;
          } else {
            input_sizes_idx_0 = 0;
          }

          m = input_sizes_idx_0;
          j = input_sizes_idx_1;
          if ((input_sizes_idx_1 == 2) && (input_sizes_idx_0 ==
               extremePoints_size_idx_0)) {
            i2 = b_reshapes[0].f1->size[0] * b_reshapes[0].f1->size[1];
            b_reshapes[0].f1->size[1] = 2;
            b_reshapes[0].f1->size[0] = input_sizes_idx_0;
            emxEnsureCapacity_real_T(b_reshapes[0].f1, i2);
            loop_ub = 2 * input_sizes_idx_0;
            for (i2 = 0; i2 < loop_ub; i2++) {
              b_reshapes[0].f1->data[i2] = extremePoints_data[i2];
            }
          } else {
            i2 = 0;
            i3 = 0;
            trueCount = 0;
            loop_ub = 0;
            i = b_reshapes[0].f1->size[0] * b_reshapes[0].f1->size[1];
            b_reshapes[0].f1->size[1] = input_sizes_idx_1;
            b_reshapes[0].f1->size[0] = input_sizes_idx_0;
            emxEnsureCapacity_real_T(b_reshapes[0].f1, i);
            for (i = 0; i < m * j; i++) {
              b_reshapes[0].f1->data[i3 + b_reshapes[0].f1->size[1] * i2] =
                extremePoints_data[loop_ub + (trueCount << 1)];
              i2++;
              trueCount++;
              if (i2 > b_reshapes[0].f1->size[0] - 1) {
                i2 = 0;
                i3++;
              }

              if (trueCount > extremePoints_size_idx_0 - 1) {
                trueCount = 0;
                loop_ub++;
              }
            }
          }

          input_sizes_idx_0 = (signed char)(n != 0);
          m = input_sizes_idx_0;
          j = input_sizes_idx_1;
          if ((input_sizes_idx_1 == n) && (input_sizes_idx_0 == 1)) {
            i2 = b_reshapes[1].f1->size[0] * b_reshapes[1].f1->size[1];
            b_reshapes[1].f1->size[1] = input_sizes_idx_1;
            b_reshapes[1].f1->size[0] = 1;
            emxEnsureCapacity_real_T(b_reshapes[1].f1, i2);
            loop_ub = input_sizes_idx_1;
            for (i2 = 0; i2 < loop_ub; i2++) {
              b_reshapes[1].f1->data[i2] = tempXs_data[i2];
            }
          } else {
            i2 = 0;
            i3 = 0;
            trueCount = 0;
            loop_ub = b_reshapes[1].f1->size[0] * b_reshapes[1].f1->size[1];
            b_reshapes[1].f1->size[1] = input_sizes_idx_1;
            b_reshapes[1].f1->size[0] = input_sizes_idx_0;
            emxEnsureCapacity_real_T(b_reshapes[1].f1, loop_ub);
            for (loop_ub = 0; loop_ub < m * j; loop_ub++) {
              b_reshapes[1].f1->data[i3 + b_reshapes[1].f1->size[1] * i2] =
                tempXs_data[trueCount];
              i2++;
              if (i2 > b_reshapes[1].f1->size[0] - 1) {
                i2 = 0;
                i3++;
              }

              trueCount++;
            }
          }

          i2 = extremePointsInds->size[0] * extremePointsInds->size[1];
          extremePointsInds->size[1] = b_reshapes[0].f1->size[1];
          extremePointsInds->size[0] = b_reshapes[0].f1->size[0] + b_reshapes[1]
            .f1->size[0];
          emxEnsureCapacity_real_T(extremePointsInds, i2);
          loop_ub = b_reshapes[0].f1->size[0];
          for (i2 = 0; i2 < loop_ub; i2++) {
            j = b_reshapes[0].f1->size[1];
            for (i3 = 0; i3 < j; i3++) {
              extremePointsInds->data[i3 + extremePointsInds->size[1] * i2] =
                b_reshapes[0].f1->data[i3 + b_reshapes[0].f1->size[1] * i2];
            }
          }

          loop_ub = b_reshapes[1].f1->size[0];
          for (i2 = 0; i2 < loop_ub; i2++) {
            j = b_reshapes[1].f1->size[1];
            for (i3 = 0; i3 < j; i3++) {
              extremePointsInds->data[i3 + extremePointsInds->size[1] * (i2 +
                b_reshapes[0].f1->size[0])] = b_reshapes[1].f1->data[i3 +
                b_reshapes[1].f1->size[1] * i2];
            }
          }

          extremePoints_size_idx_1 = b_reshapes[0].f1->size[1];
          extremePoints_size_idx_0 = b_reshapes[0].f1->size[0] + b_reshapes[1].
            f1->size[0];
          loop_ub = b_reshapes[0].f1->size[0];
          for (i2 = 0; i2 < loop_ub; i2++) {
            j = b_reshapes[0].f1->size[1];
            for (i3 = 0; i3 < j; i3++) {
              b_extremePoints_data[i3 + extremePoints_size_idx_1 * i2] =
                b_reshapes[0].f1->data[i3 + b_reshapes[0].f1->size[1] * i2];
            }
          }

          loop_ub = b_reshapes[1].f1->size[0];
          for (i2 = 0; i2 < loop_ub; i2++) {
            j = b_reshapes[1].f1->size[1];
            for (i3 = 0; i3 < j; i3++) {
              b_extremePoints_data[i3 + extremePoints_size_idx_1 * (i2 +
                b_reshapes[0].f1->size[0])] = b_reshapes[1].f1->data[i3 +
                b_reshapes[1].f1->size[1] * i2];
            }
          }

          /* 'SmartLoader:418' tempYs = [clustersXs{q}(col,:) clustersYs{q}(col,:)]; */
          loop_ub = r0.data[q].f1->size[1];
          j = clustersXs.data[q].f1->size[1];
          n = loop_ub + j;
          for (i2 = 0; i2 < loop_ub; i2++) {
            tempXs_data[i2] = r0.data[q].f1->data[i2 + r0.data[q].f1->size[1] *
              vk];
          }

          for (i2 = 0; i2 < j; i2++) {
            tempXs_data[i2 + loop_ub] = clustersXs.data[q].f1->data[i2 +
              clustersXs.data[q].f1->size[1] * vk];
          }

          /* 'SmartLoader:419' extremePoints = [extremePoints; tempYs]; */
          if (extremePointsInds->size[0] != 0) {
            input_sizes_idx_1 = 2;
          } else if (n != 0) {
            input_sizes_idx_1 = (signed char)n;
          } else {
            input_sizes_idx_1 = 2;
          }

          if (extremePointsInds->size[0] != 0) {
            input_sizes_idx_0 = (signed char)extremePoints_size_idx_0;
          } else {
            input_sizes_idx_0 = 0;
          }

          m = input_sizes_idx_0;
          j = input_sizes_idx_1;
          if ((input_sizes_idx_1 == 2) && (input_sizes_idx_0 ==
               extremePoints_size_idx_0)) {
            i2 = c_reshapes[0].f1->size[0] * c_reshapes[0].f1->size[1];
            c_reshapes[0].f1->size[1] = 2;
            c_reshapes[0].f1->size[0] = input_sizes_idx_0;
            emxEnsureCapacity_real_T(c_reshapes[0].f1, i2);
            loop_ub = 2 * input_sizes_idx_0;
            for (i2 = 0; i2 < loop_ub; i2++) {
              c_reshapes[0].f1->data[i2] = b_extremePoints_data[i2];
            }
          } else {
            i2 = 0;
            i3 = 0;
            trueCount = 0;
            loop_ub = 0;
            i = c_reshapes[0].f1->size[0] * c_reshapes[0].f1->size[1];
            c_reshapes[0].f1->size[1] = input_sizes_idx_1;
            c_reshapes[0].f1->size[0] = input_sizes_idx_0;
            emxEnsureCapacity_real_T(c_reshapes[0].f1, i);
            for (i = 0; i < m * j; i++) {
              c_reshapes[0].f1->data[i3 + c_reshapes[0].f1->size[1] * i2] =
                b_extremePoints_data[loop_ub + extremePoints_size_idx_1 *
                trueCount];
              i2++;
              trueCount++;
              if (i2 > c_reshapes[0].f1->size[0] - 1) {
                i2 = 0;
                i3++;
              }

              if (trueCount > extremePoints_size_idx_0 - 1) {
                trueCount = 0;
                loop_ub++;
              }
            }
          }

          input_sizes_idx_0 = (signed char)(n != 0);
          m = input_sizes_idx_0;
          j = input_sizes_idx_1;
          if ((input_sizes_idx_1 == n) && (input_sizes_idx_0 == 1)) {
            i2 = c_reshapes[1].f1->size[0] * c_reshapes[1].f1->size[1];
            c_reshapes[1].f1->size[1] = input_sizes_idx_1;
            c_reshapes[1].f1->size[0] = 1;
            emxEnsureCapacity_real_T(c_reshapes[1].f1, i2);
            loop_ub = input_sizes_idx_1;
            for (i2 = 0; i2 < loop_ub; i2++) {
              c_reshapes[1].f1->data[i2] = tempXs_data[i2];
            }
          } else {
            i2 = 0;
            i3 = 0;
            trueCount = 0;
            loop_ub = c_reshapes[1].f1->size[0] * c_reshapes[1].f1->size[1];
            c_reshapes[1].f1->size[1] = input_sizes_idx_1;
            c_reshapes[1].f1->size[0] = input_sizes_idx_0;
            emxEnsureCapacity_real_T(c_reshapes[1].f1, loop_ub);
            for (loop_ub = 0; loop_ub < m * j; loop_ub++) {
              c_reshapes[1].f1->data[i3 + c_reshapes[1].f1->size[1] * i2] =
                tempXs_data[trueCount];
              i2++;
              if (i2 > c_reshapes[1].f1->size[0] - 1) {
                i2 = 0;
                i3++;
              }

              trueCount++;
            }
          }

          extremePoints_size_idx_1 = c_reshapes[0].f1->size[1];
          extremePoints_size_idx_0 = c_reshapes[0].f1->size[0] + c_reshapes[1].
            f1->size[0];
          loop_ub = c_reshapes[0].f1->size[0];
          for (i2 = 0; i2 < loop_ub; i2++) {
            j = c_reshapes[0].f1->size[1];
            for (i3 = 0; i3 < j; i3++) {
              extremePoints_data[i3 + (i2 << 1)] = c_reshapes[0].f1->data[i3 +
                c_reshapes[0].f1->size[1] * i2];
            }
          }

          loop_ub = c_reshapes[1].f1->size[0];
          for (i2 = 0; i2 < loop_ub; i2++) {
            j = c_reshapes[1].f1->size[1];
            for (i3 = 0; i3 < j; i3++) {
              extremePoints_data[i3 + ((i2 + c_reshapes[0].f1->size[0]) << 1)] =
                c_reshapes[1].f1->data[i3 + c_reshapes[1].f1->size[1] * i2];
            }
          }
        }

        emxFree_real_T(&d_reshapes);
        emxFreeMatrix_cell_wrap_0(c_reshapes);
        emxFreeMatrix_cell_wrap_0(b_reshapes);
        emxFreeMatrix_cell_wrap_0(reshapes);
        emxFree_real_T(&Z);
        emxFree_real_T(&pdistOutput);

        /* 'SmartLoader:422' extremePointsInds = nchoosek(1:size(extremePoints,1),2); */
        if (extremePoints_size_idx_0 < 1) {
          y_size[1] = 0;
          y_size[0] = 1;
        } else {
          y_size[1] = extremePoints_size_idx_0;
          y_size[0] = 1;
          for (i1 = 0; i1 < extremePoints_size_idx_0; i1++) {
            modelErr1_data[i1] = (signed char)(1 + (signed char)i1);
          }
        }

        emxInit_real_T(&r1, 2);
        nchoosek(modelErr1_data, y_size, extremePointsInds);

        /* 'SmartLoader:423' xs_i = extremePoints(extremePointsInds(:,1),1); */
        loop_ub = extremePointsInds->size[0];
        i1 = r1->size[0] * r1->size[1];
        r1->size[1] = 1;
        r1->size[0] = loop_ub;
        emxEnsureCapacity_real_T(r1, i1);
        for (i1 = 0; i1 < loop_ub; i1++) {
          r1->data[i1] = extremePoints_data[((int)extremePointsInds->
            data[extremePointsInds->size[1] * i1] - 1) << 1];
        }

        emxInit_real_T(&r2, 2);

        /* 'SmartLoader:424' xs_j = extremePoints(extremePointsInds(:,2),1); */
        loop_ub = extremePointsInds->size[0];
        i1 = r2->size[0] * r2->size[1];
        r2->size[1] = 1;
        r2->size[0] = loop_ub;
        emxEnsureCapacity_real_T(r2, i1);
        for (i1 = 0; i1 < loop_ub; i1++) {
          r2->data[i1] = extremePoints_data[((int)extremePointsInds->data[1 +
            extremePointsInds->size[1] * i1] - 1) << 1];
        }

        emxInit_real_T(&r3, 2);

        /* 'SmartLoader:426' ys_i = extremePoints(extremePointsInds(:,1),2); */
        loop_ub = extremePointsInds->size[0];
        i1 = r3->size[0] * r3->size[1];
        r3->size[1] = 1;
        r3->size[0] = loop_ub;
        emxEnsureCapacity_real_T(r3, i1);
        for (i1 = 0; i1 < loop_ub; i1++) {
          r3->data[i1] = extremePoints_data[1 + (((int)extremePointsInds->
            data[extremePointsInds->size[1] * i1] - 1) << 1)];
        }

        emxInit_real_T(&r4, 2);

        /* 'SmartLoader:427' ys_j = extremePoints(extremePointsInds(:,2),2); */
        loop_ub = extremePointsInds->size[0];
        i1 = r4->size[0] * r4->size[1];
        r4->size[1] = 1;
        r4->size[0] = loop_ub;
        emxEnsureCapacity_real_T(r4, i1);
        for (i1 = 0; i1 < loop_ub; i1++) {
          r4->data[i1] = extremePoints_data[1 + (((int)extremePointsInds->data[1
            + extremePointsInds->size[1] * i1] - 1) << 1)];
        }

        /* s1 = -2 * (xs_i - xs_j); */
        /* s2 = -2 * (ys_i - ys_j); */
        /* 'SmartLoader:431' A = [-2 * (xs_i - xs_j), -2 * (ys_i - ys_j)]; */
        loop_ub = extremePointsInds->size[0];
        i1 = b->size[0];
        b->size[0] = loop_ub;
        emxEnsureCapacity_real_T(b, i1);
        for (i1 = 0; i1 < loop_ub; i1++) {
          b->data[i1] = -2.0 * (r1->data[i1] - r2->data[i1]);
        }

        emxInit_real_T(&y, 1);
        loop_ub = extremePointsInds->size[0];
        i1 = y->size[0];
        y->size[0] = loop_ub;
        emxEnsureCapacity_real_T(y, i1);
        for (i1 = 0; i1 < loop_ub; i1++) {
          y->data[i1] = -2.0 * (r3->data[i1] - r4->data[i1]);
        }

        emxInit_real_T(&A, 2);
        n = b->size[0];
        m = y->size[0];
        i1 = A->size[0] * A->size[1];
        A->size[1] = 2;
        A->size[0] = n;
        emxEnsureCapacity_real_T(A, i1);
        for (i1 = 0; i1 < n; i1++) {
          A->data[i1 << 1] = b->data[i1];
        }

        for (i1 = 0; i1 < m; i1++) {
          A->data[1 + (i1 << 1)] = y->data[i1];
        }

        emxFree_real_T(&y);

        /* 'SmartLoader:432' b = - (xs_i .* xs_i - xs_j .* xs_j + ys_i .* ys_i - ys_j .* ys_j); */
        loop_ub = extremePointsInds->size[0];
        i1 = b->size[0];
        b->size[0] = loop_ub;
        emxEnsureCapacity_real_T(b, i1);
        emxFree_real_T(&extremePointsInds);
        for (i1 = 0; i1 < loop_ub; i1++) {
          b->data[i1] = -(((r1->data[i1] * r1->data[i1] - r2->data[i1] *
                            r2->data[i1]) + r3->data[i1] * r3->data[i1]) -
                          r4->data[i1] * r4->data[i1]);
        }

        emxFree_real_T(&r4);
        emxFree_real_T(&r3);
        emxFree_real_T(&r2);
        emxFree_real_T(&r1);
        emxInit_real_T(&Atranspose, 2);

        /* 'SmartLoader:434' Atranspose = A'; */
        i1 = Atranspose->size[0] * Atranspose->size[1];
        Atranspose->size[1] = A->size[0];
        Atranspose->size[0] = 2;
        emxEnsureCapacity_real_T(Atranspose, i1);
        loop_ub = A->size[0];
        for (i1 = 0; i1 < loop_ub; i1++) {
          Atranspose->data[i1] = A->data[i1 << 1];
        }

        loop_ub = A->size[0];
        for (i1 = 0; i1 < loop_ub; i1++) {
          Atranspose->data[i1 + Atranspose->size[1]] = A->data[1 + (i1 << 1)];
        }

        /* 'SmartLoader:435' cxyEst = inv(Atranspose*A) * Atranspose * b; */
        if ((Atranspose->size[1] == 1) || (A->size[0] == 1)) {
          b_y[0] = 0.0;
          loop_ub = A->size[0];
          for (i1 = 0; i1 < loop_ub; i1++) {
            b_y[0] += A->data[i1 << 1] * Atranspose->data[i1];
          }

          b_y[2] = 0.0;
          loop_ub = A->size[0];
          for (i1 = 0; i1 < loop_ub; i1++) {
            b_y[2] += A->data[i1 << 1] * Atranspose->data[i1 + Atranspose->size
              [1]];
          }

          b_y[1] = 0.0;
          loop_ub = A->size[0];
          for (i1 = 0; i1 < loop_ub; i1++) {
            b_y[1] += A->data[1 + (i1 << 1)] * Atranspose->data[i1];
          }

          b_y[3] = 0.0;
          loop_ub = A->size[0];
          for (i1 = 0; i1 < loop_ub; i1++) {
            b_y[3] += A->data[1 + (i1 << 1)] * Atranspose->data[i1 +
              Atranspose->size[1]];
          }
        } else {
          n = Atranspose->size[1];
          singleReflectorRangeLimitMeter = 0.0;
          for (j = 0; j < n; j++) {
            singleReflectorRangeLimitMeter += Atranspose->data[j] * A->data[j <<
              1];
          }

          b_y[0] = singleReflectorRangeLimitMeter;
          singleReflectorRangeLimitMeter = 0.0;
          for (j = 0; j < n; j++) {
            singleReflectorRangeLimitMeter += Atranspose->data[j] * A->data[1 +
              (j << 1)];
          }

          b_y[1] = singleReflectorRangeLimitMeter;
          singleReflectorRangeLimitMeter = 0.0;
          for (j = 0; j < n; j++) {
            singleReflectorRangeLimitMeter += Atranspose->data[j +
              Atranspose->size[1]] * A->data[j << 1];
          }

          b_y[2] = singleReflectorRangeLimitMeter;
          singleReflectorRangeLimitMeter = 0.0;
          for (j = 0; j < n; j++) {
            singleReflectorRangeLimitMeter += Atranspose->data[j +
              Atranspose->size[1]] * A->data[1 + (j << 1)];
          }

          b_y[3] = singleReflectorRangeLimitMeter;
        }

        emxFree_real_T(&A);
        emxInit_real_T(&c_y, 2);
        inv(b_y, a);
        n = Atranspose->size[1];
        unnamed_idx_1 = (unsigned int)Atranspose->size[1];
        i1 = c_y->size[0] * c_y->size[1];
        c_y->size[1] = (int)unnamed_idx_1;
        c_y->size[0] = 2;
        emxEnsureCapacity_real_T(c_y, i1);
        for (j = 0; j < n; j++) {
          singleReflectorRangeLimitMeter = a[0] * Atranspose->data[j];
          singleReflectorRangeLimitMeter += a[1] * Atranspose->data[j +
            Atranspose->size[1]];
          c_y->data[j] = singleReflectorRangeLimitMeter;
        }

        for (j = 0; j < n; j++) {
          singleReflectorRangeLimitMeter = a[2] * Atranspose->data[j];
          singleReflectorRangeLimitMeter += a[3] * Atranspose->data[j +
            Atranspose->size[1]];
          c_y->data[j + c_y->size[1]] = singleReflectorRangeLimitMeter;
        }

        emxFree_real_T(&Atranspose);
        if ((c_y->size[1] == 1) || (b->size[0] == 1)) {
          n = b->size[0];
          singleReflectorRangeLimitMeter = 0.0;
          for (i1 = 0; i1 < n; i1++) {
            singleReflectorRangeLimitMeter += b->data[i1] * c_y->data[i1];
          }

          distanceToKmeansClusterMeter[0] = singleReflectorRangeLimitMeter;
          singleReflectorRangeLimitMeter = 0.0;
          for (i1 = 0; i1 < n; i1++) {
            singleReflectorRangeLimitMeter += b->data[i1] * c_y->data[i1 +
              c_y->size[1]];
          }

          distanceToKmeansClusterMeter[1] = singleReflectorRangeLimitMeter;
        } else {
          n = c_y->size[1];
          singleReflectorRangeLimitMeter = 0.0;
          for (j = 0; j < n; j++) {
            singleReflectorRangeLimitMeter += c_y->data[j] * b->data[j];
          }

          distanceToKmeansClusterMeter[0] = singleReflectorRangeLimitMeter;
          singleReflectorRangeLimitMeter = 0.0;
          for (j = 0; j < n; j++) {
            singleReflectorRangeLimitMeter += c_y->data[j + c_y->size[1]] *
              b->data[j];
          }

          distanceToKmeansClusterMeter[1] = singleReflectorRangeLimitMeter;
        }

        emxFree_real_T(&c_y);
        emxFree_real_T(&b);

        /* 'SmartLoader:437' if coder.target('Matlab') &&  false */
        /*     %% Calculate the mean squre error for the estiamted model */
        /* 'SmartLoader:457' modelErr1 = extremePoints - repmat(cxyEst', size(extremePoints,1), 1); */
        b_repmat(distanceToKmeansClusterMeter, (double)extremePoints_size_idx_0,
                 b_modelErr1_data, pcTrans_size);
        pcTrans_size[0] = extremePoints_size_idx_0;
        loop_ub = extremePoints_size_idx_1 * extremePoints_size_idx_0 - 1;
        for (i1 = 0; i1 <= loop_ub; i1++) {
          b_modelErr1_data[i1] = extremePoints_data[i1] - b_modelErr1_data[i1];
        }

        /* 'SmartLoader:458' modelErr1 = modelErr1 .* modelErr1; */
        pcTrans_size[1] = 2;
        loop_ub = (extremePoints_size_idx_0 << 1) - 1;
        for (i1 = 0; i1 <= loop_ub; i1++) {
          b_modelErr1_data[i1] *= b_modelErr1_data[i1];
        }

        /* 'SmartLoader:459' modelErr1 = sqrt(sum(modelErr1, 2)) - ((configParams.loaderReflectorDiameterMeter/2)); */
        c_sum(b_modelErr1_data, pcTrans_size, modelErr1_data, kmeansIdx_size);
        b_sqrt(modelErr1_data, kmeansIdx_size);
        singleReflectorRangeLimitMeter =
          configParams->loaderReflectorDiameterMeter / 2.0;
        loop_ub = kmeansIdx_size[0];
        for (i1 = 0; i1 < loop_ub; i1++) {
          modelErr1_data[i1] -= singleReflectorRangeLimitMeter;
        }

        /* 'SmartLoader:460' mse = sum(modelErr1 .* modelErr1) / size(extremePoints,1); */
        /* 'SmartLoader:462' if mse > configParams.loaderReflectorDiameterMeter */
        modelErr1_size[0] = kmeansIdx_size[0];
        loop_ub = kmeansIdx_size[0];
        for (i1 = 0; i1 < loop_ub; i1++) {
          d_modelErr1_data[i1] = modelErr1_data[i1] * modelErr1_data[i1];
        }

        c_modelErr1_data.data = &d_modelErr1_data[0];
        c_modelErr1_data.size = &modelErr1_size[0];
        c_modelErr1_data.allocatedSize = 64;
        c_modelErr1_data.numDimensions = 1;
        c_modelErr1_data.canFreeData = false;
        if (!(b_sum(&c_modelErr1_data) / (double)extremePoints_size_idx_0 >
              configParams->loaderReflectorDiameterMeter)) {
          /* 'SmartLoader:476' smartLoaderStruct.loaderLoc = [cxyEst; zMedian]; */
          smartLoaderStruct->loaderLoc[0] = distanceToKmeansClusterMeter[0];
          smartLoaderStruct->loaderLoc[1] = distanceToKmeansClusterMeter[1];
          smartLoaderStruct->loaderLoc[2] = zMedian;

          /*  Ensure minimal distance from the loader location to the ground plane */
          /*  The loader height is around 27cm, we can ensure at least half of this size from the ground plane */
          /* 'SmartLoader:481' [loaderLocToPlaneDistance, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, smartLoaderStruct.loaderLoc'); */
          CalcPlaneToPointDistance(configParams->planeModelParameters,
            smartLoaderStruct->loaderLoc, &singleReflectorRangeLimitMeter,
            &empty_non_axis_sizes);

          /* 'SmartLoader:481' ~ */
          /* 'SmartLoader:482' if loaderLocToPlaneDistance < configParams.minimumDistanceFromLoaderToPlaneMeter */
          if (!(singleReflectorRangeLimitMeter <
                configParams->minimumDistanceFromLoaderToPlaneMeter)) {
            /*  Fit plane to the points */
            /* 'SmartLoader:497' maxDistanceCm = 0.05; */
            /* 'SmartLoader:498' if coder.target('Matlab') && isempty(configParams.planeModelParameters) */
            /* 'SmartLoader:510' else */
            /*  Calculate the inlier and outliers from the point cloud */
            /* 'SmartLoader:512' [shvalHeightToPlane, isPointAbovePlane] = CalcPlaneToPointDistance(configParams.planeModelParameters, ptCloudSenceXyz); */
            /*  remoev also all the points below the plane */
            /* 'SmartLoader:515' inlierIndicesLogical = shvalHeightToPlane < maxDistanceCm | ~isPointAbovePlane; */
            /* 'SmartLoader:516' inlierIndices = find(inlierIndicesLogical); */
            /* 'SmartLoader:517' outlierIndices = find(~inlierIndicesLogical); */
            /* 'SmartLoader:520' if coder.target('Matlab') && false */
            /*  Determine the bounding box of the loader */
            /* 'SmartLoader:530' if coder.target('Matlab') && false */
            /*  Plot the point cloud with an image at the side */
            /* 'SmartLoader:562' if coder.target('Matlab') && configParams.debugMode */
            /*  */
            /*  TODO: mark anything else as the ground */
            /*  figure, scatter(ptCloudSenceWithColor.Location(:,1), ptCloudSenceWithColor.Location(:,2)) */
            /*  Show as a video - the results along time */
            /*  Ddlaunay Triangulation of the point cloud */
            /* 'SmartLoader:682' if coder.target('Matlab') && false */
            /*  Save the location history */
            /* 'SmartLoader:701' SmartLoaderGlobal.loaderLocHistory = [SmartLoaderGlobal.loaderLocHistory; smartLoaderStruct.loaderLoc']; */
            m = SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] + 1;
            loop_ub = SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0];
            for (i1 = 0; i1 < loop_ub; i1++) {
              tmp_data[3 * i1] = SD->pd->
                SmartLoaderGlobal.loaderLocHistory.data[3 * i1];
              j = 1 + 3 * i1;
              tmp_data[j] = SD->pd->SmartLoaderGlobal.loaderLocHistory.data[j];
              j = 2 + 3 * i1;
              tmp_data[j] = SD->pd->SmartLoaderGlobal.loaderLocHistory.data[j];
            }

            tmp_data[3 * SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0]] =
              distanceToKmeansClusterMeter[0];
            tmp_data[1 + 3 * SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0]]
              = distanceToKmeansClusterMeter[1];
            tmp_data[2 + 3 * SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0]]
              = zMedian;
            SD->pd->SmartLoaderGlobal.loaderLocHistory.size[1] = 3;
            SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0]++;
            loop_ub = 3 * m;
            if (0 <= loop_ub - 1) {
              memcpy(&SD->pd->SmartLoaderGlobal.loaderLocHistory.data[0],
                     &tmp_data[0], (unsigned int)(loop_ub * (int)sizeof(double)));
            }

            /* 'SmartLoader:702' SmartLoaderGlobal.loaderTimeTatHistoryMs = [SmartLoaderGlobal.loaderTimeTatHistoryMs; configParams.timeTagMs]; */
            m = SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] + 1;
            if (0 <= SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] -
                1) {
              memcpy(&b_tmp_data[0], &SD->
                     pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[0],
                     (unsigned int)(SD->
                                    pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size
                                    [0] * (int)sizeof(unsigned long long)));
            }

            b_tmp_data[SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0]]
              = configParams->timeTagMs;
            SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0]++;
            if (0 <= m - 1) {
              memcpy(&SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[0],
                     &b_tmp_data[0], (unsigned int)(m * (int)sizeof(unsigned
                       long long)));
            }

            /* 'SmartLoader:703' if size(SmartLoaderGlobal.loaderTimeTatHistoryMs,1) >= SmartLoaderCompilationConstants.MaxHistorySize - 1 */
            if (SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] >= 31)
            {
              /* 'SmartLoader:704' shiftArrayBy = 10; */
              /* 'SmartLoader:706' SmartLoaderGlobal.loaderTimeTatHistoryMs = ... */
              /* 'SmartLoader:707'         SmartLoaderGlobal.loaderTimeTatHistoryMs((SmartLoaderCompilationConstants.MaxHistorySize - shiftArrayBy):end); */
              if (22 > SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0])
              {
                i1 = 1;
                i2 = 0;
              } else {
                i1 = 22;
                i2 = SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0];
              }

              n = (signed char)i2 - i1;
              m = n + 1;
              for (i2 = 0; i2 <= n; i2++) {
                c_tmp_data[i2] = (signed char)((signed char)(i1 + i2) - 1);
              }

              for (i1 = 0; i1 < m; i1++) {
                d_tmp_data[i1] = SD->
                  pd->
                  SmartLoaderGlobal.loaderTimeTatHistoryMs.data[c_tmp_data[i1]];
              }

              SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] = m;
              if (0 <= m - 1) {
                memcpy(&SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[0],
                       &d_tmp_data[0], (unsigned int)(m * (int)sizeof(unsigned
                         long long)));
              }

              /* 'SmartLoader:709' SmartLoaderGlobal.loaderLocHistory = ... */
              /* 'SmartLoader:710'         SmartLoaderGlobal.loaderLocHistory((SmartLoaderCompilationConstants.MaxHistorySize - shiftArrayBy):end,:); */
              if (22 > SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0]) {
                i1 = 0;
                i2 = 0;
              } else {
                i1 = 21;
                i2 = SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0];
              }

              n = i2 - i1;
              for (i2 = 0; i2 < n; i2++) {
                j = 3 * (i1 + i2);
                e_tmp_data[3 * i2] = SD->
                  pd->SmartLoaderGlobal.loaderLocHistory.data[j];
                e_tmp_data[1 + 3 * i2] = SD->
                  pd->SmartLoaderGlobal.loaderLocHistory.data[1 + j];
                e_tmp_data[2 + 3 * i2] = SD->
                  pd->SmartLoaderGlobal.loaderLocHistory.data[2 + j];
              }

              SD->pd->SmartLoaderGlobal.loaderLocHistory.size[1] = 3;
              SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] = n;
              loop_ub = 3 * n;
              if (0 <= loop_ub - 1) {
                memcpy(&SD->pd->SmartLoaderGlobal.loaderLocHistory.data[0],
                       &e_tmp_data[0], (unsigned int)(loop_ub * (int)sizeof
                        (double)));
              }
            }

            /* 'SmartLoader:713' smartLoaderStruct.status = true; */
            smartLoaderStruct->status = true;

            /* 'SmartLoader:715' if coder.target('Matlab') */
            /* 'SmartLoader:718' else */
            /* 'SmartLoader:719' debugPtCloudSenceXyz = zeros(0,0,'uint8'); */
            /* 'SmartLoader:720' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); */
          } else {
            /* 'SmartLoader:483' if coder.target('Matlab') */
            /* 'SmartLoader:486' if coder.target('Matlab') */
            /* 'SmartLoader:489' else */
            /* 'SmartLoader:490' debugPtCloudSenceXyz = zeros(0,0,'uint8'); */
            /* 'SmartLoader:491' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); */
          }
        } else {
          /* 'SmartLoader:463' if coder.target('Matlab') */
          /* 'SmartLoader:466' if coder.target('Matlab') */
          /* 'SmartLoader:469' else */
          /* 'SmartLoader:470' debugPtCloudSenceXyz = zeros(0,0,'uint8'); */
          /* 'SmartLoader:471' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); */
        }
      } else {
        /* 'SmartLoader:365' if coder.target('Matlab') */
        /* 'SmartLoader:368' if coder.target('Matlab') */
        /* 'SmartLoader:371' else */
        /* 'SmartLoader:372' debugPtCloudSenceXyz = zeros(0,0,'uint8'); */
        /* 'SmartLoader:373' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); */
        /* 'SmartLoader:378' if coder.target('Matlab') */
        /* 'SmartLoader:381' if coder.target('Matlab') */
        /* 'SmartLoader:384' else */
        /* 'SmartLoader:385' debugPtCloudSenceXyz = zeros(0,0,'uint8'); */
        /* 'SmartLoader:386' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); */
      }

      emxFree_cell_wrap_0_64x1(&r0);
      emxFree_cell_wrap_0_64x1(&clustersXs);
    } else {
      /* 'SmartLoader:95' if coder.target('Matlab') */
      /* 'SmartLoader:98' if coder.target('Matlab') */
      /* 'SmartLoader:101' else */
      /* 'SmartLoader:102' debugPtCloudSenceXyz = zeros(0,0,'uint8'); */
      /* 'SmartLoader:103' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); */
    }
  }
}

void SmartLoader_initialize(SmartLoaderStackData *SD)
{
  signed char y_loaderLocHistory_data[96];
  unsigned long long y_loaderTimeTatHistoryMs_data[32];
  int i0;
  rt_InitInfAndNaN(8U);
  omp_init_nest_lock(&emlrtNestLockGlobal);
  emxInitStruct_struct_T(&SD->pd->SmartLoaderGlobal);
  memset(&y_loaderLocHistory_data[0], 0, 96U * sizeof(signed char));
  memset(&y_loaderTimeTatHistoryMs_data[0], 0, sizeof(unsigned long long) << 5);
  SD->pd->SmartLoaderGlobal.isInitialized = false;
  SD->pd->SmartLoaderGlobal.loaderLocHistory.size[1] = 3;
  SD->pd->SmartLoaderGlobal.loaderLocHistory.size[0] = 32;
  for (i0 = 0; i0 < 96; i0++) {
    SD->pd->SmartLoaderGlobal.loaderLocHistory.data[i0] =
      y_loaderLocHistory_data[i0];
  }

  SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] = 32;
  memcpy(&SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[0],
         &y_loaderTimeTatHistoryMs_data[0], (unsigned int)(32 * (int)sizeof
          (unsigned long long)));
  eml_rand_mt19937ar_stateful_init(SD);
}

void SmartLoader_terminate()
{
  omp_destroy_nest_lock(&emlrtNestLockGlobal);
}

/* End of code generation (SmartLoader.cpp) */
