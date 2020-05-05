//
// File: PerceptionSmartLoader.cpp
//
// MATLAB Coder version            : 4.1
// C/C++ source code generated on  : 05-May-2020 11:47:28
//

// Include Files
#include <cmath>
#include <string.h>
#include <math.h>
#include <float.h>
#include "PerceptionSmartLoader.h"
#include "PerceptionSmartLoader_emxutil.h"

// Variable Definitions
omp_nest_lock_t emlrtNestLockGlobal;
static PerceptionSmartLoaderTLS *PerceptionSmartLoaderTLSGlobal;

#pragma omp threadprivate (PerceptionSmartLoaderTLSGlobal)

// Function Declarations
static void CalcPlaneToPointDistance(const double planeModelParameters[4], const
  float srcPoints[3], float *distanceFromPointToPlane, boolean_T
  *isPointAbovePlane);
static void ClusterPoints2D(PerceptionSmartLoaderStackData *SD, float ptr_data[],
  int ptr_size[2], double maxDistance, cell_wrap_4 clustersXs_data[], int
  clustersXs_size[2], cell_wrap_4 clustersYs_data[], int clustersYs_size[2],
  boolean_T *status);
static void FilterPointCloudAccordingToZdifferences
  (PerceptionSmartLoaderStackData *SD, const float pc_data[], const int pc_size
   [2], double diffThreshold, float pcFiltered_data[], int pcFiltered_size[2]);
static void SmartLoaderAlignPointCloud(PerceptionSmartLoaderStackData *SD,
  PerceptionSmartLoaderStruct *smartLoaderStruct, const double
  configParams_pcAlignmentProjMat[12], const double configParams_xyzLimits[6],
  double configParams_minNumPointsInPc, const double xyz_data[], const int
  xyz_size[2], const double intensity_data[], float ptCloudSenceXyz_data[], int
  ptCloudSenceXyz_size[2], float ptCloudSenceIntensity_data[], int
  ptCloudSenceIntensity_size[1]);
static void SmartLoaderCalcEigen(PerceptionSmartLoaderStackData *SD, const float
  xy_data[], const int xy_size[2], creal_T V[4], creal_T D[4]);
static void SmartLoaderCreateHeightMap(PerceptionSmartLoaderStackData *SD,
  PerceptionSmartLoaderStruct *smartLoaderStruct, const double
  configParams_xyzLimits[6], double configParams_heightMapResolutionMeterToPixel,
  const float xyz_data[], const int xyz_size[2], float heightMap_res_data[], int
  heightMap_res_size[2]);
static void SmartLoaderCreateHeightMap_free(PerceptionSmartLoaderStackData *SD);
static void SmartLoaderCreateHeightMap_init(PerceptionSmartLoaderStackData *SD);
static void SmartLoaderEstiamteLocations(PerceptionSmartLoaderStackData *SD,
  PerceptionSmartLoaderStruct *smartLoaderStruct, const
  PerceptionSmartLoaderConfigParam *configParams, const float
  ptCloudSenceXyz_data[], const int ptCloudSenceXyz_size[2], const float
  ptCloudSenceIntensity_data[], const int ptCloudSenceIntensity_size[1]);
static void SmartLoaderGlobalInit(PerceptionSmartLoaderStackData *SD);
static void SmartLoaderSmoothAngles(PerceptionSmartLoaderStackData *SD,
  PerceptionSmartLoaderStruct *smartLoaderStruct, double
  configParams_loaderYawAngleSmoothWeight, double
  configParams_loaderToShovelYawAngleSmoothWeight);
static boolean_T all(const boolean_T x[2]);
static boolean_T any(const boolean_T x[2]);
static void apply_row_permutation(PerceptionSmartLoaderStackData *SD, float
  y_data[], int y_size[2], const int idx_data[]);
static void b_CalcPlaneToPointDistance(const double planeModelParameters[4],
  const double srcPoints[3], double *distanceFromPointToPlane, boolean_T
  *isPointAbovePlane);
static void b_abs(const float x_data[], const int x_size[1], float y_data[], int
                  y_size[1]);
static void b_apply_row_permutation(PerceptionSmartLoaderStackData *SD, float
  y_data[], int y_size[2], const int idx_data[]);
static int b_bsearch(const float x_data[], const int x_size[1], double xi);
static void b_bsxfun(const float a_data[], const int a_size[2], const double
                     b_data[], const int b_size[2], float c_data[], int c_size[2]);
static void b_ceil(double x[2]);
static void b_cosd(double *x);
static void b_distfun(float D_data[], const float X_data[], const int X_size[2],
                      const float C[6], const int crows[2], int ncrows);
static void b_gcentroids(float C[6], int counts[2], const float X_data[], const
  int X_size[2], const int idx_data[], int clusters);
static void b_inv(const float x[4], float y[4]);
static double b_log2(double x);
static void b_mean(const float x_data[], const int x_size[2], float y[2]);
static void b_mergesort(PerceptionSmartLoaderStackData *SD, int idx_data[],
  const float x_data[], const int x_size[2], const int dir_data[], const int
  dir_size[2], int n);
static void b_nestedIter(const float x_data[], const int x_size[2], float
  y_data[], int y_size[1]);
static float b_norm(const float x[3]);
static void b_nullAssignment(double x_data[], int x_size[1]);
static double b_rand(PerceptionSmartLoaderStackData *SD);
static void b_repmat(const double a[3], double varargin_1, double b_data[], int
                     b_size[2]);
static void b_sind(double *x);
static void b_sort(creal_T x[2], int idx[2]);
static void b_sortIdx(PerceptionSmartLoaderStackData *SD, const float x_data[],
                      const int x_size[2], int idx_data[], int idx_size[1]);
static boolean_T b_sortLE(const float v_data[], int idx1, int idx2);
static void b_sqrt(float x_data[], int x_size[1]);
static void b_sum(const float x_data[], const int x_size[2], float y_data[], int
                  y_size[1]);
static void b_vecnorm(const float x_data[], const int x_size[2], float y_data[],
                      int y_size[1]);
static boolean_T ball_within_bounds(const float queryPt[2], const double
  lowBounds_data[], const double upBounds_data[], float poweredRadius);
static void batchUpdate(PerceptionSmartLoaderStackData *SD, const float X_data[],
  const int X_size[2], int idx_data[], int idx_size[1], float C[6], float
  D_data[], int D_size[2], int counts[2], boolean_T *converged, int *iter);
static boolean_T bounds_overlap_ball(const float queryPt[2], const double
  lowBounds_data[], const double upBounds_data[], float radius);
static void bsxfun(const float a_data[], const int a_size[2], const float b[2],
                   float c_data[], int c_size[2]);
static void c_CalcPlaneToPointDistance(PerceptionSmartLoaderStackData *SD, const
  double planeModelParameters[4], const float srcPoints_data[], const int
  srcPoints_size[2], float distanceFromPointToPlane_data[], int
  distanceFromPointToPlane_size[1], boolean_T isPointAbovePlane_data[], int
  isPointAbovePlane_size[1]);
static void c_mergesort(PerceptionSmartLoaderStackData *SD, int idx_data[],
  const float x_data[], int n);
static double c_norm(const double x[3]);
static void c_nullAssignment(PerceptionSmartLoaderStackData *SD, float x_data[],
  int x_size[2], const int idx_data[]);
static void c_sortIdx(float x_data[], int x_size[1], int idx_data[], int
                      idx_size[1]);
static boolean_T c_sortLE(const creal_T v[2]);
static float c_sum(const float x_data[], const int x_size[1]);
static void contrib(double x1, double b_y1, double x2, double y2, signed char
                    quad1, signed char quad2, double scale, signed char
                    *diffQuad, boolean_T *onj);
static int countEmpty(int empties[2], const int counts[2], const int changed[2],
                      int nchanged);
static double d_norm(const double x[2]);
static void d_sum(const float x_data[], const int x_size[2], float y_data[], int
                  y_size[1]);
static void diag(const creal_T v[4], creal_T d[2]);
static void distfun(float D_data[], const float X_data[], const int X_size[2],
                    const float C[6], int crows);
static int div_s32(int numerator, int denominator);
static creal_T dot(const creal_T a[2], const double b[2]);
static double e_sum(const boolean_T x_data[], const int x_size[1]);
static void eig(const double A[4], creal_T V[4], creal_T D[4]);
static void eml_rand_mt19937ar_stateful_init(PerceptionSmartLoaderStackData *SD);
static void emlrtFreeThreadStackData();
static PerceptionSmartLoaderTLS *emlrtGetThreadStackData();
static void emlrtInitThreadStackData();
static int findchanged(PerceptionSmartLoaderStackData *SD, int changed[2], const
  int idx_data[], const int previdx_data[], const int moved_data[], const int
  moved_size[1], int nmoved);
static void gcentroids(float C[6], int counts[2], const float X_data[], const
  int X_size[2], const int idx_data[], const int clusters[2], int nclusters);
static void getNodeFromArray(const unsigned int idxAll_data[], const double
  idxDim_data[], double this_node, unsigned int node_idx_this_data[], int
  node_idx_this_size[1]);
static double get_starting_node(const float queryPt[2], const double
  cutDim_data[], const double cutVal_data[], const boolean_T leafNode_data[],
  const double leftChild_data[], const double rightChild_data[]);
static boolean_T ifWhileCond(const boolean_T x_data[]);
static void inpolygon(const float x_data[], const int x_size[1], const float
                      y_data[], const double xv[4], const double yv[4],
                      boolean_T in_data[], int in_size[1]);
static void inv(const double x[16], double y[16]);
static void kdsearchfun(PerceptionSmartLoaderStackData *SD, const struct_T *obj,
  const emxArray_real_T *Y, emxArray_real_T *idx, emxArray_real32_T *dist);
static void kmeans(PerceptionSmartLoaderStackData *SD, const float X_data[],
                   const int X_size[2], double idxbest_data[], int idxbest_size
                   [1], float Cbest[6], float varargout_1[2], float
                   varargout_2_data[], int varargout_2_size[2]);
static void knnsearch(PerceptionSmartLoaderStackData *SD, const float X_data[],
                      const int X_size[2], const emxArray_real_T *Y,
                      emxArray_real_T *Idx);
static void local_kmeans(PerceptionSmartLoaderStackData *SD, const float X_data[],
  const int X_size[2], int idxbest_data[], int idxbest_size[1], float Cbest[6],
  float varargout_1[2], float varargout_2_data[], int varargout_2_size[2]);
static void loopBody(PerceptionSmartLoaderStackData *SD, const float X_data[],
                     const int X_size[2], float *totsumD, int idx_data[], int
                     idx_size[1], float C[6], float sumD[2], float D_data[], int
                     D_size[2]);
static void mean(const float x_data[], const int x_size[2], float y[3]);
static void med3(float v_data[], int nv, int ia, int ib);
static float median(PerceptionSmartLoaderStackData *SD, const float x_data[],
                    const int x_size[1]);
static void medmed(float v_data[], int nv, int ia);
static void merge(int idx_data[], float x_data[], int offset, int np, int nq,
                  int iwork_data[], float xwork_data[]);
static void mergeSort(const float D1_data[], const int D1_size[1], const float
                      D2_data[], const int D2_size[1], const unsigned int
                      I1_data[], const unsigned int I2_data[], int N, float
                      dOut_data[], int dOut_size[1], unsigned int iOut_data[],
                      int iOut_size[1]);
static void merge_block(int idx_data[], float x_data[], int offset, int n, int
  preSortLevel, int iwork_data[], float xwork_data[]);
static void merge_pow2_block(int idx_data[], float x_data[], int offset);
static void mindim2(const float D_data[], const int D_size[2], float d_data[],
                    int d_size[1], int idx_data[], int idx_size[1]);
static double mpower(double b);
static void nestedIter(const float x_data[], const int x_size[2], float y_data[],
  int y_size[1]);
static int nonSingletonDim(const int x_size[1]);
static void nullAssignment(const cell_wrap_4 x_data[], const boolean_T idx_data[],
  cell_wrap_4 b_x_data[], int x_size[2]);
static void pdist(const emxArray_real32_T *Xin, emxArray_real32_T *Y);
static void pdist2(PerceptionSmartLoaderStackData *SD, const emxArray_real32_T
                   *Xin, const float Yin_data[], const int Yin_size[2],
                   emxArray_real32_T *D);
static int pivot(float v_data[], int *ip, int ia, int ib);
static void quickselect(float v_data[], int n, int vlen, float *vn, int *nfirst,
  int *nlast);
static void repmat(int varargin_1, float b_data[], int b_size[1]);
static boolean_T rows_differ(const float b_data[], int k0, int k);
static double rt_hypotd(double u0, double u1);
static double rt_remd(double u0, double u1);
static double rt_roundd(double u);
static float rt_roundf(float u);
static void search_kdtree(const double obj_cutDim_data[], const double
  obj_cutVal_data[], const double obj_lowerBounds_data[], const int
  obj_lowerBounds_size[2], const double obj_upperBounds_data[], const int
  obj_upperBounds_size[2], const double obj_leftChild_data[], const double
  obj_rightChild_data[], const boolean_T obj_leafNode_data[], const unsigned int
  obj_idxAll_data[], const double obj_idxDim_data[], const float X_data[], const
  float queryPt[2], int numNN, float pq_D_data[], int pq_D_size[1], unsigned int
  pq_I_data[], int pq_I_size[1]);
static void search_node(const float X_data[], const float queryPt[2], const
  unsigned int node_idx_start_data[], const int node_idx_start_size[1], int
  numNN, d_struct_T *pq);
static void simpleRandperm(PerceptionSmartLoaderStackData *SD, int n, int
  idx_data[], int idx_size[1]);
static void sort(float x_data[], int x_size[1], int idx_data[], int idx_size[1]);
static void sortIdx(PerceptionSmartLoaderStackData *SD, const float x_data[],
                    const int x_size[2], const int col_data[], const int
                    col_size[2], int idx_data[], int idx_size[1]);
static boolean_T sortLE(const float v_data[], const int v_size[2], const int
  dir_data[], const int dir_size[2], int idx1, int idx2);
static void sortrows(PerceptionSmartLoaderStackData *SD, float y_data[], int
                     y_size[2], double ndx_data[], int ndx_size[1]);
static void squareform(const emxArray_real32_T *Y, emxArray_real32_T *Z);
static double sum(const double x_data[], const int x_size[1]);
static int thirdOfFive(const float v_data[], int ia, int ib);
static void vecnorm(const float x[6], float y[2]);
static float vmedian(PerceptionSmartLoaderStackData *SD, float v_data[], int
                     v_size[1], int n);
static void xdlanv2(double *a, double *b, double *c, double *d, double *rt1r,
                    double *rt1i, double *rt2r, double *rt2i, double *cs, double
                    *sn);
static double xgehrd(double a[4]);
static void xzggbal(creal_T A[4], int *ilo, int *ihi, int rscale[2]);
static void xzggev(creal_T A[4], int *info, creal_T alpha1[2], creal_T beta1[2],
                   creal_T V[4]);
static void xzhgeqz(creal_T A[4], int ilo, int ihi, creal_T Z[4], int *info,
                    creal_T alpha1[2], creal_T beta1[2]);
static void xzlartg(const creal_T f, const creal_T g, double *cs, creal_T *sn);
static void xztgevc(const creal_T A[4], creal_T V[4]);

// Function Definitions

//
// function [distanceFromPointToPlane, isPointAbovePlane] = CalcPlaneToPointDistance(planeModelParameters, srcPoints)
// The function calculate the plane to point or a vector of points distance
//  Input arguments:
//  model - plane model - type of XXX
//  srcPoints - matrix of Nx3 of 3d points
//  Output arguments:
//  distanceFromPointToPlane - distance for each point, size of Nx1
//  isPointAbovePlane - boolean - represet is the current point is above or below the plane - above or below is related to the normal vector
//  of the plane
// Arguments    : const double planeModelParameters[4]
//                const float srcPoints[3]
//                float *distanceFromPointToPlane
//                boolean_T *isPointAbovePlane
// Return Type  : void
//
static void CalcPlaneToPointDistance(const double planeModelParameters[4], const
  float srcPoints[3], float *distanceFromPointToPlane, boolean_T
  *isPointAbovePlane)
{
  float temp1;

  // assert(isa(planeModelParameters, 'double'));
  // 'CalcPlaneToPointDistance:14' assert(size(planeModelParameters,1) == 1);
  // 'CalcPlaneToPointDistance:15' assert(size(planeModelParameters,2) == 4);
  // 'CalcPlaneToPointDistance:18' modelParametersRepmat = repmat(planeModelParameters(1:3), size(srcPoints,1), 1); 
  // 'CalcPlaneToPointDistance:20' temp1 = sum(bsxfun(@times, srcPoints, modelParametersRepmat), 2) + planeModelParameters(4); 
  temp1 = ((srcPoints[0] * (float)planeModelParameters[0] + srcPoints[1] *
            (float)planeModelParameters[1]) + srcPoints[2] * (float)
           planeModelParameters[2]) + (float)planeModelParameters[3];

  // 'CalcPlaneToPointDistance:22' distanceFromPointToPlane = abs(temp1) / norm(planeModelParameters(1:3)); 
  *distanceFromPointToPlane = std::abs(temp1) / (float)c_norm(*(double (*)[3])&
    planeModelParameters[0]);

  // 'CalcPlaneToPointDistance:24' isPointAbovePlane = temp1 > 0;
  *isPointAbovePlane = (temp1 > 0.0F);
}

//
// function [clustersXs, clustersYs, status] = ClusterPoints2D(ptr, maxDistance)
// Cell {end+1} command only works with vectors, not matrices, that's why we changed the output to two different cells
//  Must set upper bound for cell array compilation
// Arguments    : PerceptionSmartLoaderStackData *SD
//                float ptr_data[]
//                int ptr_size[2]
//                double maxDistance
//                cell_wrap_4 clustersXs_data[]
//                int clustersXs_size[2]
//                cell_wrap_4 clustersYs_data[]
//                int clustersYs_size[2]
//                boolean_T *status
// Return Type  : void
//
static void ClusterPoints2D(PerceptionSmartLoaderStackData *SD, float ptr_data[],
  int ptr_size[2], double maxDistance, cell_wrap_4 clustersXs_data[], int
  clustersXs_size[2], cell_wrap_4 clustersYs_data[], int clustersYs_size[2],
  boolean_T *status)
{
  int i11;
  int input_sizes_idx_0;
  int loop_ub;
  int cellArrayIndex;
  emxArray_real32_T *distanceMat;
  cell_wrap_4 reshapes[2];
  emxArray_real32_T *ex;
  emxArray_int32_T *idx;
  cell_wrap_4 b_reshapes[2];
  cell_wrap_4 c_reshapes[2];
  emxArray_real32_T *d_reshapes;
  int exitg1;
  int m;
  boolean_T empty_non_axis_sizes;
  signed char input_sizes_idx_1;
  int i12;
  int n;
  int i13;
  int i14;
  unsigned int distanceMat_idx_0;
  float minVal_data[1];
  int minInd_data[1];
  float b_ex;
  boolean_T b_minVal_data[1];
  int tmp_size[2];
  int b_minInd_data[1];

  // coder.varsize('clustersYs', 'clustersXs', [1 M], [0 1]);
  // 'ClusterPoints2D:10' clustersXs = cell(1,SmartLoaderCompilationConstants.MaxNumClusters); 
  // 'ClusterPoints2D:11' clustersXs = coder.nullcopy(clustersXs);
  i11 = clustersXs_size[0] * clustersXs_size[1];
  clustersXs_size[1] = 64;
  clustersXs_size[0] = 1;
  emxEnsureCapacity_cell_wrap_4(clustersXs_data, clustersXs_size, i11);

  // 'ClusterPoints2D:12' for i = 1:SmartLoaderCompilationConstants.MaxNumClusters 
  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[1].f1->size[1] = 0;
  clustersXs_data[1].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[2].f1->size[1] = 0;
  clustersXs_data[2].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[3].f1->size[1] = 0;
  clustersXs_data[3].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[4].f1->size[1] = 0;
  clustersXs_data[4].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[5].f1->size[1] = 0;
  clustersXs_data[5].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[6].f1->size[1] = 0;
  clustersXs_data[6].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[7].f1->size[1] = 0;
  clustersXs_data[7].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[8].f1->size[1] = 0;
  clustersXs_data[8].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[9].f1->size[1] = 0;
  clustersXs_data[9].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[10].f1->size[1] = 0;
  clustersXs_data[10].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[11].f1->size[1] = 0;
  clustersXs_data[11].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[12].f1->size[1] = 0;
  clustersXs_data[12].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[13].f1->size[1] = 0;
  clustersXs_data[13].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[14].f1->size[1] = 0;
  clustersXs_data[14].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[15].f1->size[1] = 0;
  clustersXs_data[15].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[16].f1->size[1] = 0;
  clustersXs_data[16].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[17].f1->size[1] = 0;
  clustersXs_data[17].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[18].f1->size[1] = 0;
  clustersXs_data[18].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[19].f1->size[1] = 0;
  clustersXs_data[19].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[20].f1->size[1] = 0;
  clustersXs_data[20].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[21].f1->size[1] = 0;
  clustersXs_data[21].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[22].f1->size[1] = 0;
  clustersXs_data[22].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[23].f1->size[1] = 0;
  clustersXs_data[23].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[24].f1->size[1] = 0;
  clustersXs_data[24].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[25].f1->size[1] = 0;
  clustersXs_data[25].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[26].f1->size[1] = 0;
  clustersXs_data[26].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[27].f1->size[1] = 0;
  clustersXs_data[27].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[28].f1->size[1] = 0;
  clustersXs_data[28].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[29].f1->size[1] = 0;
  clustersXs_data[29].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[30].f1->size[1] = 0;
  clustersXs_data[30].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[31].f1->size[1] = 0;
  clustersXs_data[31].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[32].f1->size[1] = 0;
  clustersXs_data[32].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[33].f1->size[1] = 0;
  clustersXs_data[33].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[34].f1->size[1] = 0;
  clustersXs_data[34].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[35].f1->size[1] = 0;
  clustersXs_data[35].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[36].f1->size[1] = 0;
  clustersXs_data[36].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[37].f1->size[1] = 0;
  clustersXs_data[37].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[38].f1->size[1] = 0;
  clustersXs_data[38].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[39].f1->size[1] = 0;
  clustersXs_data[39].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[40].f1->size[1] = 0;
  clustersXs_data[40].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[41].f1->size[1] = 0;
  clustersXs_data[41].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[42].f1->size[1] = 0;
  clustersXs_data[42].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[43].f1->size[1] = 0;
  clustersXs_data[43].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[44].f1->size[1] = 0;
  clustersXs_data[44].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[45].f1->size[1] = 0;
  clustersXs_data[45].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[46].f1->size[1] = 0;
  clustersXs_data[46].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[47].f1->size[1] = 0;
  clustersXs_data[47].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[48].f1->size[1] = 0;
  clustersXs_data[48].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[49].f1->size[1] = 0;
  clustersXs_data[49].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[50].f1->size[1] = 0;
  clustersXs_data[50].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[51].f1->size[1] = 0;
  clustersXs_data[51].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[52].f1->size[1] = 0;
  clustersXs_data[52].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[53].f1->size[1] = 0;
  clustersXs_data[53].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[54].f1->size[1] = 0;
  clustersXs_data[54].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[55].f1->size[1] = 0;
  clustersXs_data[55].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[56].f1->size[1] = 0;
  clustersXs_data[56].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[57].f1->size[1] = 0;
  clustersXs_data[57].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[58].f1->size[1] = 0;
  clustersXs_data[58].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[59].f1->size[1] = 0;
  clustersXs_data[59].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[60].f1->size[1] = 0;
  clustersXs_data[60].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[61].f1->size[1] = 0;
  clustersXs_data[61].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[62].f1->size[1] = 0;
  clustersXs_data[62].f1->size[0] = 0;

  // 'ClusterPoints2D:13' clustersXs{i} = zeros(0,0,'like', ptr);
  clustersXs_data[63].f1->size[1] = 0;
  clustersXs_data[63].f1->size[0] = 0;

  // 'ClusterPoints2D:15' clustersYs = cell(1,SmartLoaderCompilationConstants.MaxNumClusters); 
  // 'ClusterPoints2D:16' clustersYs = coder.nullcopy(clustersYs);
  i11 = clustersYs_size[0] * clustersYs_size[1];
  clustersYs_size[1] = 64;
  clustersYs_size[0] = 1;
  emxEnsureCapacity_cell_wrap_4(clustersYs_data, clustersYs_size, i11);

  // 'ClusterPoints2D:17' for i = 1:SmartLoaderCompilationConstants.MaxNumClusters 
  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[1].f1->size[1] = 0;
  clustersYs_data[1].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[2].f1->size[1] = 0;
  clustersYs_data[2].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[3].f1->size[1] = 0;
  clustersYs_data[3].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[4].f1->size[1] = 0;
  clustersYs_data[4].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[5].f1->size[1] = 0;
  clustersYs_data[5].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[6].f1->size[1] = 0;
  clustersYs_data[6].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[7].f1->size[1] = 0;
  clustersYs_data[7].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[8].f1->size[1] = 0;
  clustersYs_data[8].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[9].f1->size[1] = 0;
  clustersYs_data[9].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[10].f1->size[1] = 0;
  clustersYs_data[10].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[11].f1->size[1] = 0;
  clustersYs_data[11].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[12].f1->size[1] = 0;
  clustersYs_data[12].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[13].f1->size[1] = 0;
  clustersYs_data[13].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[14].f1->size[1] = 0;
  clustersYs_data[14].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[15].f1->size[1] = 0;
  clustersYs_data[15].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[16].f1->size[1] = 0;
  clustersYs_data[16].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[17].f1->size[1] = 0;
  clustersYs_data[17].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[18].f1->size[1] = 0;
  clustersYs_data[18].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[19].f1->size[1] = 0;
  clustersYs_data[19].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[20].f1->size[1] = 0;
  clustersYs_data[20].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[21].f1->size[1] = 0;
  clustersYs_data[21].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[22].f1->size[1] = 0;
  clustersYs_data[22].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[23].f1->size[1] = 0;
  clustersYs_data[23].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[24].f1->size[1] = 0;
  clustersYs_data[24].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[25].f1->size[1] = 0;
  clustersYs_data[25].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[26].f1->size[1] = 0;
  clustersYs_data[26].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[27].f1->size[1] = 0;
  clustersYs_data[27].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[28].f1->size[1] = 0;
  clustersYs_data[28].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[29].f1->size[1] = 0;
  clustersYs_data[29].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[30].f1->size[1] = 0;
  clustersYs_data[30].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[31].f1->size[1] = 0;
  clustersYs_data[31].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[32].f1->size[1] = 0;
  clustersYs_data[32].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[33].f1->size[1] = 0;
  clustersYs_data[33].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[34].f1->size[1] = 0;
  clustersYs_data[34].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[35].f1->size[1] = 0;
  clustersYs_data[35].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[36].f1->size[1] = 0;
  clustersYs_data[36].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[37].f1->size[1] = 0;
  clustersYs_data[37].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[38].f1->size[1] = 0;
  clustersYs_data[38].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[39].f1->size[1] = 0;
  clustersYs_data[39].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[40].f1->size[1] = 0;
  clustersYs_data[40].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[41].f1->size[1] = 0;
  clustersYs_data[41].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[42].f1->size[1] = 0;
  clustersYs_data[42].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[43].f1->size[1] = 0;
  clustersYs_data[43].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[44].f1->size[1] = 0;
  clustersYs_data[44].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[45].f1->size[1] = 0;
  clustersYs_data[45].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[46].f1->size[1] = 0;
  clustersYs_data[46].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[47].f1->size[1] = 0;
  clustersYs_data[47].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[48].f1->size[1] = 0;
  clustersYs_data[48].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[49].f1->size[1] = 0;
  clustersYs_data[49].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[50].f1->size[1] = 0;
  clustersYs_data[50].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[51].f1->size[1] = 0;
  clustersYs_data[51].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[52].f1->size[1] = 0;
  clustersYs_data[52].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[53].f1->size[1] = 0;
  clustersYs_data[53].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[54].f1->size[1] = 0;
  clustersYs_data[54].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[55].f1->size[1] = 0;
  clustersYs_data[55].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[56].f1->size[1] = 0;
  clustersYs_data[56].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[57].f1->size[1] = 0;
  clustersYs_data[57].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[58].f1->size[1] = 0;
  clustersYs_data[58].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[59].f1->size[1] = 0;
  clustersYs_data[59].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[60].f1->size[1] = 0;
  clustersYs_data[60].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[61].f1->size[1] = 0;
  clustersYs_data[61].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[62].f1->size[1] = 0;
  clustersYs_data[62].f1->size[0] = 0;

  // 'ClusterPoints2D:18' clustersYs{i} = zeros(0,0,'like', ptr);
  clustersYs_data[63].f1->size[1] = 0;
  clustersYs_data[63].f1->size[0] = 0;

  //  hold the first cluster - this is also helps for matlab coder determine the cell array type 
  // 'ClusterPoints2D:22' coder.varsize('ys','xs','ys2','xs2', [SmartLoaderCompilationConstants.MaxPointCloudSize 1], [1 0]); 
  //  coder.varsize('ptr', [SmartLoaderCompilationConstants.MaxPointCloudSize 2], [1 0]); 
  // 'ClusterPoints2D:24' xs = ptr(end,1);
  input_sizes_idx_0 = (ptr_size[0] - 1) << 1;
  SD->u2.f11.xs_data[0] = ptr_data[input_sizes_idx_0];

  // 'ClusterPoints2D:25' ys = ptr(end,2);
  SD->u2.f11.ys_data[0] = ptr_data[1 + input_sizes_idx_0];

  //  remove the last point
  // 'ClusterPoints2D:28' ptr = ptr(1:(end-1),:);
  if (1 > ptr_size[0] - 1) {
    loop_ub = 0;
  } else {
    loop_ub = ptr_size[0] - 1;
  }

  for (i11 = 0; i11 < loop_ub; i11++) {
    input_sizes_idx_0 = i11 << 1;
    SD->u2.f11.ptr_data[input_sizes_idx_0] = ptr_data[input_sizes_idx_0];
    input_sizes_idx_0++;
    SD->u2.f11.ptr_data[input_sizes_idx_0] = ptr_data[input_sizes_idx_0];
  }

  ptr_size[1] = 2;
  ptr_size[0] = loop_ub;
  loop_ub <<= 1;
  if (0 <= loop_ub - 1) {
    memcpy(&ptr_data[0], &SD->u2.f11.ptr_data[0], (unsigned int)(loop_ub * (int)
            sizeof(float)));
  }

  // 'ClusterPoints2D:30' clustersXs{1} = xs;
  i11 = clustersXs_data[0].f1->size[0] * clustersXs_data[0].f1->size[1];
  clustersXs_data[0].f1->size[1] = 1;
  clustersXs_data[0].f1->size[0] = 1;
  emxEnsureCapacity_real32_T(clustersXs_data[0].f1, i11);
  clustersXs_data[0].f1->data[0] = SD->u2.f11.xs_data[0];

  // 'ClusterPoints2D:31' clustersYs{1} = ys;
  i11 = clustersYs_data[0].f1->size[0] * clustersYs_data[0].f1->size[1];
  clustersYs_data[0].f1->size[1] = 1;
  clustersYs_data[0].f1->size[0] = 1;
  emxEnsureCapacity_real32_T(clustersYs_data[0].f1, i11);
  clustersYs_data[0].f1->data[0] = SD->u2.f11.ys_data[0];

  // 'ClusterPoints2D:34' cellArrayIndex = 1;
  cellArrayIndex = 0;

  // 'ClusterPoints2D:35' while ~isempty(ptr)
  emxInit_real32_T(&distanceMat, 2);
  emxInitMatrix_cell_wrap_4(reshapes);
  emxInit_real32_T(&ex, 1);
  emxInit_int32_T(&idx, 1);
  emxInitMatrix_cell_wrap_4(b_reshapes);
  emxInitMatrix_cell_wrap_4(c_reshapes);
  emxInit_real32_T(&d_reshapes, 2);
  do {
    exitg1 = 0;
    if (ptr_size[0] != 0) {
      //  Calculate the distance from the current cluster to all the other points - 
      //  calcualte the distance from all the points in a cluster to all the other points 
      // 'ClusterPoints2D:39' ptrTmp = [clustersXs{cellArrayIndex} clustersYs{cellArrayIndex}]; 
      if ((clustersXs_data[cellArrayIndex].f1->size[0] != 0) &&
          (clustersXs_data[cellArrayIndex].f1->size[1] != 0)) {
        m = clustersXs_data[cellArrayIndex].f1->size[0];
      } else if ((clustersYs_data[cellArrayIndex].f1->size[0] != 0) &&
                 (clustersYs_data[cellArrayIndex].f1->size[1] != 0)) {
        m = clustersYs_data[cellArrayIndex].f1->size[0];
      } else {
        m = clustersXs_data[cellArrayIndex].f1->size[0];
        if (m <= 0) {
          m = 0;
        }

        if (clustersYs_data[cellArrayIndex].f1->size[0] > m) {
          m = clustersYs_data[cellArrayIndex].f1->size[0];
        }
      }

      empty_non_axis_sizes = (m == 0);
      if (empty_non_axis_sizes || ((clustersXs_data[cellArrayIndex].f1->size[0]
            != 0) && (clustersXs_data[cellArrayIndex].f1->size[1] != 0))) {
        input_sizes_idx_1 = (signed char)clustersXs_data[cellArrayIndex]
          .f1->size[1];
      } else {
        input_sizes_idx_1 = 0;
      }

      loop_ub = input_sizes_idx_1;
      if ((input_sizes_idx_1 == clustersXs_data[cellArrayIndex].f1->size[1]) &&
          (m == clustersXs_data[cellArrayIndex].f1->size[0])) {
        i11 = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
        reshapes[0].f1->size[1] = input_sizes_idx_1;
        reshapes[0].f1->size[0] = m;
        emxEnsureCapacity_real32_T(reshapes[0].f1, i11);
        loop_ub = input_sizes_idx_1 * m;
        for (i11 = 0; i11 < loop_ub; i11++) {
          reshapes[0].f1->data[i11] = clustersXs_data[cellArrayIndex].f1->
            data[i11];
        }
      } else {
        i11 = 0;
        i12 = 0;
        n = 0;
        i13 = 0;
        i14 = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
        reshapes[0].f1->size[1] = input_sizes_idx_1;
        reshapes[0].f1->size[0] = m;
        emxEnsureCapacity_real32_T(reshapes[0].f1, i14);
        for (i14 = 0; i14 < m * loop_ub; i14++) {
          reshapes[0].f1->data[i12 + reshapes[0].f1->size[1] * i11] =
            clustersXs_data[cellArrayIndex].f1->data[i13 +
            clustersXs_data[cellArrayIndex].f1->size[1] * n];
          i11++;
          n++;
          if (i11 > reshapes[0].f1->size[0] - 1) {
            i11 = 0;
            i12++;
          }

          if (n > clustersXs_data[cellArrayIndex].f1->size[0] - 1) {
            n = 0;
            i13++;
          }
        }
      }

      if (empty_non_axis_sizes || ((clustersYs_data[cellArrayIndex].f1->size[0]
            != 0) && (clustersYs_data[cellArrayIndex].f1->size[1] != 0))) {
        input_sizes_idx_1 = (signed char)clustersYs_data[cellArrayIndex]
          .f1->size[1];
      } else {
        input_sizes_idx_1 = 0;
      }

      loop_ub = input_sizes_idx_1;
      if ((input_sizes_idx_1 == clustersYs_data[cellArrayIndex].f1->size[1]) &&
          (m == clustersYs_data[cellArrayIndex].f1->size[0])) {
        i11 = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
        reshapes[1].f1->size[1] = input_sizes_idx_1;
        reshapes[1].f1->size[0] = m;
        emxEnsureCapacity_real32_T(reshapes[1].f1, i11);
        loop_ub = input_sizes_idx_1 * m;
        for (i11 = 0; i11 < loop_ub; i11++) {
          reshapes[1].f1->data[i11] = clustersYs_data[cellArrayIndex].f1->
            data[i11];
        }
      } else {
        i11 = 0;
        i12 = 0;
        n = 0;
        i13 = 0;
        i14 = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
        reshapes[1].f1->size[1] = input_sizes_idx_1;
        reshapes[1].f1->size[0] = m;
        emxEnsureCapacity_real32_T(reshapes[1].f1, i14);
        for (i14 = 0; i14 < m * loop_ub; i14++) {
          reshapes[1].f1->data[i12 + reshapes[1].f1->size[1] * i11] =
            clustersYs_data[cellArrayIndex].f1->data[i13 +
            clustersYs_data[cellArrayIndex].f1->size[1] * n];
          i11++;
          n++;
          if (i11 > reshapes[1].f1->size[0] - 1) {
            i11 = 0;
            i12++;
          }

          if (n > clustersYs_data[cellArrayIndex].f1->size[0] - 1) {
            n = 0;
            i13++;
          }
        }
      }

      // 'ClusterPoints2D:40' distanceMat = pdist2(ptrTmp, ptr);
      i11 = d_reshapes->size[0] * d_reshapes->size[1];
      d_reshapes->size[1] = reshapes[0].f1->size[1] + reshapes[1].f1->size[1];
      d_reshapes->size[0] = reshapes[0].f1->size[0];
      emxEnsureCapacity_real32_T(d_reshapes, i11);
      loop_ub = reshapes[0].f1->size[0];
      for (i11 = 0; i11 < loop_ub; i11++) {
        input_sizes_idx_0 = reshapes[0].f1->size[1];
        for (i12 = 0; i12 < input_sizes_idx_0; i12++) {
          d_reshapes->data[i12 + d_reshapes->size[1] * i11] = reshapes[0]
            .f1->data[i12 + reshapes[0].f1->size[1] * i11];
        }
      }

      loop_ub = reshapes[1].f1->size[0];
      for (i11 = 0; i11 < loop_ub; i11++) {
        input_sizes_idx_0 = reshapes[1].f1->size[1];
        for (i12 = 0; i12 < input_sizes_idx_0; i12++) {
          d_reshapes->data[(i12 + reshapes[0].f1->size[1]) + d_reshapes->size[1]
            * i11] = reshapes[1].f1->data[i12 + reshapes[1].f1->size[1] * i11];
        }
      }

      pdist2(SD, d_reshapes, ptr_data, ptr_size, distanceMat);

      // 'ClusterPoints2D:41' if size(distanceMat,1) == 1
      if (distanceMat->size[0] == 1) {
        // 'ClusterPoints2D:42' [minVal, minInd] = min(distanceMat,[],2);
        n = distanceMat->size[1];
        i11 = ex->size[0];
        ex->size[0] = 1;
        emxEnsureCapacity_real32_T(ex, i11);
        i11 = idx->size[0];
        idx->size[0] = 1;
        emxEnsureCapacity_int32_T(idx, i11);
        idx->data[0] = 1;
        ex->data[0] = distanceMat->data[0];
        for (loop_ub = 2; loop_ub <= n; loop_ub++) {
          if (ex->data[0] > distanceMat->data[loop_ub - 1]) {
            ex->data[0] = distanceMat->data[loop_ub - 1];
            idx->data[0] = loop_ub;
          }
        }

        loop_ub = ex->size[0];
        for (i11 = 0; i11 < loop_ub; i11++) {
          minVal_data[i11] = ex->data[i11];
        }

        loop_ub = idx->size[0];
        for (i11 = 0; i11 < loop_ub; i11++) {
          minInd_data[i11] = idx->data[i11];
        }
      } else {
        // 'ClusterPoints2D:43' else
        // 'ClusterPoints2D:44' [minValTmp, minIndTmp] = min(distanceMat,[],2);
        m = distanceMat->size[0];
        n = distanceMat->size[1];
        distanceMat_idx_0 = (unsigned int)distanceMat->size[0];
        i11 = ex->size[0];
        ex->size[0] = (int)distanceMat_idx_0;
        emxEnsureCapacity_real32_T(ex, i11);
        i11 = idx->size[0];
        idx->size[0] = distanceMat->size[0];
        emxEnsureCapacity_int32_T(idx, i11);
        loop_ub = distanceMat->size[0];
        for (i11 = 0; i11 < loop_ub; i11++) {
          idx->data[i11] = 1;
        }

        if (distanceMat->size[0] >= 1) {
          for (input_sizes_idx_0 = 0; input_sizes_idx_0 < m; input_sizes_idx_0++)
          {
            ex->data[input_sizes_idx_0] = distanceMat->data[distanceMat->size[1]
              * input_sizes_idx_0];
            for (loop_ub = 2; loop_ub <= n; loop_ub++) {
              if (ex->data[input_sizes_idx_0] > distanceMat->data[(loop_ub +
                   distanceMat->size[1] * input_sizes_idx_0) - 1]) {
                ex->data[input_sizes_idx_0] = distanceMat->data[(loop_ub +
                  distanceMat->size[1] * input_sizes_idx_0) - 1];
                idx->data[input_sizes_idx_0] = loop_ub;
              }
            }
          }
        }

        // 'ClusterPoints2D:45' [~, minMinInd] = min(minValTmp);
        n = ex->size[0];
        if (ex->size[0] <= 2) {
          if (ex->size[0] == 1) {
            input_sizes_idx_0 = 0;
          } else {
            input_sizes_idx_0 = (ex->data[0] > ex->data[1]);
          }
        } else {
          b_ex = ex->data[0];
          input_sizes_idx_0 = 0;
          for (loop_ub = 2; loop_ub <= n; loop_ub++) {
            if (b_ex > ex->data[loop_ub - 1]) {
              b_ex = ex->data[loop_ub - 1];
              input_sizes_idx_0 = loop_ub - 1;
            }
          }
        }

        // 'ClusterPoints2D:45' ~
        // 'ClusterPoints2D:47' minVal = minValTmp(minMinInd);
        minVal_data[0] = ex->data[input_sizes_idx_0];

        // 'ClusterPoints2D:48' minInd = minIndTmp(minMinInd);
        minInd_data[0] = idx->data[input_sizes_idx_0];
      }

      // 'ClusterPoints2D:52' if minVal <= maxDistance
      b_minVal_data[0] = (minVal_data[0] <= maxDistance);
      if (ifWhileCond(b_minVal_data)) {
        //  Same cluster
        // 'ClusterPoints2D:54' clustersXs{cellArrayIndex} = [clustersXs{cellArrayIndex}; ptr(minInd,1)]; 
        if ((clustersXs_data[cellArrayIndex].f1->size[0] != 0) &&
            (clustersXs_data[cellArrayIndex].f1->size[1] != 0)) {
          input_sizes_idx_1 = (signed char)clustersXs_data[cellArrayIndex]
            .f1->size[1];
        } else {
          input_sizes_idx_1 = 1;
        }

        if ((input_sizes_idx_1 == 0) || ((clustersXs_data[cellArrayIndex]
              .f1->size[0] != 0) && (clustersXs_data[cellArrayIndex].f1->size[1]
              != 0))) {
          input_sizes_idx_0 = clustersXs_data[cellArrayIndex].f1->size[0];
        } else {
          input_sizes_idx_0 = 0;
        }

        loop_ub = input_sizes_idx_1;
        if ((input_sizes_idx_1 == clustersXs_data[cellArrayIndex].f1->size[1]) &&
            (input_sizes_idx_0 == clustersXs_data[cellArrayIndex].f1->size[0]))
        {
          i11 = b_reshapes[0].f1->size[0] * b_reshapes[0].f1->size[1];
          b_reshapes[0].f1->size[1] = input_sizes_idx_1;
          b_reshapes[0].f1->size[0] = input_sizes_idx_0;
          emxEnsureCapacity_real32_T(b_reshapes[0].f1, i11);
          loop_ub = input_sizes_idx_1 * input_sizes_idx_0;
          for (i11 = 0; i11 < loop_ub; i11++) {
            b_reshapes[0].f1->data[i11] = clustersXs_data[cellArrayIndex]
              .f1->data[i11];
          }
        } else {
          i11 = 0;
          i12 = 0;
          n = 0;
          i13 = 0;
          i14 = b_reshapes[0].f1->size[0] * b_reshapes[0].f1->size[1];
          b_reshapes[0].f1->size[1] = input_sizes_idx_1;
          b_reshapes[0].f1->size[0] = input_sizes_idx_0;
          emxEnsureCapacity_real32_T(b_reshapes[0].f1, i14);
          for (i14 = 0; i14 < input_sizes_idx_0 * loop_ub; i14++) {
            b_reshapes[0].f1->data[i12 + b_reshapes[0].f1->size[1] * i11] =
              clustersXs_data[cellArrayIndex].f1->data[i13 +
              clustersXs_data[cellArrayIndex].f1->size[1] * n];
            i11++;
            n++;
            if (i11 > b_reshapes[0].f1->size[0] - 1) {
              i11 = 0;
              i12++;
            }

            if (n > clustersXs_data[cellArrayIndex].f1->size[0] - 1) {
              n = 0;
              i13++;
            }
          }
        }

        loop_ub = input_sizes_idx_1;
        i11 = 0;
        i12 = 0;
        n = b_reshapes[1].f1->size[0] * b_reshapes[1].f1->size[1];
        b_reshapes[1].f1->size[1] = input_sizes_idx_1;
        b_reshapes[1].f1->size[0] = 1;
        emxEnsureCapacity_real32_T(b_reshapes[1].f1, n);
        for (n = 0; n < loop_ub; n++) {
          b_reshapes[1].f1->data[i11] = ptr_data[(minInd_data[i12] - 1) << 1];
          i12++;
          i11++;
        }

        i11 = clustersXs_data[cellArrayIndex].f1->size[0] *
          clustersXs_data[cellArrayIndex].f1->size[1];
        clustersXs_data[cellArrayIndex].f1->size[1] = b_reshapes[0].f1->size[1];
        clustersXs_data[cellArrayIndex].f1->size[0] = b_reshapes[0].f1->size[0]
          + b_reshapes[1].f1->size[0];
        emxEnsureCapacity_real32_T(clustersXs_data[cellArrayIndex].f1, i11);
        loop_ub = b_reshapes[0].f1->size[0];
        for (i11 = 0; i11 < loop_ub; i11++) {
          input_sizes_idx_0 = b_reshapes[0].f1->size[1];
          for (i12 = 0; i12 < input_sizes_idx_0; i12++) {
            clustersXs_data[cellArrayIndex].f1->data[i12 +
              clustersXs_data[cellArrayIndex].f1->size[1] * i11] = b_reshapes[0]
              .f1->data[i12 + b_reshapes[0].f1->size[1] * i11];
          }
        }

        loop_ub = b_reshapes[1].f1->size[0];
        for (i11 = 0; i11 < loop_ub; i11++) {
          input_sizes_idx_0 = b_reshapes[1].f1->size[1];
          for (i12 = 0; i12 < input_sizes_idx_0; i12++) {
            clustersXs_data[cellArrayIndex].f1->data[i12 +
              clustersXs_data[cellArrayIndex].f1->size[1] * (i11 + b_reshapes[0]
              .f1->size[0])] = b_reshapes[1].f1->data[i12 + b_reshapes[1]
              .f1->size[1] * i11];
          }
        }

        // 'ClusterPoints2D:55' clustersYs{cellArrayIndex} = [clustersYs{cellArrayIndex}; ptr(minInd,2)]; 
        if ((clustersYs_data[cellArrayIndex].f1->size[0] != 0) &&
            (clustersYs_data[cellArrayIndex].f1->size[1] != 0)) {
          input_sizes_idx_1 = (signed char)clustersYs_data[cellArrayIndex]
            .f1->size[1];
        } else {
          input_sizes_idx_1 = 1;
        }

        if ((input_sizes_idx_1 == 0) || ((clustersYs_data[cellArrayIndex]
              .f1->size[0] != 0) && (clustersYs_data[cellArrayIndex].f1->size[1]
              != 0))) {
          input_sizes_idx_0 = clustersYs_data[cellArrayIndex].f1->size[0];
        } else {
          input_sizes_idx_0 = 0;
        }

        loop_ub = input_sizes_idx_1;
        if ((input_sizes_idx_1 == clustersYs_data[cellArrayIndex].f1->size[1]) &&
            (input_sizes_idx_0 == clustersYs_data[cellArrayIndex].f1->size[0]))
        {
          i11 = c_reshapes[0].f1->size[0] * c_reshapes[0].f1->size[1];
          c_reshapes[0].f1->size[1] = input_sizes_idx_1;
          c_reshapes[0].f1->size[0] = input_sizes_idx_0;
          emxEnsureCapacity_real32_T(c_reshapes[0].f1, i11);
          loop_ub = input_sizes_idx_1 * input_sizes_idx_0;
          for (i11 = 0; i11 < loop_ub; i11++) {
            c_reshapes[0].f1->data[i11] = clustersYs_data[cellArrayIndex]
              .f1->data[i11];
          }
        } else {
          i11 = 0;
          i12 = 0;
          n = 0;
          i13 = 0;
          i14 = c_reshapes[0].f1->size[0] * c_reshapes[0].f1->size[1];
          c_reshapes[0].f1->size[1] = input_sizes_idx_1;
          c_reshapes[0].f1->size[0] = input_sizes_idx_0;
          emxEnsureCapacity_real32_T(c_reshapes[0].f1, i14);
          for (i14 = 0; i14 < input_sizes_idx_0 * loop_ub; i14++) {
            c_reshapes[0].f1->data[i12 + c_reshapes[0].f1->size[1] * i11] =
              clustersYs_data[cellArrayIndex].f1->data[i13 +
              clustersYs_data[cellArrayIndex].f1->size[1] * n];
            i11++;
            n++;
            if (i11 > c_reshapes[0].f1->size[0] - 1) {
              i11 = 0;
              i12++;
            }

            if (n > clustersYs_data[cellArrayIndex].f1->size[0] - 1) {
              n = 0;
              i13++;
            }
          }
        }

        loop_ub = input_sizes_idx_1;
        i11 = 0;
        i12 = 0;
        n = c_reshapes[1].f1->size[0] * c_reshapes[1].f1->size[1];
        c_reshapes[1].f1->size[1] = input_sizes_idx_1;
        c_reshapes[1].f1->size[0] = 1;
        emxEnsureCapacity_real32_T(c_reshapes[1].f1, n);
        for (n = 0; n < loop_ub; n++) {
          c_reshapes[1].f1->data[i11] = ptr_data[1 + ((minInd_data[i12] - 1) <<
            1)];
          i12++;
          i11++;
        }

        i11 = clustersYs_data[cellArrayIndex].f1->size[0] *
          clustersYs_data[cellArrayIndex].f1->size[1];
        clustersYs_data[cellArrayIndex].f1->size[1] = c_reshapes[0].f1->size[1];
        clustersYs_data[cellArrayIndex].f1->size[0] = c_reshapes[0].f1->size[0]
          + c_reshapes[1].f1->size[0];
        emxEnsureCapacity_real32_T(clustersYs_data[cellArrayIndex].f1, i11);
        loop_ub = c_reshapes[0].f1->size[0];
        for (i11 = 0; i11 < loop_ub; i11++) {
          input_sizes_idx_0 = c_reshapes[0].f1->size[1];
          for (i12 = 0; i12 < input_sizes_idx_0; i12++) {
            clustersYs_data[cellArrayIndex].f1->data[i12 +
              clustersYs_data[cellArrayIndex].f1->size[1] * i11] = c_reshapes[0]
              .f1->data[i12 + c_reshapes[0].f1->size[1] * i11];
          }
        }

        loop_ub = c_reshapes[1].f1->size[0];
        for (i11 = 0; i11 < loop_ub; i11++) {
          input_sizes_idx_0 = c_reshapes[1].f1->size[1];
          for (i12 = 0; i12 < input_sizes_idx_0; i12++) {
            clustersYs_data[cellArrayIndex].f1->data[i12 +
              clustersYs_data[cellArrayIndex].f1->size[1] * (i11 + c_reshapes[0]
              .f1->size[0])] = c_reshapes[1].f1->data[i12 + c_reshapes[1]
              .f1->size[1] * i11];
          }
        }

        // 'ClusterPoints2D:57' ptr(minInd,:) = [];
        tmp_size[1] = 2;
        tmp_size[0] = ptr_size[0];
        loop_ub = 2 * ptr_size[0];
        if (0 <= loop_ub - 1) {
          memcpy(&SD->u2.f11.tmp_data[0], &ptr_data[0], (unsigned int)(loop_ub *
                  (int)sizeof(float)));
        }

        b_minInd_data[0] = minInd_data[0];
        c_nullAssignment(SD, SD->u2.f11.tmp_data, tmp_size, b_minInd_data);
        ptr_size[1] = 2;
        ptr_size[0] = tmp_size[0];
        loop_ub = tmp_size[1] * tmp_size[0];
        if (0 <= loop_ub - 1) {
          memcpy(&ptr_data[0], &SD->u2.f11.tmp_data[0], (unsigned int)(loop_ub *
                  (int)sizeof(float)));
        }
      } else {
        // 'ClusterPoints2D:58' else
        //  new cluster
        // 'ClusterPoints2D:60' xs2 = ptr(minInd,1);
        input_sizes_idx_0 = (minInd_data[0] - 1) << 1;

        // 'ClusterPoints2D:61' cellArrayIndex = cellArrayIndex + 1;
        cellArrayIndex++;

        // 'ClusterPoints2D:63' if cellArrayIndex > SmartLoaderCompilationConstants.MaxNumClusters 
        if (cellArrayIndex + 1 > 64) {
          //  Run time check for the maximum cell size, use this to avoid run time exception  
          // 'ClusterPoints2D:65' status = false;
          *status = false;
          exitg1 = 1;
        } else {
          // 'ClusterPoints2D:69' clustersXs{cellArrayIndex} = xs2;
          i11 = clustersXs_data[cellArrayIndex].f1->size[0] *
            clustersXs_data[cellArrayIndex].f1->size[1];
          clustersXs_data[cellArrayIndex].f1->size[1] = 1;
          clustersXs_data[cellArrayIndex].f1->size[0] = 1;
          emxEnsureCapacity_real32_T(clustersXs_data[cellArrayIndex].f1, i11);
          clustersXs_data[cellArrayIndex].f1->data[0] =
            ptr_data[input_sizes_idx_0];

          // 'ClusterPoints2D:71' ys2 = ptr(minInd,2);
          minVal_data[0] = ptr_data[1 + input_sizes_idx_0];

          // 'ClusterPoints2D:72' clustersYs{cellArrayIndex} = ys2;
          i11 = clustersYs_data[cellArrayIndex].f1->size[0] *
            clustersYs_data[cellArrayIndex].f1->size[1];
          clustersYs_data[cellArrayIndex].f1->size[1] = 1;
          clustersYs_data[cellArrayIndex].f1->size[0] = 1;
          emxEnsureCapacity_real32_T(clustersYs_data[cellArrayIndex].f1, i11);
          clustersYs_data[cellArrayIndex].f1->data[0] = ptr_data[1 +
            input_sizes_idx_0];

          // 'ClusterPoints2D:74' ptr(minInd,:) = [];
          tmp_size[1] = 2;
          tmp_size[0] = ptr_size[0];
          loop_ub = 2 * ptr_size[0];
          if (0 <= loop_ub - 1) {
            memcpy(&SD->u2.f11.tmp_data[0], &ptr_data[0], (unsigned int)(loop_ub
                    * (int)sizeof(float)));
          }

          b_minInd_data[0] = minInd_data[0];
          c_nullAssignment(SD, SD->u2.f11.tmp_data, tmp_size, b_minInd_data);
          ptr_size[1] = 2;
          ptr_size[0] = tmp_size[0];
          loop_ub = tmp_size[1] * tmp_size[0];
          if (0 <= loop_ub - 1) {
            memcpy(&ptr_data[0], &SD->u2.f11.tmp_data[0], (unsigned int)(loop_ub
                    * (int)sizeof(float)));
          }
        }
      }
    } else {
      // 'ClusterPoints2D:79' status = true;
      *status = true;
      exitg1 = 1;
    }
  } while (exitg1 == 0);

  emxFree_real32_T(&d_reshapes);
  emxFreeMatrix_cell_wrap_4(c_reshapes);
  emxFreeMatrix_cell_wrap_4(b_reshapes);
  emxFree_int32_T(&idx);
  emxFree_real32_T(&ex);
  emxFreeMatrix_cell_wrap_4(reshapes);
  emxFree_real32_T(&distanceMat);
}

//
// function [pcFiltered] = FilterPointCloudAccordingToZdifferences(pc, diffThreshold)
// FILTERPOINTCLOUDACCORDINGTOZDIFFERENCES Summary of this function goes here
//    Detailed explanation goes here
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float pc_data[]
//                const int pc_size[2]
//                double diffThreshold
//                float pcFiltered_data[]
//                int pcFiltered_size[2]
// Return Type  : void
//
static void FilterPointCloudAccordingToZdifferences
  (PerceptionSmartLoaderStackData *SD, const float pc_data[], const int pc_size
   [2], double diffThreshold, float pcFiltered_data[], int pcFiltered_size[2])
{
  int loop_ub;
  int i;
  int iv0[1];
  float zMedian;
  int tmp_size[1];
  int b_tmp_size[1];
  int trueCount;
  int partialTrueCount;

  // 'FilterPointCloudAccordingToZdifferences:5' isMatlab2019B = false;
  // zCor = pc.Location(:,3);
  // 'FilterPointCloudAccordingToZdifferences:8' zCor = pc(:,3);
  loop_ub = pc_size[0];
  for (i = 0; i < loop_ub; i++) {
    SD->u3.f15.tmp_data[i] = pc_data[2 + 3 * i];
  }

  // 'FilterPointCloudAccordingToZdifferences:9' zMedian = median(zCor);
  iv0[0] = pc_size[0];
  zMedian = median(SD, SD->u3.f15.tmp_data, iv0);

  // 'FilterPointCloudAccordingToZdifferences:10' assert(numel(zMedian) == 1);
  // 'FilterPointCloudAccordingToZdifferences:11' inliearsInd = abs(zCor - zMedian) <= diffThreshold; 
  loop_ub = pc_size[0];
  tmp_size[0] = pc_size[0];
  for (i = 0; i < loop_ub; i++) {
    SD->u3.f15.b_tmp_data[i] = SD->u3.f15.tmp_data[i] - zMedian;
  }

  b_abs(SD->u3.f15.b_tmp_data, tmp_size, SD->u3.f15.tmp_data, b_tmp_size);
  loop_ub = b_tmp_size[0];
  for (i = 0; i < loop_ub; i++) {
    SD->u3.f15.inliearsInd_data[i] = (SD->u3.f15.tmp_data[i] <= diffThreshold);
  }

  //  sum(loaderReflectorPtrInd)
  // 'FilterPointCloudAccordingToZdifferences:14' if isMatlab2019B
  // 'FilterPointCloudAccordingToZdifferences:16' else
  // pcFiltered = select(pc, find(inliearsInd));
  // 'FilterPointCloudAccordingToZdifferences:18' pcFiltered = pc(inliearsInd, :); 
  loop_ub = b_tmp_size[0] - 1;
  trueCount = 0;
  for (i = 0; i <= loop_ub; i++) {
    if (SD->u3.f15.inliearsInd_data[i]) {
      trueCount++;
    }
  }

  partialTrueCount = 0;
  for (i = 0; i <= loop_ub; i++) {
    if (SD->u3.f15.inliearsInd_data[i]) {
      SD->u3.f15.c_tmp_data[partialTrueCount] = i + 1;
      partialTrueCount++;
    }
  }

  pcFiltered_size[1] = 3;
  pcFiltered_size[0] = trueCount;
  for (i = 0; i < trueCount; i++) {
    loop_ub = 3 * (SD->u3.f15.c_tmp_data[i] - 1);
    pcFiltered_data[3 * i] = pc_data[loop_ub];
    pcFiltered_data[1 + 3 * i] = pc_data[1 + loop_ub];
    pcFiltered_data[2 + 3 * i] = pc_data[2 + loop_ub];
  }

  //  figure, PlotPointCloud(pc);
  //  figure, PlotPointCloud(pcFiltered);
}

//
// function [smartLoaderStruct, ptCloudSenceXyz, ptCloudSenceIntensity] = ...
//     SmartLoaderAlignPointCloud(smartLoaderStruct, configParams, xyz, intensity)
// Align the point cloud to the sensor
//  used in order to first align a new point cloud an determine the rotation matrix for our coordinate system
// Arguments    : PerceptionSmartLoaderStackData *SD
//                PerceptionSmartLoaderStruct *smartLoaderStruct
//                const double configParams_pcAlignmentProjMat[12]
//                const double configParams_xyzLimits[6]
//                double configParams_minNumPointsInPc
//                const double xyz_data[]
//                const int xyz_size[2]
//                const double intensity_data[]
//                float ptCloudSenceXyz_data[]
//                int ptCloudSenceXyz_size[2]
//                float ptCloudSenceIntensity_data[]
//                int ptCloudSenceIntensity_size[1]
// Return Type  : void
//
static void SmartLoaderAlignPointCloud(PerceptionSmartLoaderStackData *SD,
  PerceptionSmartLoaderStruct *smartLoaderStruct, const double
  configParams_pcAlignmentProjMat[12], const double configParams_xyzLimits[6],
  double configParams_minNumPointsInPc, const double xyz_data[], const int
  xyz_size[2], const double intensity_data[], float ptCloudSenceXyz_data[], int
  ptCloudSenceXyz_size[2], float ptCloudSenceIntensity_data[], int
  ptCloudSenceIntensity_size[1])
{
  double trans[16];
  double dv0[16];
  int i31;
  int srcHomogenious_tmp;
  double temp_tmp;
  double xyzLimitsInXyzCoordinateSystem[6];
  double srcHomogenious[8];
  int b_srcHomogenious_tmp;
  double b_temp_tmp;
  int c_srcHomogenious_tmp;
  int d_srcHomogenious_tmp;
  double c_temp_tmp;
  double d3;
  double d4;
  int trueCount;

  // 'SmartLoaderAlignPointCloud:6' if false
  // 'SmartLoaderAlignPointCloud:10' coder.varsize('indices', 'ptCloudSenceIntensity', [SmartLoaderCompilationConstants.MaxPointCloudSize 1], [1 0]); 
  // 'SmartLoaderAlignPointCloud:11' coder.varsize('xyzFiltered', 'ptCloudSenceXyz', [SmartLoaderCompilationConstants.MaxPointCloudSize 3], [1 0]); 
  // 'SmartLoaderAlignPointCloud:13' if size(xyz,1) > SmartLoaderCompilationConstants.MaxPointCloudSize 
  // 'SmartLoaderAlignPointCloud:21' if true
  //  first transform the xyz limits, then filter the points, then transform the points to the xyz aligned axis -  
  //  This process is faster than transform and filter.
  // 'SmartLoaderAlignPointCloud:24' trans = [configParams.pcAlignmentProjMat; [0 0 0 1]]; 
  trans[0] = configParams_pcAlignmentProjMat[0];
  trans[1] = configParams_pcAlignmentProjMat[1];
  trans[2] = configParams_pcAlignmentProjMat[2];
  trans[3] = configParams_pcAlignmentProjMat[3];
  trans[4] = configParams_pcAlignmentProjMat[4];
  trans[5] = configParams_pcAlignmentProjMat[5];
  trans[6] = configParams_pcAlignmentProjMat[6];
  trans[7] = configParams_pcAlignmentProjMat[7];
  trans[8] = configParams_pcAlignmentProjMat[8];
  trans[9] = configParams_pcAlignmentProjMat[9];
  trans[10] = configParams_pcAlignmentProjMat[10];
  trans[11] = configParams_pcAlignmentProjMat[11];
  trans[12] = 0.0;
  trans[13] = 0.0;
  trans[14] = 0.0;
  trans[15] = 1.0;

  // 'SmartLoaderAlignPointCloud:25' transInv = inv(trans);
  // 'SmartLoaderAlignPointCloud:26' xyzLimitsInXyzCoordinateSystem3x2 = TransformPointsForward3DAffineCompiledVersion(transInv, configParams.xyzLimits'); 
  //  The function transform 3d points using affine transformation
  //  This implementation is special impelmentation for coder that supposes to run faster then projective transformation 
  // 'TransformPointsForward3DAffineCompiledVersion:5' coder.inline('always');
  //  Assersions - make sure the data is right
  // 'TransformPointsForward3DAffineCompiledVersion:8' assert(isequal(trans(end,:), [0 0 0 1])); 
  // 'TransformPointsForward3DAffineCompiledVersion:9' assert(isequal(size(trans), [4 4])); 
  // 'TransformPointsForward3DAffineCompiledVersion:10' assert(isequal(size(src,2), 3)); 
  //  Comment for coder varsize command - updated on 2/2020 - varsize is non needed for this function because the size is determined from the parent function 
  // coder.varsize('srcHomogenious', [PcClassificationCompilationConstants.MaxPointCloudSize, 4], [1 0]); 
  // coder.varsize('dst', [PcClassificationCompilationConstants.MaxPointCloudSize, 3], [1 0]); 
  //  We shall remove the last line from the affine transformation - the last element is one, therefor there is no need to divide the result by 1 
  // 'TransformPointsForward3DAffineCompiledVersion:17' affineTrans = trans(1:3,:)'; 
  // 'TransformPointsForward3DAffineCompiledVersion:19' srcHomogenious = coder.nullcopy(zeros(size(src,1), 4)); 
  // 'TransformPointsForward3DAffineCompiledVersion:20' srcHomogenious(:, 1:3) = src; 
  // 'TransformPointsForward3DAffineCompiledVersion:21' srcHomogenious(:, 4) = 1; 
  // 'TransformPointsForward3DAffineCompiledVersion:23' dst = srcHomogenious * affineTrans; 
  // 'SmartLoaderAlignPointCloud:27' xyzLimitsInXyzCoordinateSystem = xyzLimitsInXyzCoordinateSystem3x2'; 
  inv(trans, dv0);
  for (i31 = 0; i31 < 2; i31++) {
    srcHomogenious_tmp = i31 << 2;
    srcHomogenious[srcHomogenious_tmp] = configParams_xyzLimits[i31];
    b_srcHomogenious_tmp = 1 + srcHomogenious_tmp;
    srcHomogenious[b_srcHomogenious_tmp] = configParams_xyzLimits[i31 + 2];
    c_srcHomogenious_tmp = 2 + srcHomogenious_tmp;
    srcHomogenious[c_srcHomogenious_tmp] = configParams_xyzLimits[i31 + 4];
    d_srcHomogenious_tmp = 3 + srcHomogenious_tmp;
    srcHomogenious[d_srcHomogenious_tmp] = 1.0;
    xyzLimitsInXyzCoordinateSystem[i31] = 0.0;
    xyzLimitsInXyzCoordinateSystem[i31] = ((dv0[0] *
      srcHomogenious[srcHomogenious_tmp] + dv0[1] *
      srcHomogenious[b_srcHomogenious_tmp]) + dv0[2] *
      srcHomogenious[c_srcHomogenious_tmp]) + dv0[3] *
      srcHomogenious[d_srcHomogenious_tmp];
    xyzLimitsInXyzCoordinateSystem[i31 + 2] = 0.0;
    xyzLimitsInXyzCoordinateSystem[i31 + 2] = ((dv0[4] * srcHomogenious[i31 << 2]
      + dv0[5] * srcHomogenious[1 + (i31 << 2)]) + dv0[6] * srcHomogenious[2 +
      (i31 << 2)]) + dv0[7] * srcHomogenious[3 + (i31 << 2)];
    xyzLimitsInXyzCoordinateSystem[i31 + 4] = 0.0;
    xyzLimitsInXyzCoordinateSystem[i31 + 4] = ((dv0[8] * srcHomogenious[i31 << 2]
      + dv0[9] * srcHomogenious[1 + (i31 << 2)]) + dv0[10] * srcHomogenious[2 +
      (i31 << 2)]) + dv0[11] * srcHomogenious[3 + (i31 << 2)];
  }

  // 'SmartLoaderAlignPointCloud:28' xyzLimitsInXyzCoordinateSystemMinMax = [min(xyzLimitsInXyzCoordinateSystem, [], 2) max(xyzLimitsInXyzCoordinateSystem, [], 2)]; 
  temp_tmp = xyzLimitsInXyzCoordinateSystem[0];
  if (xyzLimitsInXyzCoordinateSystem[0] > xyzLimitsInXyzCoordinateSystem[1]) {
    temp_tmp = xyzLimitsInXyzCoordinateSystem[1];
  }

  b_temp_tmp = xyzLimitsInXyzCoordinateSystem[0];
  if (xyzLimitsInXyzCoordinateSystem[0] < xyzLimitsInXyzCoordinateSystem[1]) {
    b_temp_tmp = xyzLimitsInXyzCoordinateSystem[1];
  }

  xyzLimitsInXyzCoordinateSystem[0] = temp_tmp;
  xyzLimitsInXyzCoordinateSystem[1] = b_temp_tmp;
  temp_tmp = xyzLimitsInXyzCoordinateSystem[2];
  if (xyzLimitsInXyzCoordinateSystem[2] > xyzLimitsInXyzCoordinateSystem[3]) {
    temp_tmp = xyzLimitsInXyzCoordinateSystem[3];
  }

  b_temp_tmp = xyzLimitsInXyzCoordinateSystem[2];
  if (xyzLimitsInXyzCoordinateSystem[2] < xyzLimitsInXyzCoordinateSystem[3]) {
    b_temp_tmp = xyzLimitsInXyzCoordinateSystem[3];
  }

  xyzLimitsInXyzCoordinateSystem[2] = temp_tmp;
  xyzLimitsInXyzCoordinateSystem[3] = b_temp_tmp;
  temp_tmp = xyzLimitsInXyzCoordinateSystem[4];
  if (xyzLimitsInXyzCoordinateSystem[4] > xyzLimitsInXyzCoordinateSystem[5]) {
    temp_tmp = xyzLimitsInXyzCoordinateSystem[5];
  }

  b_temp_tmp = xyzLimitsInXyzCoordinateSystem[4];
  if (xyzLimitsInXyzCoordinateSystem[4] < xyzLimitsInXyzCoordinateSystem[5]) {
    b_temp_tmp = xyzLimitsInXyzCoordinateSystem[5];
  }

  //  Take the min and max because we flip the axis
  // 'SmartLoaderAlignPointCloud:31' indices = xyz(:,1) >= xyzLimitsInXyzCoordinateSystemMinMax(1,1) & xyz(:,1) <= xyzLimitsInXyzCoordinateSystemMinMax(1,2) & ... 
  // 'SmartLoaderAlignPointCloud:32'         xyz(:,2) >= xyzLimitsInXyzCoordinateSystemMinMax(2,1) & xyz(:,2) <= xyzLimitsInXyzCoordinateSystemMinMax(2,2) & ... 
  // 'SmartLoaderAlignPointCloud:33'         xyz(:,3) >= xyzLimitsInXyzCoordinateSystemMinMax(3,1) & xyz(:,3) <= xyzLimitsInXyzCoordinateSystemMinMax(3,2); 
  srcHomogenious_tmp = xyz_size[0];
  for (i31 = 0; i31 < srcHomogenious_tmp; i31++) {
    c_temp_tmp = xyz_data[3 * i31];
    d3 = xyz_data[1 + 3 * i31];
    d4 = xyz_data[2 + 3 * i31];
    SD->u1.f10.indices_data[i31] = ((c_temp_tmp >=
      xyzLimitsInXyzCoordinateSystem[0]) && (c_temp_tmp <=
      xyzLimitsInXyzCoordinateSystem[1]) && (d3 >=
      xyzLimitsInXyzCoordinateSystem[2]) && (d3 <=
      xyzLimitsInXyzCoordinateSystem[3]) && (d4 >= temp_tmp) && (d4 <=
      b_temp_tmp));
  }

  //  sum(indices)
  // 'SmartLoaderAlignPointCloud:36' xyzFiltered = xyz(indices,:);
  srcHomogenious_tmp = xyz_size[0] - 1;
  trueCount = 0;
  for (c_srcHomogenious_tmp = 0; c_srcHomogenious_tmp <= srcHomogenious_tmp;
       c_srcHomogenious_tmp++) {
    if (SD->u1.f10.indices_data[c_srcHomogenious_tmp]) {
      trueCount++;
    }
  }

  b_srcHomogenious_tmp = 0;
  for (c_srcHomogenious_tmp = 0; c_srcHomogenious_tmp <= srcHomogenious_tmp;
       c_srcHomogenious_tmp++) {
    if (SD->u1.f10.indices_data[c_srcHomogenious_tmp]) {
      SD->u1.f10.tmp_data[b_srcHomogenious_tmp] = c_srcHomogenious_tmp + 1;
      b_srcHomogenious_tmp++;
    }
  }

  // 'SmartLoaderAlignPointCloud:37' if false
  // 'SmartLoaderAlignPointCloud:41' else
  // 'SmartLoaderAlignPointCloud:42' ptCloudSenceIntensity = cast(intensity(indices,:), 'single'); 
  srcHomogenious_tmp = xyz_size[0] - 1;
  d_srcHomogenious_tmp = 0;
  for (c_srcHomogenious_tmp = 0; c_srcHomogenious_tmp <= srcHomogenious_tmp;
       c_srcHomogenious_tmp++) {
    if (SD->u1.f10.indices_data[c_srcHomogenious_tmp]) {
      d_srcHomogenious_tmp++;
    }
  }

  b_srcHomogenious_tmp = 0;
  for (c_srcHomogenious_tmp = 0; c_srcHomogenious_tmp <= srcHomogenious_tmp;
       c_srcHomogenious_tmp++) {
    if (SD->u1.f10.indices_data[c_srcHomogenious_tmp]) {
      SD->u1.f10.b_tmp_data[b_srcHomogenious_tmp] = c_srcHomogenious_tmp + 1;
      b_srcHomogenious_tmp++;
    }
  }

  ptCloudSenceIntensity_size[0] = d_srcHomogenious_tmp;
  for (i31 = 0; i31 < d_srcHomogenious_tmp; i31++) {
    ptCloudSenceIntensity_data[i31] = (float)intensity_data
      [SD->u1.f10.b_tmp_data[i31] - 1];
  }

  // 'SmartLoaderAlignPointCloud:43' ptCloudSenceXyz = cast(TransformPointsForward3DAffineCompiledVersion(trans, xyzFiltered), 'single'); 
  //  The function transform 3d points using affine transformation
  //  This implementation is special impelmentation for coder that supposes to run faster then projective transformation 
  // 'TransformPointsForward3DAffineCompiledVersion:5' coder.inline('always');
  //  Assersions - make sure the data is right
  // 'TransformPointsForward3DAffineCompiledVersion:8' assert(isequal(trans(end,:), [0 0 0 1])); 
  // 'TransformPointsForward3DAffineCompiledVersion:9' assert(isequal(size(trans), [4 4])); 
  // 'TransformPointsForward3DAffineCompiledVersion:10' assert(isequal(size(src,2), 3)); 
  //  Comment for coder varsize command - updated on 2/2020 - varsize is non needed for this function because the size is determined from the parent function 
  // coder.varsize('srcHomogenious', [PcClassificationCompilationConstants.MaxPointCloudSize, 4], [1 0]); 
  // coder.varsize('dst', [PcClassificationCompilationConstants.MaxPointCloudSize, 3], [1 0]); 
  //  We shall remove the last line from the affine transformation - the last element is one, therefor there is no need to divide the result by 1 
  // 'TransformPointsForward3DAffineCompiledVersion:17' affineTrans = trans(1:3,:)'; 
  // 'TransformPointsForward3DAffineCompiledVersion:19' srcHomogenious = coder.nullcopy(zeros(size(src,1), 4)); 
  // 'TransformPointsForward3DAffineCompiledVersion:20' srcHomogenious(:, 1:3) = src; 
  for (i31 = 0; i31 < trueCount; i31++) {
    srcHomogenious_tmp = 3 * (SD->u1.f10.tmp_data[i31] - 1);
    b_srcHomogenious_tmp = i31 << 2;
    SD->u1.f10.srcHomogenious_data[b_srcHomogenious_tmp] =
      xyz_data[srcHomogenious_tmp];
    SD->u1.f10.srcHomogenious_data[1 + b_srcHomogenious_tmp] = xyz_data[1 +
      srcHomogenious_tmp];
    SD->u1.f10.srcHomogenious_data[2 + b_srcHomogenious_tmp] = xyz_data[2 +
      srcHomogenious_tmp];
  }

  // 'TransformPointsForward3DAffineCompiledVersion:21' srcHomogenious(:, 4) = 1; 
  for (i31 = 0; i31 < trueCount; i31++) {
    SD->u1.f10.srcHomogenious_data[3 + (i31 << 2)] = 1.0;
  }

  // 'TransformPointsForward3DAffineCompiledVersion:23' dst = srcHomogenious * affineTrans; 
  for (c_srcHomogenious_tmp = 0; c_srcHomogenious_tmp < trueCount;
       c_srcHomogenious_tmp++) {
    srcHomogenious_tmp = c_srcHomogenious_tmp << 2;
    temp_tmp = SD->u1.f10.srcHomogenious_data[1 + srcHomogenious_tmp];
    b_temp_tmp = SD->u1.f10.srcHomogenious_data[2 + srcHomogenious_tmp];
    c_temp_tmp = SD->u1.f10.srcHomogenious_data[3 + srcHomogenious_tmp];
    SD->u1.f10.dst_data[3 * c_srcHomogenious_tmp] =
      ((SD->u1.f10.srcHomogenious_data[srcHomogenious_tmp] *
        configParams_pcAlignmentProjMat[0] + temp_tmp * trans[1]) + b_temp_tmp *
       trans[2]) + c_temp_tmp * trans[3];
    SD->u1.f10.dst_data[1 + 3 * c_srcHomogenious_tmp] =
      ((SD->u1.f10.srcHomogenious_data[c_srcHomogenious_tmp << 2] *
        configParams_pcAlignmentProjMat[4] + temp_tmp * trans[5]) + b_temp_tmp *
       trans[6]) + c_temp_tmp * trans[7];
    SD->u1.f10.dst_data[2 + 3 * c_srcHomogenious_tmp] =
      ((SD->u1.f10.srcHomogenious_data[c_srcHomogenious_tmp << 2] *
        configParams_pcAlignmentProjMat[8] + temp_tmp * trans[9]) + b_temp_tmp *
       trans[10]) + c_temp_tmp * trans[11];
  }

  ptCloudSenceXyz_size[1] = 3;
  ptCloudSenceXyz_size[0] = trueCount;
  srcHomogenious_tmp = 3 * trueCount;
  for (i31 = 0; i31 < srcHomogenious_tmp; i31++) {
    ptCloudSenceXyz_data[i31] = (float)SD->u1.f10.dst_data[i31];
  }

  //  figure, PlotPointCloud([ptCloudSenceXyz double(ptCloudSenceIntensity)])
  // 'SmartLoaderAlignPointCloud:61' if size(ptCloudSenceXyz,1) < configParams.minNumPointsInPc 
  if (trueCount < configParams_minNumPointsInPc) {
    //  || configParams.debugIsPlayerMode
    // 'SmartLoaderAlignPointCloud:62' if size(ptCloudSenceXyz,1) < configParams.minNumPointsInPc && coder.target('Matlab') 
    // 'SmartLoaderAlignPointCloud:65' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eFailedNotEnoughPoints; 
    smartLoaderStruct->status =
      PerceptionSmartLoaderReturnValue_eFailedNotEnoughPoints;
  }

  //  Fit plane to the points inside the ROI
  // 'SmartLoaderAlignPointCloud:69' if false
}

//
// function [V,D] = SmartLoaderCalcEigen(xy)
// Calcualte the covarience matrix for the input data
//  substrct the mean from each dimention
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float xy_data[]
//                const int xy_size[2]
//                creal_T V[4]
//                creal_T D[4]
// Return Type  : void
//
static void SmartLoaderCalcEigen(PerceptionSmartLoaderStackData *SD, const float
  xy_data[], const int xy_size[2], creal_T V[4], creal_T D[4])
{
  int x_size_idx_0;
  float muj;
  int loop_ub;
  float muj_tmp;
  float temp;
  int m;
  float c[4];
  float covMatrixOutputInfInf_data[4];
  int LDA;
  double covMatrixOutputInfInf[4];
  creal_T dcv0[4];
  creal_T dcv1[4];
  int x_data_tmp;
  boolean_T b0;
  int ar;
  boolean_T b1;
  boolean_T b2;
  int i24;
  int i25;
  boolean_T b3;
  boolean_T b4;
  int i26;
  int w;
  int i27;
  int i28;
  int i29;

  // 'SmartLoaderCalcEigen:5' coder.varsize('covMatrix', 'V', 'D', [2 2], [0 0]); 
  // 'SmartLoaderCalcEigen:6' covMatrixOutputInfInf = cov(xy);
  if (xy_size[0] == 1) {
    muj = (xy_data[0] + xy_data[1]) / 2.0F;
    muj_tmp = xy_data[0] - muj;
    temp = muj_tmp * muj_tmp;
    muj_tmp = xy_data[1] - muj;
    temp += muj_tmp * muj_tmp;
    loop_ub = 1;
    covMatrixOutputInfInf_data[0] = temp;
  } else {
    x_size_idx_0 = xy_size[0];
    loop_ub = xy_size[1] * xy_size[0];
    if (0 <= loop_ub - 1) {
      memcpy(&SD->u1.f0.x_data[0], &xy_data[0], (unsigned int)(loop_ub * (int)
              sizeof(float)));
    }

    m = xy_size[0] - 1;
    c[0] = 0.0F;
    c[1] = 0.0F;
    c[2] = 0.0F;
    c[3] = 0.0F;
    if (xy_size[0] != 0) {
      LDA = xy_size[0];
      if (xy_size[0] >= 2) {
        muj = 0.0F;
        for (loop_ub = 0; loop_ub <= m; loop_ub++) {
          muj += SD->u1.f0.x_data[loop_ub << 1];
        }

        muj_tmp = (float)(m + 1);
        muj /= muj_tmp;
        for (loop_ub = 0; loop_ub <= m; loop_ub++) {
          x_data_tmp = loop_ub << 1;
          SD->u1.f0.x_data[x_data_tmp] -= muj;
        }

        muj = 0.0F;
        for (loop_ub = 0; loop_ub <= m; loop_ub++) {
          muj += SD->u1.f0.x_data[1 + (loop_ub << 1)];
        }

        muj /= muj_tmp;
        for (loop_ub = 0; loop_ub <= m; loop_ub++) {
          x_data_tmp = 1 + (loop_ub << 1);
          SD->u1.f0.x_data[x_data_tmp] -= muj;
        }

        muj_tmp = 1.0F / ((float)xy_size[0] - 1.0F);
        b0 = true;
        loop_ub = 0;
        for (x_data_tmp = 1; x_data_tmp < 3; x_data_tmp++) {
          if (b0) {
            b0 = false;
            loop_ub = (((x_data_tmp - 1) % 2) << 1) + (x_data_tmp - 1) / 2;
          } else {
            loop_ub += 2;
            if (loop_ub > 3) {
              loop_ub -= 3;
            }
          }

          c[loop_ub] = 0.0F;
        }

        b0 = true;
        loop_ub = 0;
        for (x_data_tmp = 3; x_data_tmp < 5; x_data_tmp++) {
          if (b0) {
            b0 = false;
            loop_ub = (((x_data_tmp - 1) % 2) << 1) + (x_data_tmp - 1) / 2;
          } else {
            loop_ub += 2;
            if (loop_ub > 3) {
              loop_ub -= 3;
            }
          }

          c[loop_ub] = 0.0F;
        }

        ar = 0;
        b0 = true;
        loop_ub = 0;
        for (x_data_tmp = 1; x_data_tmp < 3; x_data_tmp++) {
          if (b0) {
            b0 = false;
            loop_ub = (((x_data_tmp - 1) % 2) << 1) + (x_data_tmp - 1) / 2;
          } else {
            loop_ub += 2;
            if (loop_ub > 3) {
              loop_ub -= 3;
            }
          }

          temp = 0.0F;
          b1 = true;
          b2 = (x_size_idx_0 <= 0);
          i24 = x_size_idx_0 << 1;
          i25 = 0;
          b3 = true;
          b4 = (x_size_idx_0 <= 0);
          i26 = 0;
          for (w = 0; w <= m; w++) {
            if (b4 || (w >= i24)) {
              i26 = 0;
              b3 = true;
            } else if (b3) {
              b3 = false;
              i26 = ((w % x_size_idx_0) << 1) + w / x_size_idx_0;
            } else {
              i27 = (x_size_idx_0 << 1) - 1;
              if (i26 > 2147483645) {
                i26 = ((w % x_size_idx_0) << 1) + w / x_size_idx_0;
              } else {
                i26 += 2;
                if (i26 > i27) {
                  i26 -= i27;
                }
              }
            }

            i27 = w + ar;
            if (b2 || (i27 < 0) || (i27 >= i24)) {
              i25 = 0;
              b1 = true;
            } else if (b1) {
              b1 = false;
              i25 = ((i27 % x_size_idx_0) << 1) + i27 / x_size_idx_0;
            } else {
              i28 = (x_size_idx_0 << 1) - 1;
              if (i25 > 2147483645) {
                i25 = ((i27 % x_size_idx_0) << 1) + i27 / x_size_idx_0;
              } else {
                i25 += 2;
                if (i25 > i28) {
                  i25 -= i28;
                }
              }
            }

            temp += SD->u1.f0.x_data[i25] * SD->u1.f0.x_data[i26];
          }

          c[loop_ub] += muj_tmp * temp;
          ar += LDA;
        }

        ar = 0;
        b0 = true;
        loop_ub = 0;
        for (x_data_tmp = 3; x_data_tmp < 5; x_data_tmp++) {
          if (b0) {
            b0 = false;
            loop_ub = (((x_data_tmp - 1) % 2) << 1) + (x_data_tmp - 1) / 2;
          } else {
            loop_ub += 2;
            if (loop_ub > 3) {
              loop_ub -= 3;
            }
          }

          temp = 0.0F;
          b1 = true;
          b2 = (x_size_idx_0 <= 0);
          i24 = x_size_idx_0 << 1;
          i25 = 0;
          b3 = true;
          b4 = (x_size_idx_0 <= 0);
          i27 = x_size_idx_0 << 1;
          i26 = 0;
          for (w = 0; w <= m; w++) {
            i28 = w + LDA;
            if (b4 || (i28 < 0) || (i28 >= i27)) {
              i26 = 0;
              b3 = true;
            } else if (b3) {
              b3 = false;
              i26 = ((i28 % x_size_idx_0) << 1) + i28 / x_size_idx_0;
            } else {
              i29 = (x_size_idx_0 << 1) - 1;
              if (i26 > 2147483645) {
                i26 = ((i28 % x_size_idx_0) << 1) + i28 / x_size_idx_0;
              } else {
                i26 += 2;
                if (i26 > i29) {
                  i26 -= i29;
                }
              }
            }

            i28 = w + ar;
            if (b2 || (i28 < 0) || (i28 >= i24)) {
              i25 = 0;
              b1 = true;
            } else if (b1) {
              b1 = false;
              i25 = ((i28 % x_size_idx_0) << 1) + i28 / x_size_idx_0;
            } else {
              i29 = (x_size_idx_0 << 1) - 1;
              if (i25 > 2147483645) {
                i25 = ((i28 % x_size_idx_0) << 1) + i28 / x_size_idx_0;
              } else {
                i25 += 2;
                if (i25 > i29) {
                  i25 -= i29;
                }
              }
            }

            temp += SD->u1.f0.x_data[i25] * SD->u1.f0.x_data[i26];
          }

          c[loop_ub] += muj_tmp * temp;
          ar += LDA;
        }
      }
    }

    loop_ub = 2;
    covMatrixOutputInfInf_data[0] = c[0];
    covMatrixOutputInfInf_data[1] = c[1];
    covMatrixOutputInfInf_data[2] = c[2];
    covMatrixOutputInfInf_data[3] = c[3];
  }

  // 'SmartLoaderCalcEigen:7' covMatrix = cast(covMatrixOutputInfInf(1:2,1:2), 'double'); 
  // 'SmartLoaderCalcEigen:8' [V,D] = eig(covMatrix);
  covMatrixOutputInfInf[0] = covMatrixOutputInfInf_data[0];
  covMatrixOutputInfInf[1] = covMatrixOutputInfInf_data[loop_ub];
  covMatrixOutputInfInf[2] = covMatrixOutputInfInf_data[1];
  covMatrixOutputInfInf[3] = covMatrixOutputInfInf_data[1 + loop_ub];
  eig(covMatrixOutputInfInf, dcv0, dcv1);
  D[0] = dcv1[0];
  V[0] = dcv0[0];
  D[1] = dcv1[2];
  V[1] = dcv0[2];
  D[2] = dcv1[1];
  V[2] = dcv0[1];
  D[3] = dcv1[3];
  V[3] = dcv0[3];
}

//
// function [smartLoaderStruct, heightMap_res] = SmartLoaderCreateHeightMap(smartLoaderStruct, configParams, xyz)
// Arguments    : PerceptionSmartLoaderStackData *SD
//                PerceptionSmartLoaderStruct *smartLoaderStruct
//                const double configParams_xyzLimits[6]
//                double configParams_heightMapResolutionMeterToPixel
//                const float xyz_data[]
//                const int xyz_size[2]
//                float heightMap_res_data[]
//                int heightMap_res_size[2]
// Return Type  : void
//
static void SmartLoaderCreateHeightMap(PerceptionSmartLoaderStackData *SD,
  PerceptionSmartLoaderStruct *smartLoaderStruct, const double
  configParams_xyzLimits[6], double configParams_heightMapResolutionMeterToPixel,
  const float xyz_data[], const int xyz_size[2], float heightMap_res_data[], int
  heightMap_res_size[2])
{
  double imgDims[2];
  boolean_T b_imgDims[2];
  emxArray_int32_T *colInd;
  emxArray_int32_T *y;
  emxArray_int32_T *v1;
  emxArray_int32_T *vk;
  int i32;
  int i33;
  int xyRounded_data_tmp;
  int nb;
  int loop_ub;
  boolean_T empty_non_axis_sizes;
  boolean_T b5;
  int hoistedGlobal_size[2];
  int k0;
  int input_sizes_idx_1;
  int trueCount;
  int result;
  cell_wrap_4 reshapes[2];
  signed char sizes_idx_1;
  emxArray_real32_T *b_result;
  int xyzInsideImg_size[2];
  int col_size[2];
  int col_data[3];
  int imgLinearInd_size[1];
  int xyRounded_size[2];
  int idx_size[1];
  emxArray_real_T *missingPixels;
  int xyzSortedUnique_size[2];
  emxArray_real_T *Idx;
  boolean_T b6;
  boolean_T b7;

  // 'SmartLoaderCreateHeightMap:3' coder.varsize('heightMap_res', [SmartLoaderCompilationConstants.HeightMapMaxDimSize SmartLoaderCompilationConstants.HeightMapMaxDimSize], [1 1]); 
  //  Create the height image
  // 'SmartLoaderCreateHeightMap:7' imgDims = ceil((configParams.xyzLimits(1:2,2) - configParams.xyzLimits(1:2,1)) / configParams.heightMapResolutionMeterToPixel); 
  imgDims[0] = (configParams_xyzLimits[1] - configParams_xyzLimits[0]) /
    configParams_heightMapResolutionMeterToPixel;
  imgDims[1] = (configParams_xyzLimits[3] - configParams_xyzLimits[2]) /
    configParams_heightMapResolutionMeterToPixel;
  b_ceil(imgDims);

  // 'SmartLoaderCreateHeightMap:9' if any(int32(imgDims') > SmartLoaderCompilationConstants.HeightMapMaxDimSize) 
  b_imgDims[0] = ((int)rt_roundd(imgDims[0]) > 2048);
  b_imgDims[1] = ((int)rt_roundd(imgDims[1]) > 2048);
  if (any(b_imgDims)) {
    // 'SmartLoaderCreateHeightMap:10' heightMap_res = zeros(0,0,'single');
    heightMap_res_size[1] = 0;
    heightMap_res_size[0] = 0;
  } else {
    //  Note the dim flip
    // 'SmartLoaderCreateHeightMap:16' if isempty(heightMap_resPersistent)
    emxInit_int32_T(&colInd, 2);
    emxInit_int32_T(&y, 2);
    emxInit_int32_T(&v1, 2);
    emxInit_int32_T(&vk, 2);
    if (!SD->pd->heightMap_resPersistent_not_empty) {
      // 'SmartLoaderCreateHeightMap:17' heightMap_resPersistent = ones([imgDims(2) imgDims(1)],'single')*PcClassificationCompilationConstants.MaxMapInvalidValue; 
      i32 = SD->pd->heightMap_resPersistent->size[0] * SD->
        pd->heightMap_resPersistent->size[1];
      i33 = (int)imgDims[0];
      SD->pd->heightMap_resPersistent->size[1] = i33;
      xyRounded_data_tmp = (int)imgDims[1];
      SD->pd->heightMap_resPersistent->size[0] = xyRounded_data_tmp;
      emxEnsureCapacity_real32_T(SD->pd->heightMap_resPersistent, i32);
      nb = i33 * xyRounded_data_tmp;
      for (i32 = 0; i32 < nb; i32++) {
        SD->pd->heightMap_resPersistent->data[i32] = -1024.0F;
      }

      SD->pd->heightMap_resPersistent_not_empty = ((SD->
        pd->heightMap_resPersistent->size[0] != 0) && (SD->
        pd->heightMap_resPersistent->size[1] != 0));

      // 'SmartLoaderCreateHeightMap:19' [rowInd, colInd] = ind2sub(size(heightMap_resPersistent), 1:numel(heightMap_resPersistent)); 
      nb = SD->pd->heightMap_resPersistent->size[1];
      hoistedGlobal_size[1] = SD->pd->heightMap_resPersistent->size[0];
      k0 = SD->pd->heightMap_resPersistent->size[0];
      nb *= k0;
      if (nb < 1) {
        y->size[1] = 0;
        y->size[0] = 1;
      } else {
        i32 = y->size[0] * y->size[1];
        y->size[1] = nb;
        y->size[0] = 1;
        emxEnsureCapacity_int32_T(y, i32);
        for (i32 = 0; i32 < nb; i32++) {
          y->data[i32] = 1 + i32;
        }
      }

      k0 = hoistedGlobal_size[1];
      i32 = v1->size[0] * v1->size[1];
      v1->size[1] = y->size[1];
      v1->size[0] = 1;
      emxEnsureCapacity_int32_T(v1, i32);
      loop_ub = y->size[1] * y->size[0];
      for (i32 = 0; i32 < loop_ub; i32++) {
        v1->data[i32] = y->data[i32] - 1;
      }

      i32 = vk->size[0] * vk->size[1];
      vk->size[1] = v1->size[1];
      vk->size[0] = 1;
      emxEnsureCapacity_int32_T(vk, i32);
      loop_ub = v1->size[1] * v1->size[0];
      for (i32 = 0; i32 < loop_ub; i32++) {
        vk->data[i32] = div_s32(v1->data[i32], k0);
      }

      i32 = v1->size[1] * v1->size[0];
      i33 = v1->size[0] * v1->size[1];
      v1->size[0] = 1;
      emxEnsureCapacity_int32_T(v1, i33);
      loop_ub = i32 - 1;
      for (i32 = 0; i32 <= loop_ub; i32++) {
        v1->data[i32] -= vk->data[i32] * k0;
      }

      i32 = y->size[0] * y->size[1];
      y->size[1] = v1->size[1];
      y->size[0] = 1;
      emxEnsureCapacity_int32_T(y, i32);
      loop_ub = v1->size[1] * v1->size[0];
      for (i32 = 0; i32 < loop_ub; i32++) {
        y->data[i32] = v1->data[i32] + 1;
      }

      i32 = colInd->size[0] * colInd->size[1];
      colInd->size[1] = vk->size[1];
      colInd->size[0] = 1;
      emxEnsureCapacity_int32_T(colInd, i32);
      loop_ub = vk->size[1] * vk->size[0];
      for (i32 = 0; i32 < loop_ub; i32++) {
        colInd->data[i32] = vk->data[i32] + 1;
      }

      // 'SmartLoaderCreateHeightMap:20' allImgInds = [colInd', rowInd'];
      nb = colInd->size[1];
      k0 = y->size[1];
      i32 = SD->pd->allImgInds->size[0] * SD->pd->allImgInds->size[1];
      SD->pd->allImgInds->size[1] = 2;
      SD->pd->allImgInds->size[0] = nb;
      emxEnsureCapacity_real_T(SD->pd->allImgInds, i32);
      for (i32 = 0; i32 < nb; i32++) {
        SD->pd->allImgInds->data[i32 << 1] = colInd->data[i32];
      }

      for (i32 = 0; i32 < k0; i32++) {
        SD->pd->allImgInds->data[1 + (i32 << 1)] = y->data[i32];
      }
    }

    //  Use the next condition statment in order to avoid
    // 'SmartLoaderCreateHeightMap:23' if ~isequal(size(heightMap_resPersistent),imgDims) 
    // 'SmartLoaderCreateHeightMap:24' heightMap_resPersistent = ones([imgDims(2) imgDims(1)],'single')*PcClassificationCompilationConstants.MaxMapInvalidValue; 
    i32 = SD->pd->heightMap_resPersistent->size[0] * SD->
      pd->heightMap_resPersistent->size[1];
    SD->pd->heightMap_resPersistent->size[1] = (int)imgDims[0];
    SD->pd->heightMap_resPersistent->size[0] = (int)imgDims[1];
    emxEnsureCapacity_real32_T(SD->pd->heightMap_resPersistent, i32);
    loop_ub = (int)imgDims[0] * (int)imgDims[1];
    for (i32 = 0; i32 < loop_ub; i32++) {
      SD->pd->heightMap_resPersistent->data[i32] = -1024.0F;
    }

    empty_non_axis_sizes = (SD->pd->heightMap_resPersistent->size[0] == 0);
    b5 = (SD->pd->heightMap_resPersistent->size[1] == 0);
    SD->pd->heightMap_resPersistent_not_empty = ((!empty_non_axis_sizes) && (!b5));

    // 'SmartLoaderCreateHeightMap:26' [rowInd, colInd] = ind2sub(size(heightMap_resPersistent), 1:numel(heightMap_resPersistent)); 
    nb = SD->pd->heightMap_resPersistent->size[1];
    hoistedGlobal_size[1] = SD->pd->heightMap_resPersistent->size[0];
    k0 = SD->pd->heightMap_resPersistent->size[0];
    nb *= k0;
    if (nb < 1) {
      y->size[1] = 0;
      y->size[0] = 1;
    } else {
      i32 = y->size[0] * y->size[1];
      y->size[1] = nb;
      y->size[0] = 1;
      emxEnsureCapacity_int32_T(y, i32);
      for (i32 = 0; i32 < nb; i32++) {
        y->data[i32] = 1 + i32;
      }
    }

    k0 = hoistedGlobal_size[1];
    i32 = v1->size[0] * v1->size[1];
    v1->size[1] = y->size[1];
    v1->size[0] = 1;
    emxEnsureCapacity_int32_T(v1, i32);
    loop_ub = y->size[1] * y->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      v1->data[i32] = y->data[i32] - 1;
    }

    i32 = vk->size[0] * vk->size[1];
    vk->size[1] = v1->size[1];
    vk->size[0] = 1;
    emxEnsureCapacity_int32_T(vk, i32);
    loop_ub = v1->size[1] * v1->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      vk->data[i32] = div_s32(v1->data[i32], k0);
    }

    i32 = v1->size[1] * v1->size[0];
    i33 = v1->size[0] * v1->size[1];
    v1->size[0] = 1;
    emxEnsureCapacity_int32_T(v1, i33);
    loop_ub = i32 - 1;
    for (i32 = 0; i32 <= loop_ub; i32++) {
      v1->data[i32] -= vk->data[i32] * k0;
    }

    i32 = y->size[0] * y->size[1];
    y->size[1] = v1->size[1];
    y->size[0] = 1;
    emxEnsureCapacity_int32_T(y, i32);
    loop_ub = v1->size[1] * v1->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      y->data[i32] = v1->data[i32] + 1;
    }

    emxFree_int32_T(&v1);
    i32 = colInd->size[0] * colInd->size[1];
    colInd->size[1] = vk->size[1];
    colInd->size[0] = 1;
    emxEnsureCapacity_int32_T(colInd, i32);
    loop_ub = vk->size[1] * vk->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      colInd->data[i32] = vk->data[i32] + 1;
    }

    emxFree_int32_T(&vk);

    // 'SmartLoaderCreateHeightMap:27' allImgInds = [colInd', rowInd'];
    nb = colInd->size[1];
    k0 = y->size[1];
    i32 = SD->pd->allImgInds->size[0] * SD->pd->allImgInds->size[1];
    SD->pd->allImgInds->size[1] = 2;
    SD->pd->allImgInds->size[0] = nb;
    emxEnsureCapacity_real_T(SD->pd->allImgInds, i32);
    for (i32 = 0; i32 < nb; i32++) {
      SD->pd->allImgInds->data[i32 << 1] = colInd->data[i32];
    }

    emxFree_int32_T(&colInd);
    for (i32 = 0; i32 < k0; i32++) {
      SD->pd->allImgInds->data[1 + (i32 << 1)] = y->data[i32];
    }

    emxFree_int32_T(&y);

    // 'SmartLoaderCreateHeightMap:30' heightMap_res = heightMap_resPersistent;
    heightMap_res_size[1] = SD->pd->heightMap_resPersistent->size[1];
    heightMap_res_size[0] = SD->pd->heightMap_resPersistent->size[0];
    loop_ub = SD->pd->heightMap_resPersistent->size[1] * SD->
      pd->heightMap_resPersistent->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      heightMap_res_data[i32] = SD->pd->heightMap_resPersistent->data[i32];
    }

    //  Plus 1 because matlab indexes start from one
    //  Note: cast in matlab works like round commnad
    // 'SmartLoaderCreateHeightMap:34' xyRounded = round(cast(1, 'like', xyz) + xyz(:,1:2)/cast(configParams.heightMapResolutionMeterToPixel, 'like', xyz)); 
    loop_ub = xyz_size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      k0 = i32 << 1;
      SD->f22.xyRounded_data[k0] = 1.0F + xyz_data[3 * i32] / (float)
        configParams_heightMapResolutionMeterToPixel;
      SD->f22.xyRounded_data[1 + k0] = 1.0F + xyz_data[1 + 3 * i32] / (float)
        configParams_heightMapResolutionMeterToPixel;
    }

    i32 = xyz_size[0];
    for (input_sizes_idx_1 = 0; input_sizes_idx_1 < i32; input_sizes_idx_1++) {
      k0 = input_sizes_idx_1 << 1;
      SD->f22.xyRounded_data[k0] = rt_roundf(SD->f22.xyRounded_data[k0]);
      k0++;
      SD->f22.xyRounded_data[k0] = rt_roundf(SD->f22.xyRounded_data[k0]);
    }

    //  remove everything that is outside the image boundery
    // 'SmartLoaderCreateHeightMap:37' insideBounderyInd = xyRounded(:,1) > 0 & xyRounded(:,1) <= imgDims(1) & xyRounded(:,2) > 0 & xyRounded(:,2) <= imgDims(2); 
    loop_ub = xyz_size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      nb = i32 << 1;
      SD->f22.d_tmp_data[i32] = ((SD->f22.xyRounded_data[nb] > 0.0F) &&
        (SD->f22.xyRounded_data[nb] <= imgDims[0]) && (SD->f22.xyRounded_data[1
        + nb] > 0.0F));
    }

    loop_ub = xyz_size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      SD->f22.e_tmp_data[i32] = (SD->f22.xyRounded_data[1 + (i32 << 1)] <=
        imgDims[1]);
    }

    //  sum(removeInd)
    // 'SmartLoaderCreateHeightMap:39' xyInside = xyRounded(insideBounderyInd, :); 
    // 'SmartLoaderCreateHeightMap:40' zInside = xyz(insideBounderyInd, 3);
    // 'SmartLoaderCreateHeightMap:42' xyzInsideImg = [xyInside zInside];
    nb = xyz_size[0] - 1;
    trueCount = 0;
    for (input_sizes_idx_1 = 0; input_sizes_idx_1 <= nb; input_sizes_idx_1++) {
      if (SD->f22.d_tmp_data[input_sizes_idx_1] && SD->
          f22.e_tmp_data[input_sizes_idx_1]) {
        trueCount++;
      }
    }

    k0 = 0;
    for (input_sizes_idx_1 = 0; input_sizes_idx_1 <= nb; input_sizes_idx_1++) {
      if (SD->f22.d_tmp_data[input_sizes_idx_1] && SD->
          f22.e_tmp_data[input_sizes_idx_1]) {
        SD->f22.imgLinearInd_data[k0] = input_sizes_idx_1 + 1;
        k0++;
      }
    }

    if (trueCount != 0) {
      result = trueCount;
    } else {
      result = 0;
    }

    empty_non_axis_sizes = (result == 0);
    if (empty_non_axis_sizes || (trueCount != 0)) {
      input_sizes_idx_1 = 2;
    } else {
      input_sizes_idx_1 = 0;
    }

    hoistedGlobal_size[0] = result;
    hoistedGlobal_size[1] = input_sizes_idx_1;
    emxInitMatrix_cell_wrap_4(reshapes);
    if ((input_sizes_idx_1 == 2) && (result == trueCount)) {
      for (i32 = 0; i32 < trueCount; i32++) {
        k0 = (SD->f22.imgLinearInd_data[i32] - 1) << 1;
        xyRounded_data_tmp = i32 << 1;
        SD->f22.b_xyRounded_data[xyRounded_data_tmp] = SD->f22.xyRounded_data[k0];
        SD->f22.b_xyRounded_data[1 + xyRounded_data_tmp] =
          SD->f22.xyRounded_data[1 + k0];
      }

      i32 = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
      reshapes[0].f1->size[1] = input_sizes_idx_1;
      reshapes[0].f1->size[0] = result;
      emxEnsureCapacity_real32_T(reshapes[0].f1, i32);
      loop_ub = input_sizes_idx_1 * result;
      for (i32 = 0; i32 < loop_ub; i32++) {
        reshapes[0].f1->data[i32] = SD->f22.b_xyRounded_data[i32];
      }
    } else {
      i32 = 0;
      i33 = 0;
      xyRounded_data_tmp = 0;
      k0 = 0;
      nb = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
      reshapes[0].f1->size[1] = input_sizes_idx_1;
      reshapes[0].f1->size[0] = result;
      emxEnsureCapacity_real32_T(reshapes[0].f1, nb);
      for (nb = 0; nb < hoistedGlobal_size[0] * hoistedGlobal_size[1]; nb++) {
        reshapes[0].f1->data[i33 + reshapes[0].f1->size[1] * i32] =
          SD->f22.xyRounded_data[k0 + ((SD->
          f22.imgLinearInd_data[xyRounded_data_tmp] - 1) << 1)];
        i32++;
        xyRounded_data_tmp++;
        if (i32 > reshapes[0].f1->size[0] - 1) {
          i32 = 0;
          i33++;
        }

        if (xyRounded_data_tmp > trueCount - 1) {
          xyRounded_data_tmp = 0;
          k0++;
        }
      }
    }

    if (empty_non_axis_sizes || (trueCount != 0)) {
      sizes_idx_1 = 1;
    } else {
      sizes_idx_1 = 0;
    }

    input_sizes_idx_1 = sizes_idx_1;
    i32 = 0;
    i33 = 0;
    xyRounded_data_tmp = 0;
    k0 = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
    reshapes[1].f1->size[1] = sizes_idx_1;
    reshapes[1].f1->size[0] = result;
    emxEnsureCapacity_real32_T(reshapes[1].f1, k0);
    for (k0 = 0; k0 < result * input_sizes_idx_1; k0++) {
      reshapes[1].f1->data[i33 + reshapes[1].f1->size[1] * i32] = xyz_data[2 + 3
        * (SD->f22.imgLinearInd_data[xyRounded_data_tmp] - 1)];
      i32++;
      xyRounded_data_tmp++;
      if (i32 > reshapes[1].f1->size[0] - 1) {
        i32 = 0;
        i33++;
      }
    }

    emxInit_real32_T(&b_result, 2);
    i32 = b_result->size[0] * b_result->size[1];
    b_result->size[1] = reshapes[0].f1->size[1] + reshapes[1].f1->size[1];
    b_result->size[0] = reshapes[0].f1->size[0];
    emxEnsureCapacity_real32_T(b_result, i32);
    loop_ub = reshapes[0].f1->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      input_sizes_idx_1 = reshapes[0].f1->size[1];
      for (i33 = 0; i33 < input_sizes_idx_1; i33++) {
        b_result->data[i33 + b_result->size[1] * i32] = reshapes[0].f1->data[i33
          + reshapes[0].f1->size[1] * i32];
      }
    }

    loop_ub = reshapes[1].f1->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      input_sizes_idx_1 = reshapes[1].f1->size[1];
      for (i33 = 0; i33 < input_sizes_idx_1; i33++) {
        b_result->data[(i33 + reshapes[0].f1->size[1]) + b_result->size[1] * i32]
          = reshapes[1].f1->data[i33 + reshapes[1].f1->size[1] * i32];
      }
    }

    xyzInsideImg_size[1] = reshapes[0].f1->size[1] + reshapes[1].f1->size[1];
    xyzInsideImg_size[0] = reshapes[0].f1->size[0];
    loop_ub = reshapes[0].f1->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      input_sizes_idx_1 = reshapes[0].f1->size[1];
      for (i33 = 0; i33 < input_sizes_idx_1; i33++) {
        SD->f22.xyzInsideImg_data[i33 + xyzInsideImg_size[1] * i32] = reshapes[0]
          .f1->data[i33 + reshapes[0].f1->size[1] * i32];
      }
    }

    loop_ub = reshapes[1].f1->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      input_sizes_idx_1 = reshapes[1].f1->size[1];
      for (i33 = 0; i33 < input_sizes_idx_1; i33++) {
        SD->f22.xyzInsideImg_data[(i33 + reshapes[0].f1->size[1]) +
          xyzInsideImg_size[1] * i32] = reshapes[1].f1->data[i33 + reshapes[1].
          f1->size[1] * i32];
      }
    }

    emxFreeMatrix_cell_wrap_4(reshapes);

    // 'SmartLoaderCreateHeightMap:43' xyzSorted = sortrows(xyzInsideImg);
    nb = b_result->size[1];
    col_size[1] = (signed char)xyzInsideImg_size[1];
    col_size[0] = 1;
    for (input_sizes_idx_1 = 0; input_sizes_idx_1 < nb; input_sizes_idx_1++) {
      col_data[input_sizes_idx_1] = input_sizes_idx_1 + 1;
    }

    sortIdx(SD, SD->f22.xyzInsideImg_data, xyzInsideImg_size, col_data, col_size,
            SD->f22.imgLinearInd_data, imgLinearInd_size);
    xyzInsideImg_size[1] = b_result->size[1];
    xyzInsideImg_size[0] = b_result->size[0];
    loop_ub = b_result->size[1] * b_result->size[0];
    for (i32 = 0; i32 < loop_ub; i32++) {
      SD->f22.xyzInsideImg_data[i32] = b_result->data[i32];
    }

    emxFree_real32_T(&b_result);
    apply_row_permutation(SD, SD->f22.xyzInsideImg_data, xyzInsideImg_size,
                          SD->f22.imgLinearInd_data);

    // 'SmartLoaderCreateHeightMap:45' [~,ia,~] = unique(xyzSorted(:,1:2), 'rows', 'last'); 
    if (xyzInsideImg_size[0] == 0) {
      imgLinearInd_size[0] = 0;
    } else {
      loop_ub = xyzInsideImg_size[0];
      xyRounded_size[1] = 2;
      xyRounded_size[0] = xyzInsideImg_size[0];
      for (i32 = 0; i32 < loop_ub; i32++) {
        k0 = xyzInsideImg_size[1] * i32;
        xyRounded_data_tmp = i32 << 1;
        SD->f22.xyRounded_data[xyRounded_data_tmp] = SD->
          f22.xyzInsideImg_data[k0];
        SD->f22.xyRounded_data[1 + xyRounded_data_tmp] =
          SD->f22.xyzInsideImg_data[1 + k0];
      }

      sortrows(SD, SD->f22.xyRounded_data, xyRounded_size, SD->f22.idx_data,
               idx_size);
      nb = -1;
      i32 = xyzInsideImg_size[0];
      input_sizes_idx_1 = 1;
      while (input_sizes_idx_1 <= i32) {
        k0 = input_sizes_idx_1;
        do {
          input_sizes_idx_1++;
        } while (!((input_sizes_idx_1 > i32) || rows_differ
                   (SD->f22.xyRounded_data, k0, input_sizes_idx_1)));

        nb++;
        k0 = (k0 - 1) << 1;
        xyRounded_data_tmp = nb << 1;
        SD->f22.xyRounded_data[xyRounded_data_tmp] = SD->f22.xyRounded_data[k0];
        SD->f22.xyRounded_data[1 + xyRounded_data_tmp] = SD->f22.xyRounded_data
          [1 + k0];
        SD->f22.idx_data[nb] = SD->f22.idx_data[input_sizes_idx_1 - 2];
      }

      imgLinearInd_size[0] = nb + 1;
      for (input_sizes_idx_1 = 0; input_sizes_idx_1 <= nb; input_sizes_idx_1++)
      {
        SD->f22.imgLinearInd_data[input_sizes_idx_1] = (int)SD->
          f22.idx_data[input_sizes_idx_1];
      }
    }

    // 'SmartLoaderCreateHeightMap:45' ~
    // 'SmartLoaderCreateHeightMap:45' ~
    // 'SmartLoaderCreateHeightMap:47' xyzSortedUnique = xyzSorted(ia,:);
    loop_ub = xyzInsideImg_size[1];
    input_sizes_idx_1 = imgLinearInd_size[0];
    for (i32 = 0; i32 < input_sizes_idx_1; i32++) {
      for (i33 = 0; i33 < loop_ub; i33++) {
        SD->f22.xyzSortedUnique_data[i33 + loop_ub * i32] =
          SD->f22.xyzInsideImg_data[i33 + xyzInsideImg_size[1] *
          (SD->f22.imgLinearInd_data[i32] - 1)];
      }
    }

    // 'SmartLoaderCreateHeightMap:48' zSortedUnique = xyzSortedUnique(:,3);
    input_sizes_idx_1 = imgLinearInd_size[0];
    for (i32 = 0; i32 < input_sizes_idx_1; i32++) {
      SD->f22.b_tmp_data[i32] = SD->f22.xyzSortedUnique_data[2 + loop_ub * i32];
    }

    // 'SmartLoaderCreateHeightMap:50' imgLinearInd = sub2ind(size(heightMap_res), xyzSortedUnique(:,2), xyzSortedUnique(:,1)); 
    input_sizes_idx_1 = imgLinearInd_size[0];
    for (i32 = 0; i32 < input_sizes_idx_1; i32++) {
      nb = xyzInsideImg_size[1] * i32;
      SD->f22.imgLinearInd_data[i32] = (int)SD->f22.xyzSortedUnique_data[1 + nb]
        + (short)heightMap_res_size[0] * ((int)SD->f22.xyzSortedUnique_data[nb]
        - 1);
    }

    // 'SmartLoaderCreateHeightMap:52' heightMap_res(imgLinearInd) = zSortedUnique; 
    input_sizes_idx_1 = imgLinearInd_size[0];
    for (i32 = 0; i32 < input_sizes_idx_1; i32++) {
      SD->f22.c_tmp_data[i32] = SD->f22.imgLinearInd_data[i32] - 1;
    }

    i32 = heightMap_res_size[0];
    input_sizes_idx_1 = imgLinearInd_size[0] - 1;
    for (i33 = 0; i33 <= input_sizes_idx_1; i33++) {
      heightMap_res_data[SD->f22.c_tmp_data[i33] % i32 * heightMap_res_size[1] +
        SD->f22.c_tmp_data[i33] / i32] = SD->f22.b_tmp_data[i33];
    }

    // f1 = figure, PointCloudClassificationPlottingUtility.PlotMaxMap(heightMap_res); 
    //  The easiest thing to do - is apply knn search on the data
    // 'SmartLoaderCreateHeightMap:57' missingPixelsLogical = ones(numel(heightMap_res),1,'logical'); 
    nb = heightMap_res_size[0] * heightMap_res_size[1];
    for (i32 = 0; i32 < nb; i32++) {
      SD->f22.missingPixelsLogical_data[i32] = true;
    }

    // 'SmartLoaderCreateHeightMap:58' missingPixelsLogical(imgLinearInd) = false; 
    input_sizes_idx_1 = imgLinearInd_size[0];
    for (i32 = 0; i32 < input_sizes_idx_1; i32++) {
      SD->f22.missingPixelsLogical_data[SD->f22.imgLinearInd_data[i32] - 1] =
        false;
    }

    // 'SmartLoaderCreateHeightMap:59' missingPixels = allImgInds(missingPixelsLogical,:); 
    nb--;
    trueCount = 0;
    for (input_sizes_idx_1 = 0; input_sizes_idx_1 <= nb; input_sizes_idx_1++) {
      if (SD->f22.missingPixelsLogical_data[input_sizes_idx_1]) {
        trueCount++;
      }
    }

    k0 = 0;
    for (input_sizes_idx_1 = 0; input_sizes_idx_1 <= nb; input_sizes_idx_1++) {
      if (SD->f22.missingPixelsLogical_data[input_sizes_idx_1]) {
        SD->f22.tmp_data[k0] = input_sizes_idx_1 + 1;
        k0++;
      }
    }

    emxInit_real_T(&missingPixels, 2);
    i32 = missingPixels->size[0] * missingPixels->size[1];
    missingPixels->size[1] = 2;
    missingPixels->size[0] = trueCount;
    emxEnsureCapacity_real_T(missingPixels, i32);
    for (i32 = 0; i32 < trueCount; i32++) {
      i33 = (SD->f22.tmp_data[i32] - 1) << 1;
      xyRounded_data_tmp = i32 << 1;
      missingPixels->data[xyRounded_data_tmp] = SD->pd->allImgInds->data[i33];
      missingPixels->data[1 + xyRounded_data_tmp] = SD->pd->allImgInds->data[1 +
        i33];
    }

    // 'SmartLoaderCreateHeightMap:61' Idx = knnsearch(xyzSortedUnique(:,1:2), missingPixels); 
    input_sizes_idx_1 = imgLinearInd_size[0];
    xyzSortedUnique_size[1] = 2;
    xyzSortedUnique_size[0] = imgLinearInd_size[0];
    for (i32 = 0; i32 < input_sizes_idx_1; i32++) {
      nb = loop_ub * i32;
      k0 = i32 << 1;
      SD->f22.xyRounded_data[k0] = SD->f22.xyzSortedUnique_data[nb];
      SD->f22.xyRounded_data[1 + k0] = SD->f22.xyzSortedUnique_data[1 + nb];
    }

    emxInit_real_T(&Idx, 2);
    knnsearch(SD, SD->f22.xyRounded_data, xyzSortedUnique_size, missingPixels,
              Idx);

    // 'SmartLoaderCreateHeightMap:62' for i = 1:size(missingPixels,1)
    i32 = missingPixels->size[0];
    b6 = true;
    b7 = ((Idx->size[1] <= 0) || (Idx->size[0] <= 0));
    i33 = Idx->size[1] * Idx->size[0];
    xyRounded_data_tmp = 0;
    for (input_sizes_idx_1 = 0; input_sizes_idx_1 < i32; input_sizes_idx_1++) {
      if (b7 || (input_sizes_idx_1 >= i33)) {
        xyRounded_data_tmp = 0;
        b6 = true;
      } else if (b6) {
        b6 = false;
        xyRounded_data_tmp = Idx->size[1];
        k0 = Idx->size[0];
        xyRounded_data_tmp = input_sizes_idx_1 % k0 * xyRounded_data_tmp +
          input_sizes_idx_1 / k0;
      } else {
        k0 = Idx->size[1];
        nb = k0 * Idx->size[0] - 1;
        if (xyRounded_data_tmp > MAX_int32_T - k0) {
          xyRounded_data_tmp = Idx->size[1];
          k0 = Idx->size[0];
          xyRounded_data_tmp = input_sizes_idx_1 % k0 * xyRounded_data_tmp +
            input_sizes_idx_1 / k0;
        } else {
          xyRounded_data_tmp += k0;
          if (xyRounded_data_tmp > nb) {
            xyRounded_data_tmp -= nb;
          }
        }
      }

      // 'SmartLoaderCreateHeightMap:63' heightMap_res(missingPixels(i,2), missingPixels(i,1)) = zSortedUnique(Idx(i)); 
      nb = input_sizes_idx_1 << 1;
      heightMap_res_data[((int)missingPixels->data[nb] + heightMap_res_size[1] *
                          ((int)missingPixels->data[1 + nb] - 1)) - 1] =
        SD->f22.b_tmp_data[(int)Idx->data[xyRounded_data_tmp] - 1];
    }

    emxFree_real_T(&Idx);
    emxFree_real_T(&missingPixels);

    // 'SmartLoaderCreateHeightMap:66' if false
    // 'SmartLoaderCreateHeightMap:72' smartLoaderStruct.heightMapStatus = true; 
    smartLoaderStruct->heightMapStatus = true;

    // 'SmartLoaderCreateHeightMap:74' if false && coder.target('Matlab')
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
// Return Type  : void
//
static void SmartLoaderCreateHeightMap_free(PerceptionSmartLoaderStackData *SD)
{
  emxFree_real32_T(&SD->pd->heightMap_resPersistent);
  emxFree_real_T(&SD->pd->allImgInds);
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
// Return Type  : void
//
static void SmartLoaderCreateHeightMap_init(PerceptionSmartLoaderStackData *SD)
{
  emxInit_real_T(&SD->pd->allImgInds, 2);
  emxInit_real32_T(&SD->pd->heightMap_resPersistent, 2);
  SD->pd->heightMap_resPersistent_not_empty = false;
}

//
// function [smartLoaderStruct] = SmartLoaderEstiamteLocations(smartLoaderStruct, configParams, ptCloudSenceXyz, ptCloudSenceIntensity)
// Arguments    : PerceptionSmartLoaderStackData *SD
//                PerceptionSmartLoaderStruct *smartLoaderStruct
//                const PerceptionSmartLoaderConfigParam *configParams
//                const float ptCloudSenceXyz_data[]
//                const int ptCloudSenceXyz_size[2]
//                const float ptCloudSenceIntensity_data[]
//                const int ptCloudSenceIntensity_size[1]
// Return Type  : void
//
static void SmartLoaderEstiamteLocations(PerceptionSmartLoaderStackData *SD,
  PerceptionSmartLoaderStruct *smartLoaderStruct, const
  PerceptionSmartLoaderConfigParam *configParams, const float
  ptCloudSenceXyz_data[], const int ptCloudSenceXyz_size[2], const float
  ptCloudSenceIntensity_data[], const int ptCloudSenceIntensity_size[1])
{
  int loop_ub;
  int i46;
  int m;
  int trueCount;
  int partialTrueCount;
  int ptCloudSenceReflectorsXyz_size[2];
  double singleReflectorRangeLimitMeter;
  float r[2];
  float distanceToKmeansClusterMeter[2];
  float pcFirstDistanceToPlane;
  double v_idx_0;
  boolean_T b_r[2];
  emxArray_real32_T *pdistOutput;
  emxArray_real32_T *Z;
  emxArray_cell_wrap_4_64x1 clustersXs;
  emxArray_cell_wrap_4_64x1 clustersYs;
  emxArray_cell_wrap_4_64x1 r2;
  cell_wrap_4 reshapes[2];
  emxArray_real32_T *varargin_1;
  cell_wrap_4 b_reshapes[2];
  cell_wrap_4 c_reshapes[2];
  emxArray_real32_T *d_reshapes;
  boolean_T guard1 = false;
  int kmeansIdx_size[1];
  float kmeansC[6];
  int input_sizes[2];
  float b_kmeansC[3];
  int iv1[1];
  float pcFirstRange_idx_0;
  int b_ptCloudSenceReflectorsXyz_size[2];
  int tmp_size[1];
  int x_size[1];
  int pcFirstXyz_size[2];
  float pcFirstRange_idx_1;
  int c_ptCloudSenceReflectorsXyz_size[2];
  int d_ptCloudSenceReflectorsXyz_size[2];
  boolean_T empty_non_axis_sizes;
  float pcSecondRange_idx_0;
  float pcSecondRange_idx_1;
  boolean_T isInvalid_data[64];
  int extremePoints_size[2];
  int q;
  int ptCloudShovelReflectorsXyz_size[2];
  float extremePoints_data[128];
  boolean_T isFoundLoaderPc;
  float us_data[64];
  int i47;
  int i48;
  int i49;
  float vs_data[64];
  int temp_usus_size[1];
  boolean_T guard2 = false;
  unsigned long u0;
  unsigned long u1;
  float temp_usus_data[64];
  signed char b_input_sizes;
  unsigned long u2;
  int temp_vsvs_size[1];
  float c_kmeansC[6];
  double v_idx_1;
  double d5;
  float temp_vsvs_data[64];
  float b_pcSecondRange_idx_0;
  float pcSecondDistanceToPlane;
  int temp_vsus_size[1];
  float c_pcSecondRange_idx_0;
  float b_pcFirstRange_idx_0;
  float temp_vsus_data[64];
  float c_pcFirstRange_idx_0;
  float fv0[4];
  float fv1[4];
  int b_temp_usus_size[1];
  int b_temp_vsus_size[1];
  float b_temp_usus_data[64];
  int vs_size[1];
  float b_temp_vsus_data[64];
  int c_temp_vsus_size[1];
  float b_vs_data[64];
  int vk;
  int modelErr1_size[2];
  int tempXs_size;
  float modelErr1_data[128];
  float tempXs_data[2];
  signed char sizes;
  int us_size[1];
  int b_us_size[1];
  int extremePoints_size_idx_1;
  int extremePoints_size_idx_0;
  float b_extremePoints_data[130];
  int b_ptCloudSenceXyz_size[2];
  int b_pcFirstXyz_size[2];
  creal_T V[4];
  creal_T D[4];
  creal_T largestEigenVec[2];
  double p4[2];
  creal_T b_V[2];
  double d6;
  double a_tmp;
  double loaderToLeftBackOftheLoaderDistanceMeter;
  double d7;
  double d8;
  double d9;
  double d10;
  double d11;
  double d12;
  int c_pcFirstXyz_size[1];
  double p1[4];
  double b_p1[4];
  int ptCloudSenceReflectorsInd_size[1];
  int d_pcFirstXyz_size[2];

  //  Estimate loader by circled reflector
  // 'SmartLoaderEstiamteLocations:6' ptCloudSenceReflectorsInd = ptCloudSenceIntensity > configParams.minimumIntensityReflectorValue; 
  loop_ub = ptCloudSenceIntensity_size[0];
  for (i46 = 0; i46 < loop_ub; i46++) {
    SD->u6.f20.ptCloudSenceReflectorsInd_data[i46] =
      (ptCloudSenceIntensity_data[i46] >
       configParams->minimumIntensityReflectorValue);
  }

  // 'SmartLoaderEstiamteLocations:7' ptCloudSenceReflectorsXyz = ptCloudSenceXyz(ptCloudSenceReflectorsInd,:); 
  m = ptCloudSenceIntensity_size[0] - 1;
  trueCount = 0;
  for (loop_ub = 0; loop_ub <= m; loop_ub++) {
    if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]) {
      trueCount++;
    }
  }

  partialTrueCount = 0;
  for (loop_ub = 0; loop_ub <= m; loop_ub++) {
    if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]) {
      SD->u6.f20.iidx_data[partialTrueCount] = loop_ub + 1;
      partialTrueCount++;
    }
  }

  ptCloudSenceReflectorsXyz_size[1] = 3;
  ptCloudSenceReflectorsXyz_size[0] = trueCount;
  for (i46 = 0; i46 < trueCount; i46++) {
    m = 3 * (SD->u6.f20.iidx_data[i46] - 1);
    SD->u6.f20.ptCloudSenceReflectorsXyz_data[3 * i46] = ptCloudSenceXyz_data[m];
    SD->u6.f20.ptCloudSenceReflectorsXyz_data[1 + 3 * i46] =
      ptCloudSenceXyz_data[1 + m];
    SD->u6.f20.ptCloudSenceReflectorsXyz_data[2 + 3 * i46] =
      ptCloudSenceXyz_data[2 + m];
  }

  // 'SmartLoaderEstiamteLocations:8' ptCloudSenceReflectorsIntensity = ptCloudSenceIntensity(ptCloudSenceReflectorsInd,:); 
  //  figure, PlotPointCloud(ptCloudSenceReflectorsXyz);
  //  figure, PlotPointCloud([ptCloudSenceReflectorsXyz ptCloudSenceReflectorsIntensity]); 
  // 'SmartLoaderEstiamteLocations:12' if size(ptCloudSenceReflectorsXyz,1) < configParams.minPointsForReflector 
  if (trueCount < configParams->minPointsForReflector) {
    // 'SmartLoaderEstiamteLocations:13' if coder.target('Matlab')
    // 'SmartLoaderEstiamteLocations:16' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eFailedNotEnoughReflectorPoints; 
    smartLoaderStruct->status =
      PerceptionSmartLoaderReturnValue_eFailedNotEnoughReflectorPoints;
  } else {
    //  Determine whether or not the reflectors are good enough for estimating two representing ellipses 
    //  first we have to determine whehter the reflector reside on the (loader) or (loader and shovel) 
    //  TODO - align the points according to the major and minor axis !!!
    //  I arbitrary set twich the reflector range as a limit - this number determine whether the sensor detected both the shovel and the loader. 
    // 'SmartLoaderEstiamteLocations:24' singleReflectorRangeLimitMeter = 2 * configParams.loaderReflectorDiameterMeter; 
    singleReflectorRangeLimitMeter = 2.0 *
      configParams->loaderReflectorDiameterMeter;

    // 'SmartLoaderEstiamteLocations:26' r = RangeCompiledVersion(ptCloudSenceReflectorsXyz(:,1:2)); 
    // 'RangeCompiledVersion:4' coder.inline('always');
    // 'RangeCompiledVersion:7' output = max(input) - min(input);
    r[0] = SD->u6.f20.ptCloudSenceReflectorsXyz_data[0];
    r[1] = SD->u6.f20.ptCloudSenceReflectorsXyz_data[1];
    for (loop_ub = 2; loop_ub <= trueCount; loop_ub++) {
      i46 = 3 * (loop_ub - 1);
      if (r[0] < SD->u6.f20.ptCloudSenceReflectorsXyz_data[i46]) {
        r[0] = SD->u6.f20.ptCloudSenceReflectorsXyz_data[i46];
      }

      pcFirstDistanceToPlane = SD->u6.f20.ptCloudSenceReflectorsXyz_data[1 + i46];
      if (r[1] < pcFirstDistanceToPlane) {
        r[1] = pcFirstDistanceToPlane;
      }
    }

    distanceToKmeansClusterMeter[0] = SD->u6.f20.ptCloudSenceReflectorsXyz_data
      [0];
    distanceToKmeansClusterMeter[1] = SD->u6.f20.ptCloudSenceReflectorsXyz_data
      [1];
    for (loop_ub = 2; loop_ub <= trueCount; loop_ub++) {
      i46 = 3 * (loop_ub - 1);
      if (distanceToKmeansClusterMeter[0] >
          SD->u6.f20.ptCloudSenceReflectorsXyz_data[i46]) {
        distanceToKmeansClusterMeter[0] =
          SD->u6.f20.ptCloudSenceReflectorsXyz_data[i46];
      }

      pcFirstDistanceToPlane = SD->u6.f20.ptCloudSenceReflectorsXyz_data[1 + i46];
      if (distanceToKmeansClusterMeter[1] > pcFirstDistanceToPlane) {
        distanceToKmeansClusterMeter[1] = pcFirstDistanceToPlane;
      }
    }

    // 'SmartLoaderEstiamteLocations:28' if any(r > (configParams.loaderHeightMeter + configParams.locationsBiasMeter * 10)) 
    v_idx_0 = configParams->loaderHeightMeter + configParams->locationsBiasMeter
      * 10.0;
    pcFirstDistanceToPlane = r[0] - distanceToKmeansClusterMeter[0];
    b_r[0] = (pcFirstDistanceToPlane > v_idx_0);
    r[0] = pcFirstDistanceToPlane;
    pcFirstDistanceToPlane = r[1] - distanceToKmeansClusterMeter[1];
    b_r[1] = (pcFirstDistanceToPlane > v_idx_0);
    if (any(b_r)) {
      //  Reflector size is too long then I expected
      // 'SmartLoaderEstiamteLocations:30' if coder.target('Matlab')
      // 'SmartLoaderEstiamteLocations:33' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eFailed; 
      smartLoaderStruct->status = PerceptionSmartLoaderReturnValue_eFailed;
    } else {
      // 'SmartLoaderEstiamteLocations:37' if all(r < singleReflectorRangeLimitMeter) 
      b_r[0] = (r[0] < singleReflectorRangeLimitMeter);
      b_r[1] = (pcFirstDistanceToPlane < singleReflectorRangeLimitMeter);
      emxInit_real32_T(&pdistOutput, 2);
      emxInit_real32_T(&Z, 2);
      emxInit_cell_wrap_4_64x1(&clustersXs);
      emxInit_cell_wrap_4_64x1(&clustersYs);
      emxInit_cell_wrap_4_64x1(&r2);
      emxInitMatrix_cell_wrap_4(reshapes);
      emxInit_real32_T(&varargin_1, 1);
      emxInitMatrix_cell_wrap_4(b_reshapes);
      emxInitMatrix_cell_wrap_4(c_reshapes);
      emxInit_real32_T(&d_reshapes, 2);
      guard1 = false;
      if (all(b_r)) {
        //     %%
        //  We assume the reflectors hold only the loader!
        // 'SmartLoaderEstiamteLocations:40' if coder.target('Matlab')
        //  TODO - handle the reflectors here !!!
        // ptCloudLoaderReflectors = ptCloudSenceReflectors;
        // 'SmartLoaderEstiamteLocations:45' ptCloudLoaderReflectorsXyz = ptCloudSenceReflectorsXyz; 
        // 'SmartLoaderEstiamteLocations:46' ptCloudLoaderReflectorsIntensity = ptCloudSenceReflectorsIntensity; 
        //  figure, PlotPointCloud(ptCloudSenceReflectors);
        //  figure, PlotPointCloud(ptCloudLoaderReflectors);
        guard1 = true;
      } else {
        // 'SmartLoaderEstiamteLocations:50' else
        //  Found both shovel and loader reflectors
        // 'SmartLoaderEstiamteLocations:52' [kmeansIdx,kmeansC,kmeanssumd,kmeansDistanceMat] = kmeans(ptCloudSenceReflectorsXyz, 2, 'Replicates', 5); 
        kmeans(SD, SD->u6.f20.ptCloudSenceReflectorsXyz_data,
               ptCloudSenceReflectorsXyz_size, SD->u6.f20.kmeansIdx_data,
               kmeansIdx_size, kmeansC, distanceToKmeansClusterMeter,
               SD->u6.f20.kmeansDistanceMat_data, input_sizes);

        // 'SmartLoaderEstiamteLocations:53' [~, minInd] = min(kmeansDistanceMat,[],1); 
        // 'SmartLoaderEstiamteLocations:53' ~
        // 'SmartLoaderEstiamteLocations:55' if (norm(kmeansC(1,:) - kmeansC(2,:))) > (configParams.loaderHeightMeter + configParams.locationsBiasMeter * 10) 
        b_kmeansC[0] = kmeansC[0] - kmeansC[3];
        b_kmeansC[1] = kmeansC[1] - kmeansC[4];
        b_kmeansC[2] = kmeansC[2] - kmeansC[5];
        if (b_norm(b_kmeansC) > v_idx_0) {
          //  Reflector size is too long then I expected
          // 'SmartLoaderEstiamteLocations:57' if coder.target('Matlab')
          // 'SmartLoaderEstiamteLocations:60' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eFailed; 
          smartLoaderStruct->status = PerceptionSmartLoaderReturnValue_eFailed;
        } else {
          // 'SmartLoaderEstiamteLocations:65' if coder.target('Matlab') && false 
          //  Get the first and the second point cloud
          //  Clean the first and the second point cloud with everything larger or smaller than this threashold configParams.reflectorMaxZaxisDistanceForOutlierMeter 
          // pcFirstOrg = select(ptCloudSenceReflectors, find(kmeansIdx == 1));
          // 'SmartLoaderEstiamteLocations:82' pcFirstOrgXyz = ptCloudSenceReflectorsXyz(kmeansIdx == 1,:); 
          //  figure, PlotPointCloud(pcFirst);
          // 'SmartLoaderEstiamteLocations:85' [pcFirstXyz] = FilterPointCloudAccordingToZdifferences(pcFirstOrgXyz, configParams.reflectorMaxZaxisDistanceForOutlierMeter); 
          m = kmeansIdx_size[0] - 1;
          trueCount = 0;
          for (loop_ub = 0; loop_ub <= m; loop_ub++) {
            if (SD->u6.f20.kmeansIdx_data[loop_ub] == 1.0) {
              trueCount++;
            }
          }

          partialTrueCount = 0;
          for (loop_ub = 0; loop_ub <= m; loop_ub++) {
            if (SD->u6.f20.kmeansIdx_data[loop_ub] == 1.0) {
              SD->u6.f20.b_tmp_data[partialTrueCount] = loop_ub + 1;
              partialTrueCount++;
            }
          }

          b_ptCloudSenceReflectorsXyz_size[1] = 3;
          b_ptCloudSenceReflectorsXyz_size[0] = trueCount;
          for (i46 = 0; i46 < trueCount; i46++) {
            m = 3 * (SD->u6.f20.b_tmp_data[i46] - 1);
            SD->u6.f20.b_ptCloudSenceReflectorsXyz_data[3 * i46] =
              SD->u6.f20.ptCloudSenceReflectorsXyz_data[m];
            SD->u6.f20.b_ptCloudSenceReflectorsXyz_data[1 + 3 * i46] =
              SD->u6.f20.ptCloudSenceReflectorsXyz_data[1 + m];
            SD->u6.f20.b_ptCloudSenceReflectorsXyz_data[2 + 3 * i46] =
              SD->u6.f20.ptCloudSenceReflectorsXyz_data[2 + m];
          }

          FilterPointCloudAccordingToZdifferences(SD,
            SD->u6.f20.b_ptCloudSenceReflectorsXyz_data,
            b_ptCloudSenceReflectorsXyz_size,
            configParams->reflectorMaxZaxisDistanceForOutlierMeter,
            SD->u6.f20.pcFirstXyz_data, pcFirstXyz_size);

          //  figure, PlotPointCloud(pcFirstXyz);
          // pcFirstRange = RangeCompiledVersion(pcFirst.Location);
          // 'SmartLoaderEstiamteLocations:89' pcFirstRange = RangeCompiledVersion(pcFirstXyz); 
          // 'RangeCompiledVersion:4' coder.inline('always');
          // 'RangeCompiledVersion:7' output = max(input) - min(input);
          m = pcFirstXyz_size[0];
          pcFirstRange_idx_0 = SD->u6.f20.pcFirstXyz_data[0];
          pcFirstRange_idx_1 = SD->u6.f20.pcFirstXyz_data[1];
          for (loop_ub = 2; loop_ub <= m; loop_ub++) {
            i46 = 3 * (loop_ub - 1);
            if (pcFirstRange_idx_0 < SD->u6.f20.pcFirstXyz_data[i46]) {
              pcFirstRange_idx_0 = SD->u6.f20.pcFirstXyz_data[i46];
            }

            pcFirstDistanceToPlane = SD->u6.f20.pcFirstXyz_data[1 + i46];
            if (pcFirstRange_idx_1 < pcFirstDistanceToPlane) {
              pcFirstRange_idx_1 = pcFirstDistanceToPlane;
            }
          }

          m = pcFirstXyz_size[0];
          b_kmeansC[0] = SD->u6.f20.pcFirstXyz_data[0];
          b_kmeansC[1] = SD->u6.f20.pcFirstXyz_data[1];
          for (loop_ub = 2; loop_ub <= m; loop_ub++) {
            i46 = 3 * (loop_ub - 1);
            if (b_kmeansC[0] > SD->u6.f20.pcFirstXyz_data[i46]) {
              b_kmeansC[0] = SD->u6.f20.pcFirstXyz_data[i46];
            }

            pcFirstDistanceToPlane = SD->u6.f20.pcFirstXyz_data[1 + i46];
            if (b_kmeansC[1] > pcFirstDistanceToPlane) {
              b_kmeansC[1] = pcFirstDistanceToPlane;
            }
          }

          pcFirstRange_idx_0 -= b_kmeansC[0];
          pcFirstRange_idx_1 -= b_kmeansC[1];

          // pcSecondOrg = select(ptCloudSenceReflectors, find(kmeansIdx == 2)); 
          // 'SmartLoaderEstiamteLocations:92' pcSecondOrgXyz = ptCloudSenceReflectorsXyz(kmeansIdx == 2,:); 
          //  figure, PlotPointCloud(pcSecondOrg);
          // 'SmartLoaderEstiamteLocations:95' [pcSecondXyz] = FilterPointCloudAccordingToZdifferences(pcSecondOrgXyz, configParams.reflectorMaxZaxisDistanceForOutlierMeter); 
          m = kmeansIdx_size[0] - 1;
          trueCount = 0;
          for (loop_ub = 0; loop_ub <= m; loop_ub++) {
            if (SD->u6.f20.kmeansIdx_data[loop_ub] == 2.0) {
              trueCount++;
            }
          }

          partialTrueCount = 0;
          for (loop_ub = 0; loop_ub <= m; loop_ub++) {
            if (SD->u6.f20.kmeansIdx_data[loop_ub] == 2.0) {
              SD->u6.f20.e_tmp_data[partialTrueCount] = loop_ub + 1;
              partialTrueCount++;
            }
          }

          d_ptCloudSenceReflectorsXyz_size[1] = 3;
          d_ptCloudSenceReflectorsXyz_size[0] = trueCount;
          for (i46 = 0; i46 < trueCount; i46++) {
            m = 3 * (SD->u6.f20.e_tmp_data[i46] - 1);
            SD->u6.f20.b_ptCloudSenceReflectorsXyz_data[3 * i46] =
              SD->u6.f20.ptCloudSenceReflectorsXyz_data[m];
            SD->u6.f20.b_ptCloudSenceReflectorsXyz_data[1 + 3 * i46] =
              SD->u6.f20.ptCloudSenceReflectorsXyz_data[1 + m];
            SD->u6.f20.b_ptCloudSenceReflectorsXyz_data[2 + 3 * i46] =
              SD->u6.f20.ptCloudSenceReflectorsXyz_data[2 + m];
          }

          FilterPointCloudAccordingToZdifferences(SD,
            SD->u6.f20.b_ptCloudSenceReflectorsXyz_data,
            d_ptCloudSenceReflectorsXyz_size,
            configParams->reflectorMaxZaxisDistanceForOutlierMeter,
            SD->u6.f20.pcSecondXyz_data, input_sizes);

          //  figure, PlotPointCloud(pcSecond);
          // pcSecondRange = RangeCompiledVersion(pcSecond.Location);
          // 'SmartLoaderEstiamteLocations:98' pcSecondRange = RangeCompiledVersion(pcSecondXyz); 
          // 'RangeCompiledVersion:4' coder.inline('always');
          // 'RangeCompiledVersion:7' output = max(input) - min(input);
          m = input_sizes[0];
          pcSecondRange_idx_0 = SD->u6.f20.pcSecondXyz_data[0];
          pcSecondRange_idx_1 = SD->u6.f20.pcSecondXyz_data[1];
          for (loop_ub = 2; loop_ub <= m; loop_ub++) {
            i46 = 3 * (loop_ub - 1);
            if (pcSecondRange_idx_0 < SD->u6.f20.pcSecondXyz_data[i46]) {
              pcSecondRange_idx_0 = SD->u6.f20.pcSecondXyz_data[i46];
            }

            pcFirstDistanceToPlane = SD->u6.f20.pcSecondXyz_data[1 + i46];
            if (pcSecondRange_idx_1 < pcFirstDistanceToPlane) {
              pcSecondRange_idx_1 = pcFirstDistanceToPlane;
            }
          }

          m = input_sizes[0];
          b_kmeansC[0] = SD->u6.f20.pcSecondXyz_data[0];
          b_kmeansC[1] = SD->u6.f20.pcSecondXyz_data[1];
          for (loop_ub = 2; loop_ub <= m; loop_ub++) {
            i46 = 3 * (loop_ub - 1);
            if (b_kmeansC[0] > SD->u6.f20.pcSecondXyz_data[i46]) {
              b_kmeansC[0] = SD->u6.f20.pcSecondXyz_data[i46];
            }

            pcFirstDistanceToPlane = SD->u6.f20.pcSecondXyz_data[1 + i46];
            if (b_kmeansC[1] > pcFirstDistanceToPlane) {
              b_kmeansC[1] = pcFirstDistanceToPlane;
            }
          }

          pcSecondRange_idx_0 -= b_kmeansC[0];
          pcSecondRange_idx_1 -= b_kmeansC[1];

          //     %% Determine which cluster is the loader or the shovel
          //  First stradegy - the loader point cloud reflector is the cluster center closest to the previous loader location 
          //  This is simply for letting matlab coder know that ptCloudLoaderReflectors and  ptCloudShovelReflectors will eventually set to a certain value 
          //  we give them an initial value that will change during the code
          // ptCloudLoaderReflectors = pcSecond;
          // 'SmartLoaderEstiamteLocations:106' ptCloudLoaderReflectorsXyz = pcSecondXyz; 
          ptCloudSenceReflectorsXyz_size[0] = input_sizes[0];
          m = input_sizes[1] * input_sizes[0];
          if (0 <= m - 1) {
            memcpy(&SD->u6.f20.ptCloudSenceReflectorsXyz_data[0],
                   &SD->u6.f20.pcSecondXyz_data[0], (unsigned int)(m * (int)
                    sizeof(float)));
          }

          // ptCloudShovelReflectors = pcFirst;
          // 'SmartLoaderEstiamteLocations:108' ptCloudShovelReflectorsXyz = pcFirstXyz; 
          ptCloudShovelReflectorsXyz_size[1] = 3;
          ptCloudShovelReflectorsXyz_size[0] = pcFirstXyz_size[0];
          partialTrueCount = pcFirstXyz_size[1] * pcFirstXyz_size[0];
          if (0 <= partialTrueCount - 1) {
            memcpy(&SD->u6.f20.ptCloudShovelReflectorsXyz_data[0],
                   &SD->u6.f20.pcFirstXyz_data[0], (unsigned int)
                   (partialTrueCount * (int)sizeof(float)));
          }

          // 'SmartLoaderEstiamteLocations:111' isFoundLoaderPc = false;
          isFoundLoaderPc = false;

          // 'SmartLoaderEstiamteLocations:112' if ~isempty(SmartLoaderGlobal.smartLoaderStructHistory) 
          if (SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] != 0) {
            // 'SmartLoaderEstiamteLocations:113' if size(SmartLoaderGlobal.smartLoaderStructHistory,1) >= 2 && ... 
            // 'SmartLoaderEstiamteLocations:114'                 configParams.timeTagMs - SmartLoaderGlobal.loaderTimeTatHistoryMs(end) < configParams.maximumTimeTagDiffMs && ... 
            // 'SmartLoaderEstiamteLocations:115'                 configParams.timeTagMs - SmartLoaderGlobal.loaderTimeTatHistoryMs(end-1) < configParams.maximumTimeTagDiffMs 
            guard2 = false;
            if (SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] >= 2)
            {
              u0 = SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[SD->
                pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] - 1];
              u1 = configParams->timeTagMs - u0;
              if (u1 < configParams->maximumTimeTagDiffMs) {
                u2 = SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data
                  [SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] - 2];
                if (configParams->timeTagMs - u2 <
                    configParams->maximumTimeTagDiffMs) {
                  //  Estiamte where loader location should be according to the previous locations of the loader 
                  // 'SmartLoaderEstiamteLocations:118' v = (SmartLoaderGlobal.smartLoaderStructHistory(end).loaderLoc - SmartLoaderGlobal.smartLoaderStructHistory(end-1).loaderLoc) / ... 
                  // 'SmartLoaderEstiamteLocations:119'                 double(SmartLoaderGlobal.loaderTimeTatHistoryMs(end) - SmartLoaderGlobal.loaderTimeTatHistoryMs(end-1)); 
                  d5 = (double)(u0 - u2);

                  // .
                  // 'SmartLoaderEstiamteLocations:121' estimatedLoaderLoc = SmartLoaderGlobal.smartLoaderStructHistory(end).loaderLoc + .... 
                  // 'SmartLoaderEstiamteLocations:122'                 v * double(configParams.timeTagMs - SmartLoaderGlobal.loaderTimeTatHistoryMs(end)); 
                  v_idx_0 = SD->
                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                    loaderLoc[0] + (SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                                    [SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size
                                    [0] - 1].loaderLoc[0] - SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                                    [SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size
                                    [0] - 2].loaderLoc[0]) / d5 * (double)u1;
                  v_idx_1 = SD->
                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                    loaderLoc[1] + (SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                                    [SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size
                                    [0] - 1].loaderLoc[1] - SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                                    [SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size
                                    [0] - 2].loaderLoc[1]) / d5 * (double)u1;
                  singleReflectorRangeLimitMeter = SD->
                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                    loaderLoc[2] + (SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                                    [SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size
                                    [0] - 1].loaderLoc[2] - SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                                    [SD->
                                    pd->SmartLoaderGlobal.smartLoaderStructHistory.size
                                    [0] - 2].loaderLoc[2]) / d5 * (double)u1;
                } else {
                  guard2 = true;
                }
              } else {
                guard2 = true;
              }
            } else {
              guard2 = true;
            }

            if (guard2) {
              // 'SmartLoaderEstiamteLocations:123' else
              // 'SmartLoaderEstiamteLocations:124' estimatedLoaderLoc = SmartLoaderGlobal.smartLoaderStructHistory(end).loaderLoc; 
              v_idx_0 = SD->pd->
                SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
                pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                loaderLoc[0];
              v_idx_1 = SD->pd->
                SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
                pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                loaderLoc[1];
              singleReflectorRangeLimitMeter = SD->
                pd->SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
                pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                loaderLoc[2];
            }

            // 'SmartLoaderEstiamteLocations:127' if ~isempty(estimatedLoaderLoc) 
            // 'SmartLoaderEstiamteLocations:129' distanceToKmeansClusterMeter = vecnorm((kmeansC - [estimatedLoaderLoc'; estimatedLoaderLoc'])'); 
            c_kmeansC[0] = kmeansC[0] - (float)v_idx_0;
            c_kmeansC[1] = kmeansC[3] - (float)v_idx_0;
            c_kmeansC[2] = kmeansC[1] - (float)v_idx_1;
            c_kmeansC[3] = kmeansC[4] - (float)v_idx_1;
            c_kmeansC[4] = kmeansC[2] - (float)singleReflectorRangeLimitMeter;
            c_kmeansC[5] = kmeansC[5] - (float)singleReflectorRangeLimitMeter;
            vecnorm(c_kmeansC, distanceToKmeansClusterMeter);

            //  There are 38 cm distance between the two reflectors,
            //  we'd like ensure the previous location of the loader reside within a 10cm margin 
            // 'SmartLoaderEstiamteLocations:133' if any(distanceToKmeansClusterMeter - configParams.previousLoaderLocationToCurrentLocationMaximumDistanceMeter < 0) 
            b_r[0] = (distanceToKmeansClusterMeter[0] - (float)
                      configParams->previousLoaderLocationToCurrentLocationMaximumDistanceMeter
                      < 0.0F);
            b_r[1] = (distanceToKmeansClusterMeter[1] - (float)
                      configParams->previousLoaderLocationToCurrentLocationMaximumDistanceMeter
                      < 0.0F);
            if (any(b_r)) {
              // 'SmartLoaderEstiamteLocations:135' if distanceToKmeansClusterMeter(1) < distanceToKmeansClusterMeter(2) 
              if (distanceToKmeansClusterMeter[0] <
                  distanceToKmeansClusterMeter[1]) {
                // ptCloudLoaderReflectors = pcFirst; ptCloudShovelReflectors = pcSecond; 
                // 'SmartLoaderEstiamteLocations:137' ptCloudLoaderReflectorsXyz = pcFirstXyz; 
                ptCloudSenceReflectorsXyz_size[0] = pcFirstXyz_size[0];
                if (0 <= partialTrueCount - 1) {
                  memcpy(&SD->u6.f20.ptCloudSenceReflectorsXyz_data[0],
                         &SD->u6.f20.pcFirstXyz_data[0], (unsigned int)
                         (partialTrueCount * (int)sizeof(float)));
                }

                // 'SmartLoaderEstiamteLocations:137' ptCloudShovelReflectorsXyz = pcSecondXyz; 
                ptCloudShovelReflectorsXyz_size[1] = 3;
                ptCloudShovelReflectorsXyz_size[0] = input_sizes[0];
                if (0 <= m - 1) {
                  memcpy(&SD->u6.f20.ptCloudShovelReflectorsXyz_data[0],
                         &SD->u6.f20.pcSecondXyz_data[0], (unsigned int)(m *
                          (int)sizeof(float)));
                }
              } else {
                // 'SmartLoaderEstiamteLocations:138' else
                // ptCloudLoaderReflectors = pcSecond; ptCloudShovelReflectors = pcFirst; 
                // 'SmartLoaderEstiamteLocations:140' ptCloudLoaderReflectorsXyz = pcSecondXyz; 
                // 'SmartLoaderEstiamteLocations:140' ptCloudShovelReflectorsXyz = pcFirstXyz; 
              }

              // 'SmartLoaderEstiamteLocations:143' isFoundLoaderPc = true;
              isFoundLoaderPc = true;
            }
          }

          // 'SmartLoaderEstiamteLocations:148' if ~isFoundLoaderPc
          if (!isFoundLoaderPc) {
            //  Stradegy - we know that the loader height from the ground plane is fix number, however the shovel is mostly 
            //  reside below this height - therefor we'll determine the loader and the shovel according to the loader minimum height 
            //  Find the point with the median z coordiante
            // [~, I] = sort(pcFirst.Location(:,3));
            // 'SmartLoaderEstiamteLocations:154' if size(pcFirstXyz,1) == 1
            if (pcFirstXyz_size[0] == 1) {
              // 'SmartLoaderEstiamteLocations:155' [pcFirstDistanceToPlane, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, pcFirstXyz(1,:)); 
              CalcPlaneToPointDistance(configParams->planeModelParameters,
                *(float (*)[3])&SD->u6.f20.pcFirstXyz_data[0],
                &pcFirstDistanceToPlane, &empty_non_axis_sizes);

              // 'SmartLoaderEstiamteLocations:155' ~
            } else {
              // 'SmartLoaderEstiamteLocations:156' else
              // 'SmartLoaderEstiamteLocations:157' [~, I] = sort(pcFirstXyz(:,3)); 
              x_size[0] = pcFirstXyz_size[0];
              loop_ub = pcFirstXyz_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                SD->u6.f20.x_data[i46] = SD->u6.f20.pcFirstXyz_data[2 + 3 * i46];
              }

              sort(SD->u6.f20.x_data, x_size, SD->u6.f20.iidx_data,
                   kmeansIdx_size);

              // 'SmartLoaderEstiamteLocations:157' ~
              // 'SmartLoaderEstiamteLocations:158' zMedianPointFirst = pcFirstXyz(floor(size(I,1)/2),:); 
              // 'SmartLoaderEstiamteLocations:159' [pcFirstDistanceToPlane, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, zMedianPointFirst); 
              CalcPlaneToPointDistance(configParams->planeModelParameters,
                *(float (*)[3])&SD->u6.f20.pcFirstXyz_data[3 * ((int)std::floor
                ((double)kmeansIdx_size[0] / 2.0) - 1)], &pcFirstDistanceToPlane,
                &empty_non_axis_sizes);

              // 'SmartLoaderEstiamteLocations:159' ~
            }

            // 'SmartLoaderEstiamteLocations:162' if size(pcSecondXyz,1) == 1
            if (input_sizes[0] == 1) {
              // 'SmartLoaderEstiamteLocations:163' [pcSecondDistanceToPlane, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, pcSecondXyz(1,:)); 
              CalcPlaneToPointDistance(configParams->planeModelParameters,
                *(float (*)[3])&SD->u6.f20.pcSecondXyz_data[0],
                &pcSecondDistanceToPlane, &empty_non_axis_sizes);

              // 'SmartLoaderEstiamteLocations:163' ~
            } else {
              // 'SmartLoaderEstiamteLocations:164' else
              // 'SmartLoaderEstiamteLocations:165' [~, I] = sort(pcSecondXyz(:,3)); 
              x_size[0] = input_sizes[0];
              loop_ub = input_sizes[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                SD->u6.f20.x_data[i46] = SD->u6.f20.pcSecondXyz_data[2 + 3 * i46];
              }

              sort(SD->u6.f20.x_data, x_size, SD->u6.f20.iidx_data,
                   kmeansIdx_size);

              // 'SmartLoaderEstiamteLocations:165' ~
              // 'SmartLoaderEstiamteLocations:166' zMedianPointSecond = pcSecondXyz(floor(size(I,1)/2),:); 
              // 'SmartLoaderEstiamteLocations:167' [pcSecondDistanceToPlane, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, zMedianPointSecond); 
              CalcPlaneToPointDistance(configParams->planeModelParameters,
                *(float (*)[3])&SD->u6.f20.pcSecondXyz_data[3 * ((int)std::floor
                ((double)kmeansIdx_size[0] / 2.0) - 1)],
                &pcSecondDistanceToPlane, &empty_non_axis_sizes);

              // 'SmartLoaderEstiamteLocations:167' ~
            }

            // 'SmartLoaderEstiamteLocations:171' if pcFirstDistanceToPlane > configParams.minimumDistanceFromLoaderToPlaneMeter &&... 
            // 'SmartLoaderEstiamteLocations:172'                 pcSecondDistanceToPlane < configParams.minimumDistanceFromLoaderToPlaneMeter 
            if ((pcFirstDistanceToPlane >
                 configParams->minimumDistanceFromLoaderToPlaneMeter) &&
                (pcSecondDistanceToPlane <
                 configParams->minimumDistanceFromLoaderToPlaneMeter)) {
              // 'SmartLoaderEstiamteLocations:173' ptCloudLoaderReflectorsXyz = pcFirstXyz; 
              ptCloudSenceReflectorsXyz_size[0] = pcFirstXyz_size[0];
              if (0 <= partialTrueCount - 1) {
                memcpy(&SD->u6.f20.ptCloudSenceReflectorsXyz_data[0],
                       &SD->u6.f20.pcFirstXyz_data[0], (unsigned int)
                       (partialTrueCount * (int)sizeof(float)));
              }

              // 'SmartLoaderEstiamteLocations:173' ptCloudShovelReflectorsXyz = pcSecondXyz; 
              ptCloudShovelReflectorsXyz_size[1] = 3;
              ptCloudShovelReflectorsXyz_size[0] = input_sizes[0];
              if (0 <= m - 1) {
                memcpy(&SD->u6.f20.ptCloudShovelReflectorsXyz_data[0],
                       &SD->u6.f20.pcSecondXyz_data[0], (unsigned int)(m * (int)
                        sizeof(float)));
              }

              // 'SmartLoaderEstiamteLocations:174' isFoundLoaderPc = true;
              isFoundLoaderPc = true;
            } else {
              if ((pcFirstDistanceToPlane <
                   configParams->minimumDistanceFromLoaderToPlaneMeter) &&
                  (pcSecondDistanceToPlane >
                   configParams->minimumDistanceFromLoaderToPlaneMeter)) {
                // 'SmartLoaderEstiamteLocations:176' elseif pcFirstDistanceToPlane < configParams.minimumDistanceFromLoaderToPlaneMeter &&... 
                // 'SmartLoaderEstiamteLocations:177'                 pcSecondDistanceToPlane > configParams.minimumDistanceFromLoaderToPlaneMeter 
                // 'SmartLoaderEstiamteLocations:178' ptCloudLoaderReflectorsXyz = pcSecondXyz; 
                ptCloudSenceReflectorsXyz_size[0] = input_sizes[0];
                if (0 <= m - 1) {
                  memcpy(&SD->u6.f20.ptCloudSenceReflectorsXyz_data[0],
                         &SD->u6.f20.pcSecondXyz_data[0], (unsigned int)(m *
                          (int)sizeof(float)));
                }

                // 'SmartLoaderEstiamteLocations:178' ptCloudShovelReflectorsXyz = pcFirstXyz; 
                ptCloudShovelReflectorsXyz_size[1] = 3;
                ptCloudShovelReflectorsXyz_size[0] = pcFirstXyz_size[0];
                if (0 <= partialTrueCount - 1) {
                  memcpy(&SD->u6.f20.ptCloudShovelReflectorsXyz_data[0],
                         &SD->u6.f20.pcFirstXyz_data[0], (unsigned int)
                         (partialTrueCount * (int)sizeof(float)));
                }

                // 'SmartLoaderEstiamteLocations:179' isFoundLoaderPc = true;
                isFoundLoaderPc = true;
              }
            }
          }

          // 'SmartLoaderEstiamteLocations:184' if ~isFoundLoaderPc
          // 'SmartLoaderEstiamteLocations:193' if ~isFoundLoaderPc
          if (!isFoundLoaderPc) {
            //  Third stradegy - determine the range for both clusters, the loader reflector suppose to be circle shaped 
            //  and the shovel reflector supposes to be much more rectangular shaped 
            // 'SmartLoaderEstiamteLocations:196' pcFirstMinorToMajorRation = min(pcFirstRange(1:2)) / max(pcFirstRange(1:2)); 
            // 'SmartLoaderEstiamteLocations:197' pcSecondMinorToMajorRation = min(pcSecondRange(1:2)) / max(pcSecondRange(1:2)); 
            //  TODO : add a limit to the differenct between pcFirstMinorToMajorRation to pcSecondMinorToMajorRation 
            // 'SmartLoaderEstiamteLocations:200' if pcSecondMinorToMajorRation < pcFirstMinorToMajorRation 
            if (pcSecondRange_idx_0 > pcSecondRange_idx_1) {
              b_pcSecondRange_idx_0 = pcSecondRange_idx_1;
            } else {
              b_pcSecondRange_idx_0 = pcSecondRange_idx_0;
            }

            if (pcSecondRange_idx_0 < pcSecondRange_idx_1) {
              c_pcSecondRange_idx_0 = pcSecondRange_idx_1;
            } else {
              c_pcSecondRange_idx_0 = pcSecondRange_idx_0;
            }

            if (pcFirstRange_idx_0 > pcFirstRange_idx_1) {
              b_pcFirstRange_idx_0 = pcFirstRange_idx_1;
            } else {
              b_pcFirstRange_idx_0 = pcFirstRange_idx_0;
            }

            if (pcFirstRange_idx_0 < pcFirstRange_idx_1) {
              c_pcFirstRange_idx_0 = pcFirstRange_idx_1;
            } else {
              c_pcFirstRange_idx_0 = pcFirstRange_idx_0;
            }

            if (b_pcSecondRange_idx_0 / c_pcSecondRange_idx_0 <
                b_pcFirstRange_idx_0 / c_pcFirstRange_idx_0) {
              // 'SmartLoaderEstiamteLocations:201' ptCloudLoaderReflectorsXyz = pcFirstXyz; 
              ptCloudSenceReflectorsXyz_size[0] = pcFirstXyz_size[0];
              if (0 <= partialTrueCount - 1) {
                memcpy(&SD->u6.f20.ptCloudSenceReflectorsXyz_data[0],
                       &SD->u6.f20.pcFirstXyz_data[0], (unsigned int)
                       (partialTrueCount * (int)sizeof(float)));
              }

              // 'SmartLoaderEstiamteLocations:201' ptCloudShovelReflectorsXyz = pcSecondXyz; 
              ptCloudShovelReflectorsXyz_size[1] = 3;
              ptCloudShovelReflectorsXyz_size[0] = input_sizes[0];
              if (0 <= m - 1) {
                memcpy(&SD->u6.f20.ptCloudShovelReflectorsXyz_data[0],
                       &SD->u6.f20.pcSecondXyz_data[0], (unsigned int)(m * (int)
                        sizeof(float)));
              }
            } else {
              // 'SmartLoaderEstiamteLocations:202' else
              // 'SmartLoaderEstiamteLocations:203' ptCloudLoaderReflectorsXyz = pcSecondXyz; 
              ptCloudSenceReflectorsXyz_size[0] = input_sizes[0];
              if (0 <= m - 1) {
                memcpy(&SD->u6.f20.ptCloudSenceReflectorsXyz_data[0],
                       &SD->u6.f20.pcSecondXyz_data[0], (unsigned int)(m * (int)
                        sizeof(float)));
              }

              // 'SmartLoaderEstiamteLocations:203' ptCloudShovelReflectorsXyz = pcFirstXyz; 
              ptCloudShovelReflectorsXyz_size[1] = 3;
              ptCloudShovelReflectorsXyz_size[0] = pcFirstXyz_size[0];
              if (0 <= partialTrueCount - 1) {
                memcpy(&SD->u6.f20.ptCloudShovelReflectorsXyz_data[0],
                       &SD->u6.f20.pcFirstXyz_data[0], (unsigned int)
                       (partialTrueCount * (int)sizeof(float)));
              }
            }
          }

          //  figure, PlotPointCloud(ptCloudLoaderReflectorsXyz);
          //  figure, PlotPointCloud(ptCloudShovelReflectorsXyz);
          //     %% Estimate the shovel loc
          // 'SmartLoaderEstiamteLocations:211' if size(ptCloudShovelReflectorsXyz,1) >= configParams.minPointsForReflector 
          if (ptCloudShovelReflectorsXyz_size[0] >=
              configParams->minPointsForReflector) {
            // 'SmartLoaderEstiamteLocations:212' smartLoaderStruct.shovelLoc = double(mean(ptCloudShovelReflectorsXyz)'); 
            mean(SD->u6.f20.ptCloudShovelReflectorsXyz_data,
                 ptCloudShovelReflectorsXyz_size, b_kmeansC);
            smartLoaderStruct->shovelLoc[0] = b_kmeansC[0];
            smartLoaderStruct->shovelLoc[1] = b_kmeansC[1];
            smartLoaderStruct->shovelLoc[2] = b_kmeansC[2];

            // 'SmartLoaderEstiamteLocations:213' smartLoaderStruct.shovelLocStatus = true; 
            smartLoaderStruct->shovelLocStatus = true;
          }

          guard1 = true;
        }
      }

      if (guard1) {
        //  Remove outliers points in the loader point cloud. these points z's coordinate difference is larger than the following threshold 
        //  Remove points which are outliers in the z axis
        //  figure, PlotPointCloud(ptCloudLoaderReflectors);
        // 'SmartLoaderEstiamteLocations:220' zCor = ptCloudLoaderReflectorsXyz(:,3); 
        loop_ub = ptCloudSenceReflectorsXyz_size[0];
        for (i46 = 0; i46 < loop_ub; i46++) {
          SD->u6.f20.tmp_data[i46] = SD->u6.f20.ptCloudSenceReflectorsXyz_data[2
            + 3 * i46];
        }

        // 'SmartLoaderEstiamteLocations:221' zMedian = median(zCor);
        iv1[0] = ptCloudSenceReflectorsXyz_size[0];
        pcFirstRange_idx_0 = median(SD, SD->u6.f20.tmp_data, iv1);

        // 'SmartLoaderEstiamteLocations:222' assert(numel(zMedian) == 1);
        // 'SmartLoaderEstiamteLocations:223' loaderReflectorPtrInd = abs(zCor - zMedian) <= configParams.loaderReflectorMaxZaxisDistanceForOutlierMeter; 
        loop_ub = ptCloudSenceReflectorsXyz_size[0];
        tmp_size[0] = ptCloudSenceReflectorsXyz_size[0];
        for (i46 = 0; i46 < loop_ub; i46++) {
          SD->u6.f20.c_tmp_data[i46] = SD->u6.f20.tmp_data[i46] -
            pcFirstRange_idx_0;
        }

        b_abs(SD->u6.f20.c_tmp_data, tmp_size, SD->u6.f20.x_data, x_size);
        loop_ub = x_size[0];
        for (i46 = 0; i46 < loop_ub; i46++) {
          SD->u6.f20.ptCloudSenceReflectorsInd_data[i46] = (SD->
            u6.f20.x_data[i46] <=
            configParams->loaderReflectorMaxZaxisDistanceForOutlierMeter);
        }

        //  sum(loaderReflectorPtrInd)
        // ptCloudLoaderReflectorsFilterd = select(ptCloudLoaderReflectors, find(loaderReflectorPtrInd)); 
        // 'SmartLoaderEstiamteLocations:226' ptCloudLoaderReflectorsFilterdXyz = ptCloudLoaderReflectorsXyz(loaderReflectorPtrInd,:); 
        //  figure, PlotPointCloud(ptCloudLoaderReflectorsXyz);
        //  figure, PlotPointCloud(ptCloudLoaderReflectorsFilterdXyz);
        //  Determine the number of lines, determine the cluster of lines.
        //  Determine Find the number of lines
        // 'SmartLoaderEstiamteLocations:233' coder.varsize('ptr', [SmartLoaderCompilationConstants.MaxPointCloudSize 3], [1 0]); 
        // 'SmartLoaderEstiamteLocations:235' if coder.target('Matlab') && false 
        // 'SmartLoaderEstiamteLocations:247' else
        //  Cluster in 2D
        // 'SmartLoaderEstiamteLocations:249' coder.varsize('clustersYs', 'clustersXs', [1 64], [0 1]); 
        // 'SmartLoaderEstiamteLocations:250' [clustersXs, clustersYs, status] = ClusterPoints2D(ptCloudLoaderReflectorsFilterdXyz(:,1:2), configParams.maxDistanceBetweenEachRayMeter); 
        m = x_size[0] - 1;
        trueCount = 0;
        for (loop_ub = 0; loop_ub <= m; loop_ub++) {
          if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]) {
            trueCount++;
          }
        }

        partialTrueCount = 0;
        for (loop_ub = 0; loop_ub <= m; loop_ub++) {
          if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]) {
            SD->u6.f20.d_tmp_data[partialTrueCount] = loop_ub + 1;
            partialTrueCount++;
          }
        }

        for (i46 = 0; i46 < trueCount; i46++) {
          m = 3 * (SD->u6.f20.d_tmp_data[i46] - 1);
          SD->u6.f20.pcSecondXyz_data[3 * i46] =
            SD->u6.f20.ptCloudSenceReflectorsXyz_data[m];
          SD->u6.f20.pcSecondXyz_data[1 + 3 * i46] =
            SD->u6.f20.ptCloudSenceReflectorsXyz_data[1 + m];
          SD->u6.f20.pcSecondXyz_data[2 + 3 * i46] =
            SD->u6.f20.ptCloudSenceReflectorsXyz_data[2 + m];
        }

        c_ptCloudSenceReflectorsXyz_size[1] = 2;
        c_ptCloudSenceReflectorsXyz_size[0] = trueCount;
        for (i46 = 0; i46 < trueCount; i46++) {
          m = i46 << 1;
          SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[m] =
            SD->u6.f20.pcSecondXyz_data[3 * i46];
          SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[1 + m] =
            SD->u6.f20.pcSecondXyz_data[1 + 3 * i46];
        }

        ClusterPoints2D(SD, SD->u6.f20.c_ptCloudSenceReflectorsXyz_data,
                        c_ptCloudSenceReflectorsXyz_size,
                        configParams->maxDistanceBetweenEachRayMeter,
                        clustersXs.data, clustersXs.size, clustersYs.data,
                        clustersYs.size, &empty_non_axis_sizes);

        // 'SmartLoaderEstiamteLocations:251' if isempty(clustersXs) || ~status
        if (empty_non_axis_sizes) {
          //  remove small clusters with less than 3 points
          // 'SmartLoaderEstiamteLocations:259' minimumNumPointsInCluster = 3;
          //  cellfun doens't works with matlab coder
          //  Previous code: clustersXs(cellfun('length',clustersXs)<minimumNumPointsInCluster) = []; 
          //  Previous code: clustersYs(cellfun('length',clustersYs)<minimumNumPointsInCluster) = []; 
          //  In order to solve this issue - I have coded the filter function by my own 
          // 'SmartLoaderEstiamteLocations:265' isInvalid = zeros(1,size(clustersXs,2),'logical'); 
          // 'SmartLoaderEstiamteLocations:266' for i = 1:size(clustersXs,2)
          for (loop_ub = 0; loop_ub < 64; loop_ub++) {
            isInvalid_data[loop_ub] = false;

            // 'SmartLoaderEstiamteLocations:267' if size(clustersXs{i},1) < minimumNumPointsInCluster 
            if (clustersXs.data[loop_ub].f1->size[0] < 3) {
              // 'SmartLoaderEstiamteLocations:268' isInvalid(i) = true;
              isInvalid_data[loop_ub] = true;
            }
          }

          // 'SmartLoaderEstiamteLocations:271' clustersXs(isInvalid) = [];
          nullAssignment(clustersXs.data, isInvalid_data, r2.data, r2.size);

          // 'SmartLoaderEstiamteLocations:272' clustersYs(isInvalid) = [];
          nullAssignment(clustersYs.data, isInvalid_data, clustersXs.data,
                         clustersXs.size);

          // 'SmartLoaderEstiamteLocations:275' if isempty(clustersXs)
          if (r2.size[1] == 0) {
            // 'SmartLoaderEstiamteLocations:276' if coder.target('Matlab')
            // 'SmartLoaderEstiamteLocations:279' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eFailed; 
            smartLoaderStruct->status = PerceptionSmartLoaderReturnValue_eFailed;
          } else {
            // 'SmartLoaderEstiamteLocations:282' if numel(clustersXs) == 1
            if (r2.size[1] == 1) {
              // 'SmartLoaderEstiamteLocations:283' if coder.target('Matlab')
              // 'SmartLoaderEstiamteLocations:286' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eFailed; 
              smartLoaderStruct->status =
                PerceptionSmartLoaderReturnValue_eFailed;
            } else {
              // 'SmartLoaderEstiamteLocations:290' if coder.target('Matlab') && false 
              //     %% Get the extreme points for each cluster - these points will be use for circle center estimation 
              // 'SmartLoaderEstiamteLocations:303' coder.varsize('extremePoints', [SmartLoaderCompilationConstants.MaxNumClusters 2], [1 0]); 
              // 'SmartLoaderEstiamteLocations:305' extremePoints = zeros(0,2,'like', ptCloudLoaderReflectorsFilterdXyz); 
              extremePoints_size[1] = 2;
              extremePoints_size[0] = 0;

              // 'SmartLoaderEstiamteLocations:306' for q = 1:numel(clustersXs)
              i46 = r2.size[1];
              for (q = 0; q < i46; q++) {
                // 'SmartLoaderEstiamteLocations:307' pdistOutput = pdist([clustersXs{q} clustersYs{q}]); 
                if ((r2.data[q].f1->size[0] != 0) && (r2.data[q].f1->size[1] !=
                     0)) {
                  m = r2.data[q].f1->size[0];
                } else if ((clustersXs.data[q].f1->size[0] != 0) &&
                           (clustersXs.data[q].f1->size[1] != 0)) {
                  m = clustersXs.data[q].f1->size[0];
                } else {
                  m = r2.data[q].f1->size[0];
                  if (m <= 0) {
                    m = 0;
                  }

                  if (clustersXs.data[q].f1->size[0] > m) {
                    m = clustersXs.data[q].f1->size[0];
                  }
                }

                empty_non_axis_sizes = (m == 0);
                if (empty_non_axis_sizes || ((r2.data[q].f1->size[0] != 0) &&
                     (r2.data[q].f1->size[1] != 0))) {
                  partialTrueCount = r2.data[q].f1->size[1];
                } else {
                  partialTrueCount = 0;
                }

                if ((partialTrueCount == r2.data[q].f1->size[1]) && (m ==
                     r2.data[q].f1->size[0])) {
                  i47 = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
                  reshapes[0].f1->size[1] = partialTrueCount;
                  reshapes[0].f1->size[0] = m;
                  emxEnsureCapacity_real32_T(reshapes[0].f1, i47);
                  loop_ub = partialTrueCount * m;
                  for (i47 = 0; i47 < loop_ub; i47++) {
                    reshapes[0].f1->data[i47] = r2.data[q].f1->data[i47];
                  }
                } else {
                  i47 = 0;
                  i48 = 0;
                  trueCount = 0;
                  i49 = 0;
                  loop_ub = reshapes[0].f1->size[0] * reshapes[0].f1->size[1];
                  reshapes[0].f1->size[1] = partialTrueCount;
                  reshapes[0].f1->size[0] = m;
                  emxEnsureCapacity_real32_T(reshapes[0].f1, loop_ub);
                  for (loop_ub = 0; loop_ub < m * partialTrueCount; loop_ub++) {
                    reshapes[0].f1->data[i48 + reshapes[0].f1->size[1] * i47] =
                      r2.data[q].f1->data[i49 + r2.data[q].f1->size[1] *
                      trueCount];
                    i47++;
                    trueCount++;
                    if (i47 > reshapes[0].f1->size[0] - 1) {
                      i47 = 0;
                      i48++;
                    }

                    if (trueCount > r2.data[q].f1->size[0] - 1) {
                      trueCount = 0;
                      i49++;
                    }
                  }
                }

                if (empty_non_axis_sizes || ((clustersXs.data[q].f1->size[0] !=
                      0) && (clustersXs.data[q].f1->size[1] != 0))) {
                  b_input_sizes = (signed char)clustersXs.data[q].f1->size[1];
                } else {
                  b_input_sizes = 0;
                }

                partialTrueCount = b_input_sizes;
                if ((b_input_sizes == clustersXs.data[q].f1->size[1]) && (m ==
                     clustersXs.data[q].f1->size[0])) {
                  i47 = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
                  reshapes[1].f1->size[1] = b_input_sizes;
                  reshapes[1].f1->size[0] = m;
                  emxEnsureCapacity_real32_T(reshapes[1].f1, i47);
                  loop_ub = b_input_sizes * m;
                  for (i47 = 0; i47 < loop_ub; i47++) {
                    reshapes[1].f1->data[i47] = clustersXs.data[q].f1->data[i47];
                  }
                } else {
                  i47 = 0;
                  i48 = 0;
                  trueCount = 0;
                  i49 = 0;
                  loop_ub = reshapes[1].f1->size[0] * reshapes[1].f1->size[1];
                  reshapes[1].f1->size[1] = b_input_sizes;
                  reshapes[1].f1->size[0] = m;
                  emxEnsureCapacity_real32_T(reshapes[1].f1, loop_ub);
                  for (loop_ub = 0; loop_ub < m * partialTrueCount; loop_ub++) {
                    reshapes[1].f1->data[i48 + reshapes[1].f1->size[1] * i47] =
                      clustersXs.data[q].f1->data[i49 + clustersXs.data[q]
                      .f1->size[1] * trueCount];
                    i47++;
                    trueCount++;
                    if (i47 > reshapes[1].f1->size[0] - 1) {
                      i47 = 0;
                      i48++;
                    }

                    if (trueCount > clustersXs.data[q].f1->size[0] - 1) {
                      trueCount = 0;
                      i49++;
                    }
                  }
                }

                i47 = d_reshapes->size[0] * d_reshapes->size[1];
                d_reshapes->size[1] = reshapes[0].f1->size[1] + reshapes[1]
                  .f1->size[1];
                d_reshapes->size[0] = reshapes[0].f1->size[0];
                emxEnsureCapacity_real32_T(d_reshapes, i47);
                loop_ub = reshapes[0].f1->size[0];
                for (i47 = 0; i47 < loop_ub; i47++) {
                  partialTrueCount = reshapes[0].f1->size[1];
                  for (i48 = 0; i48 < partialTrueCount; i48++) {
                    d_reshapes->data[i48 + d_reshapes->size[1] * i47] =
                      reshapes[0].f1->data[i48 + reshapes[0].f1->size[1] * i47];
                  }
                }

                loop_ub = reshapes[1].f1->size[0];
                for (i47 = 0; i47 < loop_ub; i47++) {
                  partialTrueCount = reshapes[1].f1->size[1];
                  for (i48 = 0; i48 < partialTrueCount; i48++) {
                    d_reshapes->data[(i48 + reshapes[0].f1->size[1]) +
                      d_reshapes->size[1] * i47] = reshapes[1].f1->data[i48 +
                      reshapes[1].f1->size[1] * i47];
                  }
                }

                pdist(d_reshapes, pdistOutput);

                // 'SmartLoaderEstiamteLocations:308' Z = squareform(pdistOutput); 
                squareform(pdistOutput, Z);

                //  Find the cooridnate of the most distanced points
                // 'SmartLoaderEstiamteLocations:310' [~, maxInd] = max(Z(:));
                m = Z->size[0] * Z->size[1];
                i47 = 0;
                i48 = 0;
                trueCount = 0;
                i49 = varargin_1->size[0];
                varargin_1->size[0] = m;
                emxEnsureCapacity_real32_T(varargin_1, i49);
                for (i49 = 0; i49 < m; i49++) {
                  varargin_1->data[i47] = Z->data[trueCount + Z->size[1] * i48];
                  i47++;
                  i48++;
                  if (i48 > Z->size[0] - 1) {
                    i48 = 0;
                    trueCount++;
                  }
                }

                m = varargin_1->size[0];
                if (varargin_1->size[0] <= 2) {
                  if (varargin_1->size[0] == 1) {
                    loop_ub = 0;
                  } else {
                    loop_ub = (varargin_1->data[0] < varargin_1->data[1]);
                  }
                } else {
                  pcFirstDistanceToPlane = varargin_1->data[0];
                  loop_ub = 0;
                  for (partialTrueCount = 2; partialTrueCount <= m;
                       partialTrueCount++) {
                    if (pcFirstDistanceToPlane < varargin_1->
                        data[partialTrueCount - 1]) {
                      pcFirstDistanceToPlane = varargin_1->data[partialTrueCount
                        - 1];
                      loop_ub = partialTrueCount - 1;
                    }
                  }
                }

                // 'SmartLoaderEstiamteLocations:310' ~
                // 'SmartLoaderEstiamteLocations:312' [row,col] = ind2sub(size(Z), maxInd); 
                singleReflectorRangeLimitMeter = Z->size[0];
                m = (int)(unsigned int)singleReflectorRangeLimitMeter;
                vk = loop_ub / m;
                m = loop_ub - vk * m;

                // 'SmartLoaderEstiamteLocations:314' tempXs = [clustersXs{q}(row,:) clustersYs{q}(row,:)]; 
                loop_ub = r2.data[q].f1->size[1];
                partialTrueCount = clustersXs.data[q].f1->size[1];
                tempXs_size = loop_ub + partialTrueCount;
                for (i47 = 0; i47 < loop_ub; i47++) {
                  tempXs_data[i47] = r2.data[q].f1->data[i47 + r2.data[q]
                    .f1->size[1] * m];
                }

                for (i47 = 0; i47 < partialTrueCount; i47++) {
                  tempXs_data[i47 + loop_ub] = clustersXs.data[q].f1->data[i47 +
                    clustersXs.data[q].f1->size[1] * m];
                }

                // 'SmartLoaderEstiamteLocations:315' extremePoints = [extremePoints; tempXs]; 
                if (extremePoints_size[0] != 0) {
                  sizes = 2;
                } else if (tempXs_size != 0) {
                  sizes = (signed char)tempXs_size;
                } else {
                  sizes = 2;
                }

                if (extremePoints_size[0] != 0) {
                  b_input_sizes = (signed char)extremePoints_size[0];
                } else {
                  b_input_sizes = 0;
                }

                m = b_input_sizes;
                partialTrueCount = sizes;
                if ((sizes == 2) && (b_input_sizes == extremePoints_size[0])) {
                  i47 = b_reshapes[0].f1->size[0] * b_reshapes[0].f1->size[1];
                  b_reshapes[0].f1->size[1] = 2;
                  b_reshapes[0].f1->size[0] = b_input_sizes;
                  emxEnsureCapacity_real32_T(b_reshapes[0].f1, i47);
                  loop_ub = 2 * b_input_sizes;
                  for (i47 = 0; i47 < loop_ub; i47++) {
                    b_reshapes[0].f1->data[i47] = extremePoints_data[i47];
                  }
                } else {
                  i47 = 0;
                  i48 = 0;
                  trueCount = 0;
                  i49 = 0;
                  loop_ub = b_reshapes[0].f1->size[0] * b_reshapes[0].f1->size[1];
                  b_reshapes[0].f1->size[1] = sizes;
                  b_reshapes[0].f1->size[0] = b_input_sizes;
                  emxEnsureCapacity_real32_T(b_reshapes[0].f1, loop_ub);
                  for (loop_ub = 0; loop_ub < m * partialTrueCount; loop_ub++) {
                    b_reshapes[0].f1->data[i48 + b_reshapes[0].f1->size[1] * i47]
                      = extremePoints_data[i49 + (trueCount << 1)];
                    i47++;
                    trueCount++;
                    if (i47 > b_reshapes[0].f1->size[0] - 1) {
                      i47 = 0;
                      i48++;
                    }

                    if (trueCount > extremePoints_size[0] - 1) {
                      trueCount = 0;
                      i49++;
                    }
                  }
                }

                b_input_sizes = (signed char)(tempXs_size != 0);
                m = b_input_sizes;
                partialTrueCount = sizes;
                if ((sizes == tempXs_size) && (b_input_sizes == 1)) {
                  i47 = b_reshapes[1].f1->size[0] * b_reshapes[1].f1->size[1];
                  b_reshapes[1].f1->size[1] = sizes;
                  b_reshapes[1].f1->size[0] = 1;
                  emxEnsureCapacity_real32_T(b_reshapes[1].f1, i47);
                  loop_ub = sizes;
                  for (i47 = 0; i47 < loop_ub; i47++) {
                    b_reshapes[1].f1->data[i47] = tempXs_data[i47];
                  }
                } else {
                  i47 = 0;
                  i48 = 0;
                  trueCount = 0;
                  i49 = b_reshapes[1].f1->size[0] * b_reshapes[1].f1->size[1];
                  b_reshapes[1].f1->size[1] = sizes;
                  b_reshapes[1].f1->size[0] = b_input_sizes;
                  emxEnsureCapacity_real32_T(b_reshapes[1].f1, i49);
                  for (i49 = 0; i49 < m * partialTrueCount; i49++) {
                    b_reshapes[1].f1->data[i48 + b_reshapes[1].f1->size[1] * i47]
                      = tempXs_data[trueCount];
                    i47++;
                    if (i47 > b_reshapes[1].f1->size[0] - 1) {
                      i47 = 0;
                      i48++;
                    }

                    trueCount++;
                  }
                }

                extremePoints_size_idx_1 = b_reshapes[0].f1->size[1];
                extremePoints_size_idx_0 = b_reshapes[0].f1->size[0] +
                  b_reshapes[1].f1->size[0];
                loop_ub = b_reshapes[0].f1->size[0];
                for (i47 = 0; i47 < loop_ub; i47++) {
                  partialTrueCount = b_reshapes[0].f1->size[1];
                  for (i48 = 0; i48 < partialTrueCount; i48++) {
                    b_extremePoints_data[i48 + extremePoints_size_idx_1 * i47] =
                      b_reshapes[0].f1->data[i48 + b_reshapes[0].f1->size[1] *
                      i47];
                  }
                }

                loop_ub = b_reshapes[1].f1->size[0];
                for (i47 = 0; i47 < loop_ub; i47++) {
                  partialTrueCount = b_reshapes[1].f1->size[1];
                  for (i48 = 0; i48 < partialTrueCount; i48++) {
                    b_extremePoints_data[i48 + extremePoints_size_idx_1 * (i47 +
                      b_reshapes[0].f1->size[0])] = b_reshapes[1].f1->data[i48 +
                      b_reshapes[1].f1->size[1] * i47];
                  }
                }

                // 'SmartLoaderEstiamteLocations:317' tempYs = [clustersXs{q}(col,:) clustersYs{q}(col,:)]; 
                loop_ub = r2.data[q].f1->size[1];
                partialTrueCount = clustersXs.data[q].f1->size[1];
                tempXs_size = loop_ub + partialTrueCount;
                for (i47 = 0; i47 < loop_ub; i47++) {
                  tempXs_data[i47] = r2.data[q].f1->data[i47 + r2.data[q]
                    .f1->size[1] * vk];
                }

                for (i47 = 0; i47 < partialTrueCount; i47++) {
                  tempXs_data[i47 + loop_ub] = clustersXs.data[q].f1->data[i47 +
                    clustersXs.data[q].f1->size[1] * vk];
                }

                // 'SmartLoaderEstiamteLocations:318' extremePoints = [extremePoints; tempYs]; 
                if (extremePoints_size_idx_0 != 0) {
                  sizes = 2;
                } else if (tempXs_size != 0) {
                  sizes = (signed char)tempXs_size;
                } else {
                  sizes = 2;
                }

                if (extremePoints_size_idx_0 != 0) {
                  b_input_sizes = (signed char)extremePoints_size_idx_0;
                } else {
                  b_input_sizes = 0;
                }

                m = b_input_sizes;
                partialTrueCount = sizes;
                if ((sizes == 2) && (b_input_sizes == extremePoints_size_idx_0))
                {
                  i47 = c_reshapes[0].f1->size[0] * c_reshapes[0].f1->size[1];
                  c_reshapes[0].f1->size[1] = 2;
                  c_reshapes[0].f1->size[0] = b_input_sizes;
                  emxEnsureCapacity_real32_T(c_reshapes[0].f1, i47);
                  loop_ub = 2 * b_input_sizes;
                  for (i47 = 0; i47 < loop_ub; i47++) {
                    c_reshapes[0].f1->data[i47] = b_extremePoints_data[i47];
                  }
                } else {
                  i47 = 0;
                  i48 = 0;
                  trueCount = 0;
                  i49 = 0;
                  loop_ub = c_reshapes[0].f1->size[0] * c_reshapes[0].f1->size[1];
                  c_reshapes[0].f1->size[1] = sizes;
                  c_reshapes[0].f1->size[0] = b_input_sizes;
                  emxEnsureCapacity_real32_T(c_reshapes[0].f1, loop_ub);
                  for (loop_ub = 0; loop_ub < m * partialTrueCount; loop_ub++) {
                    c_reshapes[0].f1->data[i48 + c_reshapes[0].f1->size[1] * i47]
                      = b_extremePoints_data[i49 + extremePoints_size_idx_1 *
                      trueCount];
                    i47++;
                    trueCount++;
                    if (i47 > c_reshapes[0].f1->size[0] - 1) {
                      i47 = 0;
                      i48++;
                    }

                    if (trueCount > extremePoints_size_idx_0 - 1) {
                      trueCount = 0;
                      i49++;
                    }
                  }
                }

                b_input_sizes = (signed char)(tempXs_size != 0);
                m = b_input_sizes;
                partialTrueCount = sizes;
                if ((sizes == tempXs_size) && (b_input_sizes == 1)) {
                  i47 = c_reshapes[1].f1->size[0] * c_reshapes[1].f1->size[1];
                  c_reshapes[1].f1->size[1] = sizes;
                  c_reshapes[1].f1->size[0] = 1;
                  emxEnsureCapacity_real32_T(c_reshapes[1].f1, i47);
                  loop_ub = sizes;
                  for (i47 = 0; i47 < loop_ub; i47++) {
                    c_reshapes[1].f1->data[i47] = tempXs_data[i47];
                  }
                } else {
                  i47 = 0;
                  i48 = 0;
                  trueCount = 0;
                  i49 = c_reshapes[1].f1->size[0] * c_reshapes[1].f1->size[1];
                  c_reshapes[1].f1->size[1] = sizes;
                  c_reshapes[1].f1->size[0] = b_input_sizes;
                  emxEnsureCapacity_real32_T(c_reshapes[1].f1, i49);
                  for (i49 = 0; i49 < m * partialTrueCount; i49++) {
                    c_reshapes[1].f1->data[i48 + c_reshapes[1].f1->size[1] * i47]
                      = tempXs_data[trueCount];
                    i47++;
                    if (i47 > c_reshapes[1].f1->size[0] - 1) {
                      i47 = 0;
                      i48++;
                    }

                    trueCount++;
                  }
                }

                extremePoints_size[1] = c_reshapes[0].f1->size[1];
                extremePoints_size[0] = c_reshapes[0].f1->size[0] + c_reshapes[1]
                  .f1->size[0];
                loop_ub = c_reshapes[0].f1->size[0];
                for (i47 = 0; i47 < loop_ub; i47++) {
                  partialTrueCount = c_reshapes[0].f1->size[1];
                  for (i48 = 0; i48 < partialTrueCount; i48++) {
                    extremePoints_data[i48 + (i47 << 1)] = c_reshapes[0]
                      .f1->data[i48 + c_reshapes[0].f1->size[1] * i47];
                  }
                }

                loop_ub = c_reshapes[1].f1->size[0];
                for (i47 = 0; i47 < loop_ub; i47++) {
                  partialTrueCount = c_reshapes[1].f1->size[1];
                  for (i48 = 0; i48 < partialTrueCount; i48++) {
                    extremePoints_data[i48 + ((i47 + c_reshapes[0].f1->size[0]) <<
                      1)] = c_reshapes[1].f1->data[i48 + c_reshapes[1].f1->size
                      [1] * i47];
                  }
                }
              }

              // 'SmartLoaderEstiamteLocations:321' if false
              // 'SmartLoaderEstiamteLocations:337' else
              //  2nd method - Least squere circle fit - 2nd order development to finite equations 
              // 'SmartLoaderEstiamteLocations:339' meanXsYs = mean(extremePoints,1); 
              b_mean(extremePoints_data, extremePoints_size,
                     distanceToKmeansClusterMeter);

              // 'SmartLoaderEstiamteLocations:340' us = extremePoints(:,1) - meanXsYs(1); 
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                us_data[i46] = extremePoints_data[i46 << 1] -
                  distanceToKmeansClusterMeter[0];
              }

              // 'SmartLoaderEstiamteLocations:341' vs = extremePoints(:,2) - meanXsYs(2); 
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                vs_data[i46] = extremePoints_data[1 + (i46 << 1)] -
                  distanceToKmeansClusterMeter[1];
              }

              // 'SmartLoaderEstiamteLocations:343' temp_usus = us.*us;
              temp_usus_size[0] = extremePoints_size[0];
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                temp_usus_data[i46] = us_data[i46] * us_data[i46];
              }

              // 'SmartLoaderEstiamteLocations:344' s_uu = sum(temp_usus);
              // 'SmartLoaderEstiamteLocations:345' s_uuu = sum(temp_usus.*us);
              // 'SmartLoaderEstiamteLocations:346' temp_vsvs = vs.*vs;
              temp_vsvs_size[0] = extremePoints_size[0];
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                temp_vsvs_data[i46] = vs_data[i46] * vs_data[i46];
              }

              // 'SmartLoaderEstiamteLocations:347' s_vv = sum(temp_vsvs);
              // 'SmartLoaderEstiamteLocations:348' s_vvv = sum(vs.*temp_vsvs);
              // 'SmartLoaderEstiamteLocations:350' temp_vsus = vs.*us;
              temp_vsus_size[0] = extremePoints_size[0];
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                temp_vsus_data[i46] = vs_data[i46] * us_data[i46];
              }

              // 'SmartLoaderEstiamteLocations:351' s_vu = sum(temp_vsus);
              pcFirstDistanceToPlane = c_sum(temp_vsus_data, temp_vsus_size);

              // 'SmartLoaderEstiamteLocations:352' s_uvv = sum(temp_vsus.*vs);
              // 'SmartLoaderEstiamteLocations:353' s_vuu = sum(temp_vsus.*us);
              // 'SmartLoaderEstiamteLocations:355' A = [s_uu s_vu; s_vu s_vv];
              // 'SmartLoaderEstiamteLocations:356' b = 0.5 * [s_uuu + s_uvv; s_vvv + s_vuu]; 
              // 'SmartLoaderEstiamteLocations:358' cxyEst = inv(A) * b;
              // 'SmartLoaderEstiamteLocations:359' cxyEst = cxyEst + meanXsYs'; 
              fv0[0] = c_sum(temp_usus_data, temp_usus_size);
              fv0[1] = pcFirstDistanceToPlane;
              fv0[2] = pcFirstDistanceToPlane;
              fv0[3] = c_sum(temp_vsvs_data, temp_vsvs_size);
              b_inv(fv0, fv1);
              b_temp_usus_size[0] = extremePoints_size[0];
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                b_temp_usus_data[i46] = temp_usus_data[i46] * us_data[i46];
              }

              b_temp_vsus_size[0] = extremePoints_size[0];
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                b_temp_vsus_data[i46] = temp_vsus_data[i46] * vs_data[i46];
              }

              vs_size[0] = extremePoints_size[0];
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                b_vs_data[i46] = vs_data[i46] * temp_vsvs_data[i46];
              }

              c_temp_vsus_size[0] = extremePoints_size[0];
              loop_ub = extremePoints_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                temp_usus_data[i46] = temp_vsus_data[i46] * us_data[i46];
              }

              pcFirstDistanceToPlane = 0.5F * (c_sum(b_temp_usus_data,
                b_temp_usus_size) + c_sum(b_temp_vsus_data, b_temp_vsus_size));
              pcSecondDistanceToPlane = 0.5F * (c_sum(b_vs_data, vs_size) +
                c_sum(temp_usus_data, c_temp_vsus_size));
              r[0] = (pcFirstDistanceToPlane * fv1[0] + pcSecondDistanceToPlane *
                      fv1[1]) + distanceToKmeansClusterMeter[0];
              r[1] = (pcFirstDistanceToPlane * fv1[2] + pcSecondDistanceToPlane *
                      fv1[3]) + distanceToKmeansClusterMeter[1];

              // 'SmartLoaderEstiamteLocations:362' if coder.target('Matlab') &&  false 
              //     %% Calculate the mean squre error for the estiamted model
              // 'SmartLoaderEstiamteLocations:382' modelErr1 = bsxfun(@minus, extremePoints, cxyEst'); 
              bsxfun(extremePoints_data, extremePoints_size, r,
                     SD->u6.f20.kmeansDistanceMat_data, input_sizes);
              modelErr1_size[0] = input_sizes[0];
              loop_ub = input_sizes[1] * input_sizes[0];
              if (0 <= loop_ub - 1) {
                memcpy(&modelErr1_data[0], &SD->u6.f20.kmeansDistanceMat_data[0],
                       (unsigned int)(loop_ub * (int)sizeof(float)));
              }

              // 'SmartLoaderEstiamteLocations:383' modelErr1 = modelErr1 .* modelErr1; 
              modelErr1_size[1] = 2;
              loop_ub = (input_sizes[0] << 1) - 1;
              for (i46 = 0; i46 <= loop_ub; i46++) {
                modelErr1_data[i46] *= modelErr1_data[i46];
              }

              // 'SmartLoaderEstiamteLocations:384' modelErr1 = sqrt(sum(modelErr1, 2)) - ((configParams.loaderReflectorDiameterMeter/2)); 
              b_sum(modelErr1_data, modelErr1_size, SD->u6.f20.x_data, x_size);
              us_size[0] = x_size[0];
              if (0 <= x_size[0] - 1) {
                memcpy(&us_data[0], &SD->u6.f20.x_data[0], (unsigned int)
                       (x_size[0] * (int)sizeof(float)));
              }

              b_sqrt(us_data, us_size);
              v_idx_0 = configParams->loaderReflectorDiameterMeter / 2.0;
              loop_ub = us_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                us_data[i46] -= (float)v_idx_0;
              }

              // 'SmartLoaderEstiamteLocations:385' mse = sum(modelErr1 .* modelErr1) / size(extremePoints,1); 
              // 'SmartLoaderEstiamteLocations:387' if mse > configParams.loaderReflectorDiameterMeter 
              b_us_size[0] = us_size[0];
              loop_ub = us_size[0];
              for (i46 = 0; i46 < loop_ub; i46++) {
                temp_usus_data[i46] = us_data[i46] * us_data[i46];
              }

              if (c_sum(temp_usus_data, b_us_size) / (float)extremePoints_size[0]
                  > configParams->loaderReflectorDiameterMeter) {
                // 'SmartLoaderEstiamteLocations:388' if coder.target('Matlab')
                // 'SmartLoaderEstiamteLocations:391' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eFailedLoaderLocation; 
                smartLoaderStruct->status =
                  PerceptionSmartLoaderReturnValue_eFailedLoaderLocation;
              } else {
                // 'SmartLoaderEstiamteLocations:395' smartLoaderStruct.loaderLoc = double([cxyEst; zMedian]); 
                smartLoaderStruct->loaderLoc[0] = r[0];
                smartLoaderStruct->loaderLoc[1] = r[1];
                smartLoaderStruct->loaderLoc[2] = pcFirstRange_idx_0;

                //  Ensure minimal distance from the loader location to the ground plane 
                //  The loader height is around 27cm, we can ensure at least half of this size from the ground plane 
                // 'SmartLoaderEstiamteLocations:400' [loaderLocToPlaneDistance, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, smartLoaderStruct.loaderLoc'); 
                b_CalcPlaneToPointDistance(configParams->planeModelParameters,
                  smartLoaderStruct->loaderLoc, &singleReflectorRangeLimitMeter,
                  &empty_non_axis_sizes);

                // 'SmartLoaderEstiamteLocations:400' ~
                // 'SmartLoaderEstiamteLocations:401' if loaderLocToPlaneDistance < configParams.minimumDistanceFromLoaderToPlaneMeter 
                if (singleReflectorRangeLimitMeter <
                    configParams->minimumDistanceFromLoaderToPlaneMeter) {
                  // 'SmartLoaderEstiamteLocations:402' smartLoaderStruct.loaderLocStatus = false; 
                  smartLoaderStruct->loaderLocStatus = false;

                  // 'SmartLoaderEstiamteLocations:404' if coder.target('Matlab') 
                  // 'SmartLoaderEstiamteLocations:407' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eFailedLoaderLocation; 
                  smartLoaderStruct->status =
                    PerceptionSmartLoaderReturnValue_eFailedLoaderLocation;
                } else {
                  // 'SmartLoaderEstiamteLocations:411' smartLoaderStruct.loaderLocStatus = true; 
                  smartLoaderStruct->loaderLocStatus = true;

                  //  Calcualte the yaw angle of the loader
                  //  Get all the poitns that are close to the loader
                  // 'SmartLoaderEstiamteLocations:415' if smartLoaderStruct.loaderLocStatus && smartLoaderStruct.shovelLocStatus 
                  if (smartLoaderStruct->shovelLocStatus) {
                    //  figure, PlotPointCloud(ptCloudSenceXyz);
                    // 'SmartLoaderEstiamteLocations:418' ptCloudSenceXyzFromLoaderLoc = bsxfun(@minus, ptCloudSenceXyz(:,1:2), cast(smartLoaderStruct.loaderLoc(1:2)', 'like', ptCloudSenceXyz)); 
                    // 'SmartLoaderEstiamteLocations:419' distanecToLoader = vecnorm(ptCloudSenceXyzFromLoaderLoc,2,2); 
                    // 'SmartLoaderEstiamteLocations:420' isInsideLoaderReflector = distanecToLoader < configParams.loaderCenterToBackwardPointMeter + configParams.loaderWhiteHatMeter / 2; 
                    loop_ub = ptCloudSenceXyz_size[0];
                    b_ptCloudSenceXyz_size[1] = 2;
                    b_ptCloudSenceXyz_size[0] = ptCloudSenceXyz_size[0];
                    for (i46 = 0; i46 < loop_ub; i46++) {
                      m = i46 << 1;
                      SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[m] =
                        ptCloudSenceXyz_data[3 * i46];
                      SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[1 + m] =
                        ptCloudSenceXyz_data[1 + 3 * i46];
                    }

                    r[0] = (float)smartLoaderStruct->loaderLoc[0];
                    r[1] = (float)smartLoaderStruct->loaderLoc[1];
                    bsxfun(SD->u6.f20.c_ptCloudSenceReflectorsXyz_data,
                           b_ptCloudSenceXyz_size, r,
                           SD->u6.f20.kmeansDistanceMat_data, input_sizes);
                    b_vecnorm(SD->u6.f20.kmeansDistanceMat_data, input_sizes,
                              SD->u6.f20.x_data, x_size);
                    singleReflectorRangeLimitMeter =
                      configParams->loaderCenterToBackwardPointMeter +
                      configParams->loaderWhiteHatMeter / 2.0;
                    loop_ub = x_size[0];
                    for (i46 = 0; i46 < loop_ub; i46++) {
                      SD->u6.f20.ptCloudSenceReflectorsInd_data[i46] =
                        (SD->u6.f20.x_data[i46] < singleReflectorRangeLimitMeter);
                    }

                    // 'SmartLoaderEstiamteLocations:422' loaderXyz = ptCloudSenceXyz(isInsideLoaderReflector,:); 
                    m = x_size[0] - 1;
                    trueCount = 0;
                    for (loop_ub = 0; loop_ub <= m; loop_ub++) {
                      if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]) {
                        trueCount++;
                      }
                    }

                    partialTrueCount = 0;
                    for (loop_ub = 0; loop_ub <= m; loop_ub++) {
                      if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]) {
                        SD->u6.f20.f_tmp_data[partialTrueCount] = loop_ub + 1;
                        partialTrueCount++;
                      }
                    }

                    ptCloudSenceReflectorsXyz_size[1] = 3;
                    ptCloudSenceReflectorsXyz_size[0] = trueCount;
                    for (i46 = 0; i46 < trueCount; i46++) {
                      m = 3 * (SD->u6.f20.f_tmp_data[i46] - 1);
                      SD->u6.f20.ptCloudSenceReflectorsXyz_data[3 * i46] =
                        ptCloudSenceXyz_data[m];
                      SD->u6.f20.ptCloudSenceReflectorsXyz_data[1 + 3 * i46] =
                        ptCloudSenceXyz_data[1 + m];
                      SD->u6.f20.ptCloudSenceReflectorsXyz_data[2 + 3 * i46] =
                        ptCloudSenceXyz_data[2 + m];
                    }

                    //  figure, PlotPointCloud(loaderXyz);
                    //  Get all points higher than 10cm from the ground
                    // 'SmartLoaderEstiamteLocations:426' [loaderXyzPlaneDistance, ~] = CalcPlaneToPointDistance(configParams.planeModelParameters, loaderXyz); 
                    c_CalcPlaneToPointDistance(SD,
                      configParams->planeModelParameters,
                      SD->u6.f20.ptCloudSenceReflectorsXyz_data,
                      ptCloudSenceReflectorsXyz_size, SD->u6.f20.x_data, x_size,
                      SD->u6.f20.ptCloudSenceReflectorsInd_data, kmeansIdx_size);

                    // 'SmartLoaderEstiamteLocations:426' ~
                    // 'SmartLoaderEstiamteLocations:428' coder.varsize('loaderXyzFiltered', [SmartLoaderCompilationConstants.MaxPointCloudSize 3], [1 0]); 
                    // 'SmartLoaderEstiamteLocations:429' loaderXyzFiltered = loaderXyz(loaderXyzPlaneDistance > configParams.maxDistanceFromThePlaneForLoaderYawCalculation,:); 
                    loop_ub = x_size[0];
                    for (i46 = 0; i46 < loop_ub; i46++) {
                      SD->u6.f20.ptCloudSenceReflectorsInd_data[i46] =
                        (SD->u6.f20.x_data[i46] >
                         configParams->maxDistanceFromThePlaneForLoaderYawCalculation);
                    }

                    m = x_size[0] - 1;
                    trueCount = 0;
                    for (loop_ub = 0; loop_ub <= m; loop_ub++) {
                      if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]) {
                        trueCount++;
                      }
                    }

                    partialTrueCount = 0;
                    for (loop_ub = 0; loop_ub <= m; loop_ub++) {
                      if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]) {
                        SD->u6.f20.g_tmp_data[partialTrueCount] = loop_ub + 1;
                        partialTrueCount++;
                      }
                    }

                    for (i46 = 0; i46 < trueCount; i46++) {
                      m = 3 * (SD->u6.f20.g_tmp_data[i46] - 1);
                      SD->u6.f20.pcFirstXyz_data[3 * i46] =
                        SD->u6.f20.ptCloudSenceReflectorsXyz_data[m];
                      SD->u6.f20.pcFirstXyz_data[1 + 3 * i46] =
                        SD->u6.f20.ptCloudSenceReflectorsXyz_data[1 + m];
                      SD->u6.f20.pcFirstXyz_data[2 + 3 * i46] =
                        SD->u6.f20.ptCloudSenceReflectorsXyz_data[2 + m];
                    }

                    // 'SmartLoaderEstiamteLocations:430' if false
                    //  In case we have the previous loader yaw angle, use it in order to mask the loader points - Othewise - use all the points (loader and shovel) 
                    // 'SmartLoaderEstiamteLocations:436' if ~isempty(SmartLoaderGlobal.smartLoaderStructHistory) && SmartLoaderGlobal.smartLoaderStructHistory(end).loaderYawAngleStatus && ... 
                    // 'SmartLoaderEstiamteLocations:437'             configParams.timeTagMs - SmartLoaderGlobal.loaderTimeTatHistoryMs(end) < configParams.maximumTimeTagDiffMs 
                    if ((SD->pd->
                         SmartLoaderGlobal.smartLoaderStructHistory.size[0] != 0)
                        && SD->
                        pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                        [SD->pd->
                        SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                        loaderYawAngleStatus && (configParams->timeTagMs -
                         SD->pd->
                         SmartLoaderGlobal.loaderTimeTatHistoryMs.data[SD->
                         pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] -
                         1] < configParams->maximumTimeTagDiffMs)) {
                      //  Filter the point according to the previous yaw angle
                      //  determine which points reside inside the loader reflector  
                      // 'SmartLoaderEstiamteLocations:442' loaderXyzFilteredStartFromLoaderLoc = bsxfun(@minus, loaderXyzFiltered(:,1:2), cast(smartLoaderStruct.loaderLoc(1:2)', 'like', ptCloudSenceXyz)); 
                      // 'SmartLoaderEstiamteLocations:443' distanceFromLoaderToXYZ = vecnorm(loaderXyzFilteredStartFromLoaderLoc,2,2); 
                      // 'SmartLoaderEstiamteLocations:444' isInsideLoaderReflector = distanceFromLoaderToXYZ < (configParams.locationsBiasMeter + configParams.loaderReflectorDiameterMeter / 2); 
                      b_pcFirstXyz_size[1] = 2;
                      b_pcFirstXyz_size[0] = trueCount;
                      for (i46 = 0; i46 < trueCount; i46++) {
                        m = i46 << 1;
                        SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[m] =
                          SD->u6.f20.pcFirstXyz_data[3 * i46];
                        SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[1 + m] =
                          SD->u6.f20.pcFirstXyz_data[1 + 3 * i46];
                      }

                      r[0] = (float)smartLoaderStruct->loaderLoc[0];
                      r[1] = (float)smartLoaderStruct->loaderLoc[1];
                      bsxfun(SD->u6.f20.c_ptCloudSenceReflectorsXyz_data,
                             b_pcFirstXyz_size, r,
                             SD->u6.f20.kmeansDistanceMat_data, input_sizes);
                      b_vecnorm(SD->u6.f20.kmeansDistanceMat_data, input_sizes,
                                SD->u6.f20.x_data, x_size);
                      singleReflectorRangeLimitMeter =
                        configParams->locationsBiasMeter + v_idx_0;
                      loop_ub = x_size[0];
                      for (i46 = 0; i46 < loop_ub; i46++) {
                        SD->u6.f20.ptCloudSenceReflectorsInd_data[i46] =
                          (SD->u6.f20.x_data[i46] <
                           singleReflectorRangeLimitMeter);
                      }

                      // 'SmartLoaderEstiamteLocations:446' if false
                      //  Find the bounding box location of the loader
                      // 'SmartLoaderEstiamteLocations:452' loaderToShovel2Dvec = [1 0]; 
                      // 'SmartLoaderEstiamteLocations:453' currentAngleDeg = SmartLoaderGlobal.smartLoaderStructHistory(end).loaderYawAngleDeg; 
                      // 'SmartLoaderEstiamteLocations:454' rotMatCCW = [cosd(currentAngleDeg) sind(currentAngleDeg); -sind(currentAngleDeg) cosd(currentAngleDeg)]; 
                      // 'SmartLoaderEstiamteLocations:455' loaderToShovel2Dvec = loaderToShovel2Dvec * rotMatCCW; 
                      d5 = SD->
                        pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                        [SD->pd->
                        SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                        loaderYawAngleDeg;
                      b_cosd(&d5);
                      d6 = SD->
                        pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                        [SD->pd->
                        SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                        loaderYawAngleDeg;
                      b_sind(&d6);
                      v_idx_0 = SD->
                        pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                        [SD->pd->
                        SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                        loaderYawAngleDeg;
                      b_sind(&v_idx_0);
                      v_idx_0 = SD->
                        pd->SmartLoaderGlobal.smartLoaderStructHistory.data
                        [SD->pd->
                        SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
                        loaderYawAngleDeg;
                      b_cosd(&v_idx_0);

                      // 'SmartLoaderEstiamteLocations:456' if false
                      // 'SmartLoaderEstiamteLocations:462' currentAngleDeg = 90; 
                      // 'SmartLoaderEstiamteLocations:463' rotMatCCW = [cosd(currentAngleDeg) sind(currentAngleDeg); -sind(currentAngleDeg) cosd(currentAngleDeg)]; 
                      // 'SmartLoaderEstiamteLocations:464' p1 = smartLoaderStruct.loaderLoc(1:2) + (configParams.locationsBiasMeter + configParams.loaderWidthMeter/2) * (loaderToShovel2Dvec * rotMatCCW)'; 
                      singleReflectorRangeLimitMeter =
                        configParams->loaderWidthMeter / 2.0;
                      a_tmp = configParams->locationsBiasMeter +
                        singleReflectorRangeLimitMeter;

                      // 'SmartLoaderEstiamteLocations:466' currentAngleDeg = -90; 
                      // 'SmartLoaderEstiamteLocations:467' rotMatCCW = [cosd(currentAngleDeg) sind(currentAngleDeg); -sind(currentAngleDeg) cosd(currentAngleDeg)]; 
                      // 'SmartLoaderEstiamteLocations:468' p4 = smartLoaderStruct.loaderLoc(1:2) + (configParams.locationsBiasMeter + configParams.loaderWidthMeter/2) * (loaderToShovel2Dvec * rotMatCCW)'; 
                      // 'SmartLoaderEstiamteLocations:470' loaderToLeftBackOftheLoaderAng = atan2d(configParams.loaderCenterToBackwardPointMeter, configParams.loaderWidthMeter/2); 
                      v_idx_1 = 57.295779513082323 * atan2
                        (configParams->loaderCenterToBackwardPointMeter,
                         singleReflectorRangeLimitMeter);

                      // 'SmartLoaderEstiamteLocations:471' loaderToLeftBackOftheLoaderDistanceMeter = configParams.locationsBiasMeter + norm( [configParams.loaderWidthMeter/2; configParams.loaderCenterToBackwardPointMeter]); 
                      p4[0] = singleReflectorRangeLimitMeter;
                      p4[1] = configParams->loaderCenterToBackwardPointMeter;
                      loaderToLeftBackOftheLoaderDistanceMeter =
                        configParams->locationsBiasMeter + d_norm(p4);

                      // 'SmartLoaderEstiamteLocations:473' currentAngleDeg = loaderToLeftBackOftheLoaderAng + 90; 
                      // 'SmartLoaderEstiamteLocations:474' rotMatCCW = [cosd(currentAngleDeg) sind(currentAngleDeg); -sind(currentAngleDeg) cosd(currentAngleDeg)]; 
                      // 'SmartLoaderEstiamteLocations:475' p2 = smartLoaderStruct.loaderLoc(1:2) + loaderToLeftBackOftheLoaderDistanceMeter * (loaderToShovel2Dvec * rotMatCCW)'; 
                      v_idx_0 = v_idx_1 + 90.0;
                      b_cosd(&v_idx_0);
                      d7 = v_idx_1 + 90.0;
                      b_sind(&d7);
                      d8 = v_idx_1 + 90.0;
                      b_sind(&d8);
                      d9 = v_idx_1 + 90.0;
                      b_cosd(&d9);

                      // 'SmartLoaderEstiamteLocations:477' currentAngleDeg = currentAngleDeg * -1; 
                      // 'SmartLoaderEstiamteLocations:478' rotMatCCW = [cosd(currentAngleDeg) sind(currentAngleDeg); -sind(currentAngleDeg) cosd(currentAngleDeg)]; 
                      // 'SmartLoaderEstiamteLocations:479' p3 = smartLoaderStruct.loaderLoc(1:2) + loaderToLeftBackOftheLoaderDistanceMeter * (loaderToShovel2Dvec * rotMatCCW)'; 
                      d10 = -(v_idx_1 + 90.0);
                      b_cosd(&d10);
                      d11 = -(v_idx_1 + 90.0);
                      b_sind(&d11);
                      d12 = -(v_idx_1 + 90.0);
                      b_sind(&d12);
                      singleReflectorRangeLimitMeter = -(v_idx_1 + 90.0);
                      b_cosd(&singleReflectorRangeLimitMeter);

                      // 'SmartLoaderEstiamteLocations:481' loaderBBxs = [p1(1);p2(1);p3(1);p4(1)]; 
                      // 'SmartLoaderEstiamteLocations:482' loaderBBys = [p1(2);p2(2);p3(2);p4(2)]; 
                      // 'SmartLoaderEstiamteLocations:484' if false
                      // 'SmartLoaderEstiamteLocations:507' isInsidePolygon = inpolygon(loaderXyzFiltered(:,1), loaderXyzFiltered(:,2), loaderBBxs, loaderBBys); 
                      c_pcFirstXyz_size[0] = trueCount;
                      for (i46 = 0; i46 < trueCount; i46++) {
                        SD->u6.f20.x_data[i46] = SD->u6.f20.pcFirstXyz_data[3 *
                          i46];
                      }

                      for (i46 = 0; i46 < trueCount; i46++) {
                        SD->u6.f20.tmp_data[i46] = SD->u6.f20.pcFirstXyz_data[1
                          + 3 * i46];
                      }

                      p1[0] = smartLoaderStruct->loaderLoc[0] + a_tmp * -d6;
                      p1[1] = smartLoaderStruct->loaderLoc[0] +
                        loaderToLeftBackOftheLoaderDistanceMeter * (v_idx_0 * d5
                        + -d8 * d6);
                      p1[2] = smartLoaderStruct->loaderLoc[0] +
                        loaderToLeftBackOftheLoaderDistanceMeter * (d10 * d5 +
                        -d12 * d6);
                      p1[3] = smartLoaderStruct->loaderLoc[0] + a_tmp * d6;
                      b_p1[0] = smartLoaderStruct->loaderLoc[1] + a_tmp * d5;
                      b_p1[1] = smartLoaderStruct->loaderLoc[1] +
                        loaderToLeftBackOftheLoaderDistanceMeter * (d7 * d5 + d9
                        * d6);
                      b_p1[2] = smartLoaderStruct->loaderLoc[1] +
                        loaderToLeftBackOftheLoaderDistanceMeter * (d11 * d5 +
                        singleReflectorRangeLimitMeter * d6);
                      b_p1[3] = smartLoaderStruct->loaderLoc[1] + a_tmp * -d5;
                      inpolygon(SD->u6.f20.x_data, c_pcFirstXyz_size,
                                SD->u6.f20.tmp_data, p1, b_p1,
                                SD->u6.f20.isInsidePolygon_data, kmeansIdx_size);

                      //  loaderXyzFilteredInsidePoly = loaderXyzFiltered(isInsidePolygon,:); figure, PlotPointCloud(loaderXyzFilteredInsidePoly); 
                      // 'SmartLoaderEstiamteLocations:510' isInsideLoaderReflectorOrPolygon = isInsideLoaderReflector | isInsidePolygon; 
                      // 'SmartLoaderEstiamteLocations:511' sum_isInsideLoaderReflectorOrPolygon = sum(isInsideLoaderReflectorOrPolygon); 
                      ptCloudSenceReflectorsInd_size[0] = x_size[0];
                      loop_ub = x_size[0];
                      for (i46 = 0; i46 < loop_ub; i46++) {
                        SD->u6.f20.b_ptCloudSenceReflectorsInd_data[i46] =
                          (SD->u6.f20.ptCloudSenceReflectorsInd_data[i46] ||
                           SD->u6.f20.isInsidePolygon_data[i46]);
                      }

                      singleReflectorRangeLimitMeter = e_sum
                        (SD->u6.f20.b_ptCloudSenceReflectorsInd_data,
                         ptCloudSenceReflectorsInd_size);

                      // 'SmartLoaderEstiamteLocations:513' if sum_isInsideLoaderReflectorOrPolygon / size(loaderXyzFiltered,1) > configParams.yawEstimationMinPercentageOfPointsInLoaderBody && ... 
                      // 'SmartLoaderEstiamteLocations:514'                  sum_isInsideLoaderReflectorOrPolygon > configParams.yawEstimationMinNumPointsInLoaderBody 
                      if ((singleReflectorRangeLimitMeter / (double)trueCount >
                           configParams->yawEstimationMinPercentageOfPointsInLoaderBody)
                          && (singleReflectorRangeLimitMeter >
                              configParams->yawEstimationMinNumPointsInLoaderBody))
                      {
                        // 'SmartLoaderEstiamteLocations:516' loaderXyzFilteredInsidePolyAndInsideLoaderBB = loaderXyzFiltered(isInsideLoaderReflectorOrPolygon,:); 
                        //  figure, PlotPointCloud(loaderXyzFilteredInsidePolyAndInsideLoaderBB);  
                        // 'SmartLoaderEstiamteLocations:518' [V,D] = SmartLoaderCalcEigen(loaderXyzFilteredInsidePolyAndInsideLoaderBB(:,1:2)); 
                        m = x_size[0] - 1;
                        trueCount = 0;
                        for (loop_ub = 0; loop_ub <= m; loop_ub++) {
                          if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]
                              || SD->u6.f20.isInsidePolygon_data[loop_ub]) {
                            trueCount++;
                          }
                        }

                        partialTrueCount = 0;
                        for (loop_ub = 0; loop_ub <= m; loop_ub++) {
                          if (SD->u6.f20.ptCloudSenceReflectorsInd_data[loop_ub]
                              || SD->u6.f20.isInsidePolygon_data[loop_ub]) {
                            SD->u6.f20.h_tmp_data[partialTrueCount] = loop_ub +
                              1;
                            partialTrueCount++;
                          }
                        }

                        for (i46 = 0; i46 < trueCount; i46++) {
                          m = 3 * (SD->u6.f20.h_tmp_data[i46] - 1);
                          SD->u6.f20.pcSecondXyz_data[3 * i46] =
                            SD->u6.f20.pcFirstXyz_data[m];
                          SD->u6.f20.pcSecondXyz_data[1 + 3 * i46] =
                            SD->u6.f20.pcFirstXyz_data[1 + m];
                          SD->u6.f20.pcSecondXyz_data[2 + 3 * i46] =
                            SD->u6.f20.pcFirstXyz_data[2 + m];
                        }

                        d_pcFirstXyz_size[1] = 2;
                        d_pcFirstXyz_size[0] = trueCount;
                        for (i46 = 0; i46 < trueCount; i46++) {
                          m = i46 << 1;
                          SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[m] =
                            SD->u6.f20.pcSecondXyz_data[3 * i46];
                          SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[1 + m] =
                            SD->u6.f20.pcSecondXyz_data[1 + 3 * i46];
                        }

                        SmartLoaderCalcEigen(SD,
                                             SD->u6.f20.c_ptCloudSenceReflectorsXyz_data,
                                             d_pcFirstXyz_size, V, D);
                      } else {
                        // 'SmartLoaderEstiamteLocations:519' else
                        // 'SmartLoaderEstiamteLocations:520' [V,D] = SmartLoaderCalcEigen(loaderXyzFiltered(:,1:2)); 
                        d_pcFirstXyz_size[1] = 2;
                        d_pcFirstXyz_size[0] = trueCount;
                        for (i46 = 0; i46 < trueCount; i46++) {
                          m = i46 << 1;
                          SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[m] =
                            SD->u6.f20.pcFirstXyz_data[3 * i46];
                          SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[1 + m] =
                            SD->u6.f20.pcFirstXyz_data[1 + 3 * i46];
                        }

                        SmartLoaderCalcEigen(SD,
                                             SD->u6.f20.c_ptCloudSenceReflectorsXyz_data,
                                             d_pcFirstXyz_size, V, D);
                      }
                    } else {
                      // 'SmartLoaderEstiamteLocations:522' else
                      // 'SmartLoaderEstiamteLocations:523' [V,D] = SmartLoaderCalcEigen(loaderXyzFiltered(:,1:2)); 
                      b_pcFirstXyz_size[1] = 2;
                      b_pcFirstXyz_size[0] = trueCount;
                      for (i46 = 0; i46 < trueCount; i46++) {
                        m = i46 << 1;
                        SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[m] =
                          SD->u6.f20.pcFirstXyz_data[3 * i46];
                        SD->u6.f20.c_ptCloudSenceReflectorsXyz_data[1 + m] =
                          SD->u6.f20.pcFirstXyz_data[1 + 3 * i46];
                      }

                      SmartLoaderCalcEigen(SD,
                                           SD->u6.f20.c_ptCloudSenceReflectorsXyz_data,
                                           b_pcFirstXyz_size, V, D);
                    }

                    // 'SmartLoaderEstiamteLocations:525' [~,ind] = sort(diag(D)); 
                    diag(D, largestEigenVec);
                    b_sort(largestEigenVec, input_sizes);

                    // 'SmartLoaderEstiamteLocations:525' ~
                    //  Plot everything
                    //  Pick the largest eigen vector
                    // 'SmartLoaderEstiamteLocations:529' largestEigenVec = V(:,ind(end)); 
                    //  determine whether or not the vector reside on the shovel side or the opposite side 
                    // 'SmartLoaderEstiamteLocations:531' shovelToLoader2DVec = smartLoaderStruct.shovelLoc(1:2) - smartLoaderStruct.loaderLoc(1:2); 
                    largestEigenVec[0] = V[input_sizes[1] - 1];
                    p4[0] = smartLoaderStruct->shovelLoc[0] -
                      smartLoaderStruct->loaderLoc[0];
                    largestEigenVec[1] = V[input_sizes[1] + 1];
                    p4[1] = smartLoaderStruct->shovelLoc[1] -
                      smartLoaderStruct->loaderLoc[1];

                    // 'SmartLoaderEstiamteLocations:532' shovelToLoader2DVec = shovelToLoader2DVec / norm(shovelToLoader2DVec); 
                    d5 = d_norm(p4);

                    // 'SmartLoaderEstiamteLocations:533' if dot(largestEigenVec, shovelToLoader2DVec) < 0 
                    b_V[0] = V[input_sizes[1] - 1];
                    p4[0] = (smartLoaderStruct->shovelLoc[0] -
                             smartLoaderStruct->loaderLoc[0]) / d5;
                    b_V[1] = V[input_sizes[1] + 1];
                    p4[1] = (smartLoaderStruct->shovelLoc[1] -
                             smartLoaderStruct->loaderLoc[1]) / d5;
                    if ((dot(b_V, p4)).re < 0.0) {
                      // 'SmartLoaderEstiamteLocations:534' largestEigenVec = largestEigenVec * -1; 
                      largestEigenVec[0].re = -V[input_sizes[1] - 1].re;
                      largestEigenVec[1].re = -V[input_sizes[1] + 1].re;
                    }

                    // 'SmartLoaderEstiamteLocations:536' smartLoaderStruct.loaderYawAngleDeg = atan2d(real(largestEigenVec(2,1)),real(largestEigenVec(1,1))); 
                    smartLoaderStruct->loaderYawAngleDeg = 57.295779513082323 *
                      atan2(largestEigenVec[1].re, largestEigenVec[0].re);

                    // 'SmartLoaderEstiamteLocations:537' smartLoaderStruct.loaderYawAngleStatus = true; 
                    smartLoaderStruct->loaderYawAngleStatus = true;

                    // 'SmartLoaderEstiamteLocations:539' if false
                    // 'SmartLoaderEstiamteLocations:551' if false
                    //  From these two angles calculate the loader to shovel angle 
                    // 'SmartLoaderEstiamteLocations:566' loaderToShovel2Dvec = smartLoaderStruct.shovelLoc(1:2) - smartLoaderStruct.loaderLoc(1:2); 
                    p4[0] = smartLoaderStruct->shovelLoc[0] -
                      smartLoaderStruct->loaderLoc[0];
                    p4[1] = smartLoaderStruct->shovelLoc[1] -
                      smartLoaderStruct->loaderLoc[1];

                    // 'SmartLoaderEstiamteLocations:567' loaderToShovel2Dvec = loaderToShovel2Dvec / norm(loaderToShovel2Dvec); 
                    d5 = d_norm(p4);
                    p4[0] = (smartLoaderStruct->shovelLoc[0] -
                             smartLoaderStruct->loaderLoc[0]) / d5;
                    p4[1] = (smartLoaderStruct->shovelLoc[1] -
                             smartLoaderStruct->loaderLoc[1]) / d5;

                    // 'SmartLoaderEstiamteLocations:568' loader2Dvec = ([1 0] * [cosd(smartLoaderStruct.loaderYawAngleDeg) sind(smartLoaderStruct.loaderYawAngleDeg); -sind(smartLoaderStruct.loaderYawAngleDeg) cosd(smartLoaderStruct.loaderYawAngleDeg)])'; 
                    d5 = smartLoaderStruct->loaderYawAngleDeg;
                    b_cosd(&d5);
                    d6 = smartLoaderStruct->loaderYawAngleDeg;
                    b_sind(&d6);
                    v_idx_0 = smartLoaderStruct->loaderYawAngleDeg;
                    b_sind(&v_idx_0);
                    v_idx_0 = smartLoaderStruct->loaderYawAngleDeg;
                    b_cosd(&v_idx_0);

                    //  there is no need to normalize the loader2Dvec vector becuase rotation matrix is orthonormalize 
                    //  This equations relate to right hand coordinate system
                    // 'SmartLoaderEstiamteLocations:572' tempDot = loaderToShovel2Dvec(1)*loader2Dvec(1) + loaderToShovel2Dvec(2)*loader2Dvec(2); 
                    // 'SmartLoaderEstiamteLocations:573' tempDet = loaderToShovel2Dvec(2)*loader2Dvec(1) - loaderToShovel2Dvec(1)*loader2Dvec(2); 
                    // 'SmartLoaderEstiamteLocations:574' smartLoaderStruct.loaderToShovelYawAngleDeg = atan2d(tempDet, tempDot); 
                    smartLoaderStruct->loaderToShovelYawAngleDeg =
                      57.295779513082323 * atan2(p4[1] * d5 - p4[0] * d6, p4[0] *
                      d5 + p4[1] * d6);

                    // 'SmartLoaderEstiamteLocations:575' smartLoaderStruct.loaderToShovelYawAngleDegStatus = true; 
                    smartLoaderStruct->loaderToShovelYawAngleDegStatus = true;
                  }

                  //  Determine the bounding box of the loader
                  // 'SmartLoaderEstiamteLocations:580' if coder.target('Matlab') && false 
                }
              }
            }
          }
        } else {
          // 'SmartLoaderEstiamteLocations:252' if coder.target('Matlab')
        }
      }

      emxFree_real32_T(&d_reshapes);
      emxFreeMatrix_cell_wrap_4(c_reshapes);
      emxFreeMatrix_cell_wrap_4(b_reshapes);
      emxFree_real32_T(&varargin_1);
      emxFreeMatrix_cell_wrap_4(reshapes);
      emxFree_cell_wrap_4_64x1(&r2);
      emxFree_cell_wrap_4_64x1(&clustersYs);
      emxFree_cell_wrap_4_64x1(&clustersXs);
      emxFree_real32_T(&Z);
      emxFree_real32_T(&pdistOutput);
    }
  }
}

//
// function [SmartLoaderGlobalStruct] = SmartLoaderGlobalInit
// Arguments    : PerceptionSmartLoaderStackData *SD
// Return Type  : void
//
static void SmartLoaderGlobalInit(PerceptionSmartLoaderStackData *SD)
{
  //  % boolean represent whether or not the smart loader global has been initialized 
  //  % N vector that hold the history of the loader struct output
  //  % N*1 vector that hold the history of the loader location's time tag
  // 'SmartLoaderGlobalInit:8' SmartLoaderGlobalStruct = struct('isInitialized', true, ... % boolean represent whether or not the smart loader global has been initialized 
  // 'SmartLoaderGlobalInit:9'     'smartLoaderStructHistory', repmat(GetSmartLoaderStruct(),0,1), ... % N vector that hold the history of the loader struct output 
  // 'SmartLoaderGlobalInit:10'     'loaderTimeTatHistoryMs', zeros(0,1,'uint64') ... % N*1 vector that hold the history of the loader location's time tag 
  // 'SmartLoaderGlobalInit:11'     );
  // 'SmartLoaderGlobalInit:13' SmartLoaderGlobal = SmartLoaderGlobalStruct;
  SD->pd->SmartLoaderGlobal.isInitialized = true;
  SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] = 0;
  SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] = 0;
}

//
// function [smartLoaderStruct] = SmartLoaderSmoothAngles(smartLoaderStruct, configParams)
// Smooth the angles
// Arguments    : PerceptionSmartLoaderStackData *SD
//                PerceptionSmartLoaderStruct *smartLoaderStruct
//                double configParams_loaderYawAngleSmoothWeight
//                double configParams_loaderToShovelYawAngleSmoothWeight
// Return Type  : void
//
static void SmartLoaderSmoothAngles(PerceptionSmartLoaderStackData *SD,
  PerceptionSmartLoaderStruct *smartLoaderStruct, double
  configParams_loaderYawAngleSmoothWeight, double
  configParams_loaderToShovelYawAngleSmoothWeight)
{
  // 'SmartLoaderSmoothAngles:6' if smartLoaderStruct.loaderYawAngleStatus
  if (smartLoaderStruct->loaderYawAngleStatus) {
    // 'SmartLoaderSmoothAngles:7' if numel(SmartLoaderGlobal.smartLoaderStructHistory) > 1 && SmartLoaderGlobal.smartLoaderStructHistory(end).loaderYawAngleStatus 
    if ((SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] > 1) &&
        SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
        pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
        loaderYawAngleStatus) {
      // 'SmartLoaderSmoothAngles:8' smartLoaderStruct.loaderYawAngleDegSmooth = configParams.loaderYawAngleSmoothWeight * smartLoaderStruct.loaderYawAngleDeg + ... 
      // 'SmartLoaderSmoothAngles:9'             (1-configParams.loaderYawAngleSmoothWeight) * SmartLoaderGlobal.smartLoaderStructHistory(end).loaderYawAngleDegSmooth; 
      smartLoaderStruct->loaderYawAngleDegSmooth =
        configParams_loaderYawAngleSmoothWeight *
        smartLoaderStruct->loaderYawAngleDeg + (1.0 -
        configParams_loaderYawAngleSmoothWeight) * SD->
        pd->SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
        pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
        loaderYawAngleDegSmooth;
    } else {
      // 'SmartLoaderSmoothAngles:10' else
      // 'SmartLoaderSmoothAngles:11' smartLoaderStruct.loaderYawAngleDegSmooth = smartLoaderStruct.loaderYawAngleDeg; 
      smartLoaderStruct->loaderYawAngleDegSmooth =
        smartLoaderStruct->loaderYawAngleDeg;
    }
  }

  // 'SmartLoaderSmoothAngles:15' if smartLoaderStruct.loaderToShovelYawAngleDegStatus 
  if (smartLoaderStruct->loaderToShovelYawAngleDegStatus) {
    // 'SmartLoaderSmoothAngles:16' if numel(SmartLoaderGlobal.smartLoaderStructHistory) > 1 && SmartLoaderGlobal.smartLoaderStructHistory(end).loaderToShovelYawAngleDegStatus 
    if ((SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] > 1) &&
        SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
        pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
        loaderToShovelYawAngleDegStatus) {
      // 'SmartLoaderSmoothAngles:17' smartLoaderStruct.loaderToShovelYawAngleDegSmooth = configParams.loaderToShovelYawAngleSmoothWeight * smartLoaderStruct.loaderToShovelYawAngleDeg + ... 
      // 'SmartLoaderSmoothAngles:18'             (1-configParams.loaderToShovelYawAngleSmoothWeight) * SmartLoaderGlobal.smartLoaderStructHistory(end).loaderToShovelYawAngleDegSmooth; 
      smartLoaderStruct->loaderToShovelYawAngleDegSmooth =
        configParams_loaderToShovelYawAngleSmoothWeight *
        smartLoaderStruct->loaderToShovelYawAngleDeg + (1.0 -
        configParams_loaderToShovelYawAngleSmoothWeight) * SD->
        pd->SmartLoaderGlobal.smartLoaderStructHistory.data[SD->
        pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1].
        loaderToShovelYawAngleDegSmooth;
    } else {
      // 'SmartLoaderSmoothAngles:19' else
      // 'SmartLoaderSmoothAngles:20' smartLoaderStruct.loaderToShovelYawAngleDegSmooth = smartLoaderStruct.loaderToShovelYawAngleDeg; 
      smartLoaderStruct->loaderToShovelYawAngleDegSmooth =
        smartLoaderStruct->loaderToShovelYawAngleDeg;
    }
  }
}

//
// Arguments    : const boolean_T x[2]
// Return Type  : boolean_T
//
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

//
// Arguments    : const boolean_T x[2]
// Return Type  : boolean_T
//
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

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                float y_data[]
//                int y_size[2]
//                const int idx_data[]
// Return Type  : void
//
static void apply_row_permutation(PerceptionSmartLoaderStackData *SD, float
  y_data[], int y_size[2], const int idx_data[])
{
  int m;
  int n;
  int j;
  int i;
  m = y_size[0] - 1;
  n = y_size[1];
  for (j = 0; j < n; j++) {
    for (i = 0; i <= m; i++) {
      SD->u1.f8.ycol_data[i] = y_data[j + y_size[1] * (idx_data[i] - 1)];
    }

    for (i = 0; i <= m; i++) {
      y_data[j + y_size[1] * i] = SD->u1.f8.ycol_data[i];
    }
  }
}

//
// function [distanceFromPointToPlane, isPointAbovePlane] = CalcPlaneToPointDistance(planeModelParameters, srcPoints)
// The function calculate the plane to point or a vector of points distance
//  Input arguments:
//  model - plane model - type of XXX
//  srcPoints - matrix of Nx3 of 3d points
//  Output arguments:
//  distanceFromPointToPlane - distance for each point, size of Nx1
//  isPointAbovePlane - boolean - represet is the current point is above or below the plane - above or below is related to the normal vector
//  of the plane
// Arguments    : const double planeModelParameters[4]
//                const double srcPoints[3]
//                double *distanceFromPointToPlane
//                boolean_T *isPointAbovePlane
// Return Type  : void
//
static void b_CalcPlaneToPointDistance(const double planeModelParameters[4],
  const double srcPoints[3], double *distanceFromPointToPlane, boolean_T
  *isPointAbovePlane)
{
  double temp1;

  // assert(isa(planeModelParameters, 'double'));
  // 'CalcPlaneToPointDistance:14' assert(size(planeModelParameters,1) == 1);
  // 'CalcPlaneToPointDistance:15' assert(size(planeModelParameters,2) == 4);
  // 'CalcPlaneToPointDistance:18' modelParametersRepmat = repmat(planeModelParameters(1:3), size(srcPoints,1), 1); 
  // 'CalcPlaneToPointDistance:20' temp1 = sum(bsxfun(@times, srcPoints, modelParametersRepmat), 2) + planeModelParameters(4); 
  temp1 = ((srcPoints[0] * planeModelParameters[0] + srcPoints[1] *
            planeModelParameters[1]) + srcPoints[2] * planeModelParameters[2]) +
    planeModelParameters[3];

  // 'CalcPlaneToPointDistance:22' distanceFromPointToPlane = abs(temp1) / norm(planeModelParameters(1:3)); 
  *distanceFromPointToPlane = std::abs(temp1) / c_norm(*(double (*)[3])&
    planeModelParameters[0]);

  // 'CalcPlaneToPointDistance:24' isPointAbovePlane = temp1 > 0;
  *isPointAbovePlane = (temp1 > 0.0);
}

//
// Arguments    : const float x_data[]
//                const int x_size[1]
//                float y_data[]
//                int y_size[1]
// Return Type  : void
//
static void b_abs(const float x_data[], const int x_size[1], float y_data[], int
                  y_size[1])
{
  int i10;
  int k;
  y_size[0] = x_size[0];
  if (x_size[0] != 0) {
    i10 = x_size[0];
    for (k = 0; k < i10; k++) {
      y_data[k] = std::abs(x_data[k]);
    }
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                float y_data[]
//                int y_size[2]
//                const int idx_data[]
// Return Type  : void
//
static void b_apply_row_permutation(PerceptionSmartLoaderStackData *SD, float
  y_data[], int y_size[2], const int idx_data[])
{
  int m;
  int i;
  m = y_size[0] - 1;
  for (i = 0; i <= m; i++) {
    SD->u1.f6.ycol_data[i] = y_data[(idx_data[i] - 1) << 1];
  }

  for (i = 0; i <= m; i++) {
    y_data[i << 1] = SD->u1.f6.ycol_data[i];
  }

  for (i = 0; i <= m; i++) {
    SD->u1.f6.ycol_data[i] = y_data[1 + ((idx_data[i] - 1) << 1)];
  }

  for (i = 0; i <= m; i++) {
    y_data[1 + (i << 1)] = SD->u1.f6.ycol_data[i];
  }
}

//
// Arguments    : const float x_data[]
//                const int x_size[1]
//                double xi
// Return Type  : int
//
static int b_bsearch(const float x_data[], const int x_size[1], double xi)
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

//
// Arguments    : const float a_data[]
//                const int a_size[2]
//                const double b_data[]
//                const int b_size[2]
//                float c_data[]
//                int c_size[2]
// Return Type  : void
//
static void b_bsxfun(const float a_data[], const int a_size[2], const double
                     b_data[], const int b_size[2], float c_data[], int c_size[2])
{
  int u0;
  int u1;
  int csz_idx_0;
  int acoef;
  int bcoef;
  int varargin_3;
  u0 = b_size[0];
  u1 = a_size[0];
  if (u0 < u1) {
    u1 = u0;
  }

  if (b_size[0] == 1) {
    csz_idx_0 = a_size[0];
  } else if (a_size[0] == 1) {
    csz_idx_0 = b_size[0];
  } else if (a_size[0] == b_size[0]) {
    csz_idx_0 = a_size[0];
  } else {
    csz_idx_0 = u1;
  }

  c_size[1] = 3;
  u0 = b_size[0];
  u1 = a_size[0];
  if (u0 < u1) {
    u1 = u0;
  }

  if (b_size[0] == 1) {
    c_size[0] = a_size[0];
  } else if (a_size[0] == 1) {
    c_size[0] = b_size[0];
  } else if (a_size[0] == b_size[0]) {
    c_size[0] = a_size[0];
  } else {
    c_size[0] = u1;
  }

  if (csz_idx_0 != 0) {
    acoef = (a_size[0] != 1);
    bcoef = (b_size[0] != 1);
    u0 = csz_idx_0 - 1;
    for (u1 = 0; u1 <= u0; u1++) {
      csz_idx_0 = acoef * u1;
      varargin_3 = bcoef * u1;
      c_data[3 * u1] = a_data[3 * csz_idx_0] * (float)b_data[3 * varargin_3];
      c_data[1 + 3 * u1] = a_data[1 + 3 * csz_idx_0] * (float)b_data[1 + 3 *
        varargin_3];
      c_data[2 + 3 * u1] = a_data[2 + 3 * csz_idx_0] * (float)b_data[2 + 3 *
        varargin_3];
    }
  }
}

//
// Arguments    : double x[2]
// Return Type  : void
//
static void b_ceil(double x[2])
{
  x[0] = std::ceil(x[0]);
  x[1] = std::ceil(x[1]);
}

//
// Arguments    : double *x
// Return Type  : void
//
static void b_cosd(double *x)
{
  double absx;
  signed char n;
  *x = rt_remd(*x, 360.0);
  absx = std::abs(*x);
  if (absx > 180.0) {
    if (*x > 0.0) {
      *x -= 360.0;
    } else {
      *x += 360.0;
    }

    absx = std::abs(*x);
  }

  if (absx <= 45.0) {
    *x *= 0.017453292519943295;
    n = 0;
  } else if (absx <= 135.0) {
    if (*x > 0.0) {
      *x = 0.017453292519943295 * (*x - 90.0);
      n = 1;
    } else {
      *x = 0.017453292519943295 * (*x + 90.0);
      n = -1;
    }
  } else if (*x > 0.0) {
    *x = 0.017453292519943295 * (*x - 180.0);
    n = 2;
  } else {
    *x = 0.017453292519943295 * (*x + 180.0);
    n = -2;
  }

  if (n == 0) {
    *x = std::cos(*x);
  } else if (n == 1) {
    *x = -std::sin(*x);
  } else if (n == -1) {
    *x = std::sin(*x);
  } else {
    *x = -std::cos(*x);
  }
}

//
// Arguments    : float D_data[]
//                const float X_data[]
//                const int X_size[2]
//                const float C[6]
//                const int crows[2]
//                int ncrows
// Return Type  : void
//
static void b_distfun(float D_data[], const float X_data[], const int X_size[2],
                      const float C[6], const int crows[2], int ncrows)
{
  int n;
  int i;
  int cr;
  int r;
  int i51;
  n = X_size[0] - 1;
  for (i = 0; i < ncrows; i++) {
    cr = crows[i] - 1;
    for (r = 0; r <= n; r++) {
      D_data[cr + (r << 1)] = std::pow(X_data[3 * r] - C[3 * (crows[i] - 1)],
        2.0F);
    }

    for (r = 0; r <= n; r++) {
      i51 = cr + (r << 1);
      D_data[i51] += std::pow(X_data[3 * r + 1] - C[3 * cr + 1], 2.0F);
    }

    for (r = 0; r <= n; r++) {
      D_data[cr + (r << 1)] += std::pow(X_data[3 * r + 2] - C[3 * cr + 2], 2.0F);
    }
  }
}

//
// Arguments    : float C[6]
//                int counts[2]
//                const float X_data[]
//                const int X_size[2]
//                const int idx_data[]
//                int clusters
// Return Type  : void
//
static void b_gcentroids(float C[6], int counts[2], const float X_data[], const
  int X_size[2], const int idx_data[], int clusters)
{
  int n;
  int i55;
  int i56;
  int i57;
  int cc;
  int i;
  n = X_size[0];
  counts[clusters - 1] = 0;
  i55 = 3 * (clusters - 1);
  C[i55] = 0.0F;
  i56 = 1 + i55;
  C[i56] = 0.0F;
  i57 = 2 + i55;
  C[i57] = 0.0F;
  cc = 0;
  C[i55] = 0.0F;
  C[i56] = 0.0F;
  C[i57] = 0.0F;
  for (i = 0; i < n; i++) {
    if (idx_data[i] == clusters) {
      cc++;
      C[i55] += X_data[3 * i];
      C[i56] += X_data[1 + 3 * i];
      C[i57] += X_data[2 + 3 * i];
    }
  }

  counts[clusters - 1] = cc;
  C[i55] /= (float)cc;
  C[i56] /= (float)cc;
  C[i57] /= (float)cc;
}

//
// Arguments    : const float x[4]
//                float y[4]
// Return Type  : void
//
static void b_inv(const float x[4], float y[4])
{
  float r;
  float t;
  if (std::abs(x[2]) > std::abs(x[0])) {
    r = x[0] / x[2];
    t = 1.0F / (r * x[3] - x[1]);
    y[0] = x[3] / x[2] * t;
    y[2] = -t;
    y[1] = -x[1] / x[2] * t;
    y[3] = r * t;
  } else {
    r = x[2] / x[0];
    t = 1.0F / (x[3] - r * x[1]);
    y[0] = x[3] / x[0] * t;
    y[2] = -r * t;
    y[1] = -x[1] / x[0] * t;
    y[3] = t;
  }
}

//
// Arguments    : double x
// Return Type  : double
//
static double b_log2(double x)
{
  double f;
  double t;
  int eint;
  t = frexp(x, &eint);
  if (t == 0.5) {
    f = (double)eint - 1.0;
  } else if ((eint == 1) && (t < 0.75)) {
    f = std::log(2.0 * t) / 0.69314718055994529;
  } else {
    f = std::log(t) / 0.69314718055994529 + (double)eint;
  }

  return f;
}

//
// Arguments    : const float x_data[]
//                const int x_size[2]
//                float y[2]
// Return Type  : void
//
static void b_mean(const float x_data[], const int x_size[2], float y[2])
{
  int vlen;
  int k;
  vlen = x_size[0];
  if (x_size[0] == 0) {
    y[0] = 0.0F;
    y[1] = 0.0F;
  } else {
    y[0] = x_data[0];
    y[1] = x_data[1];
    for (k = 2; k <= vlen; k++) {
      if (vlen >= 2) {
        y[0] += x_data[(k - 1) << 1];
        y[1] += x_data[1 + ((k - 1) << 1)];
      }
    }
  }

  y[0] /= (float)x_size[0];
  y[1] /= (float)x_size[0];
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                int idx_data[]
//                const float x_data[]
//                const int x_size[2]
//                const int dir_data[]
//                const int dir_size[2]
//                int n
// Return Type  : void
//
static void b_mergesort(PerceptionSmartLoaderStackData *SD, int idx_data[],
  const float x_data[], const int x_size[2], const int dir_data[], const int
  dir_size[2], int n)
{
  int i34;
  int k;
  int i;
  int i2;
  int j;
  int pEnd;
  int p;
  int q;
  int qEnd;
  int kEnd;
  int i35;
  i34 = n - 1;
  for (k = 1; k <= i34; k += 2) {
    if (sortLE(x_data, x_size, dir_data, dir_size, k, k + 1)) {
      idx_data[k - 1] = k;
      idx_data[k] = k + 1;
    } else {
      idx_data[k - 1] = k + 1;
      idx_data[k] = k;
    }
  }

  if ((n & 1) != 0) {
    idx_data[n - 1] = n;
  }

  i = 2;
  while (i < n) {
    i2 = i << 1;
    j = 1;
    for (pEnd = 1 + i; pEnd < n + 1; pEnd = qEnd + i) {
      p = j;
      q = pEnd;
      qEnd = j + i2;
      if (qEnd > n + 1) {
        qEnd = n + 1;
      }

      k = 0;
      kEnd = qEnd - j;
      while (k + 1 <= kEnd) {
        i34 = idx_data[q - 1];
        i35 = idx_data[p - 1];
        if (sortLE(x_data, x_size, dir_data, dir_size, i35, i34)) {
          SD->u1.f9.iwork_data[k] = i35;
          p++;
          if (p == pEnd) {
            while (q < qEnd) {
              k++;
              SD->u1.f9.iwork_data[k] = idx_data[q - 1];
              q++;
            }
          }
        } else {
          SD->u1.f9.iwork_data[k] = i34;
          q++;
          if (q == qEnd) {
            while (p < pEnd) {
              k++;
              SD->u1.f9.iwork_data[k] = idx_data[p - 1];
              p++;
            }
          }
        }

        k++;
      }

      for (k = 0; k < kEnd; k++) {
        idx_data[(j + k) - 1] = SD->u1.f9.iwork_data[k];
      }

      j = qEnd;
    }

    i = i2;
  }
}

//
// Arguments    : const float x_data[]
//                const int x_size[2]
//                float y_data[]
//                int y_size[1]
// Return Type  : void
//
static void b_nestedIter(const float x_data[], const int x_size[2], float
  y_data[], int y_size[1])
{
  int i22;
  int k;
  y_size[0] = x_size[0];
  i22 = x_size[0];
  for (k = 0; k < i22; k++) {
    y_data[k] = x_data[3 * k];
    y_data[k] += x_data[3 * k + 1];
    y_data[k] += x_data[3 * k + 2];
  }
}

//
// Arguments    : const float x[3]
// Return Type  : float
//
static float b_norm(const float x[3])
{
  float y;
  float scale;
  float absxk;
  float t;
  scale = 1.29246971E-26F;
  absxk = std::abs(x[0]);
  if (absxk > 1.29246971E-26F) {
    y = 1.0F;
    scale = absxk;
  } else {
    t = absxk / 1.29246971E-26F;
    y = t * t;
  }

  absxk = std::abs(x[1]);
  if (absxk > scale) {
    t = scale / absxk;
    y = 1.0F + y * t * t;
    scale = absxk;
  } else {
    t = absxk / scale;
    y += t * t;
  }

  absxk = std::abs(x[2]);
  if (absxk > scale) {
    t = scale / absxk;
    y = 1.0F + y * t * t;
    scale = absxk;
  } else {
    t = absxk / scale;
    y += t * t;
  }

  return scale * std::sqrt(y);
}

//
// Arguments    : double x_data[]
//                int x_size[1]
// Return Type  : void
//
static void b_nullAssignment(double x_data[], int x_size[1])
{
  int nxout;
  int k;
  PerceptionSmartLoaderTLS *PerceptionSmartLoaderTLSThread;
  PerceptionSmartLoaderTLSThread = emlrtGetThreadStackData();
  nxout = x_size[0] - 1;
  for (k = 0; k < nxout; k++) {
    x_data[k] = x_data[k + 1];
  }

  if (1 > nxout) {
    nxout = 0;
  } else {
    nxout = x_size[0] - 1;
  }

  if (0 <= nxout - 1) {
    memcpy(&PerceptionSmartLoaderTLSThread->u1.f2.x_data[0], &x_data[0],
           (unsigned int)(nxout * (int)sizeof(double)));
  }

  x_size[0] = nxout;
  if (0 <= nxout - 1) {
    memcpy(&x_data[0], &PerceptionSmartLoaderTLSThread->u1.f2.x_data[0],
           (unsigned int)(nxout * (int)sizeof(double)));
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
// Return Type  : double
//
static double b_rand(PerceptionSmartLoaderStackData *SD)
{
  double r;
  int j;
  unsigned int u[2];
  unsigned int mti;
  int kk;
  unsigned int y;

  // ========================= COPYRIGHT NOTICE ============================
  //  This is a uniform (0,1) pseudorandom number generator based on:
  //
  //  A C-program for MT19937, with initialization improved 2002/1/26.
  //  Coded by Takuji Nishimura and Makoto Matsumoto.
  //
  //  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
  //  All rights reserved.
  //
  //  Redistribution and use in source and binary forms, with or without
  //  modification, are permitted provided that the following conditions
  //  are met:
  //
  //    1. Redistributions of source code must retain the above copyright
  //       notice, this list of conditions and the following disclaimer.
  //
  //    2. Redistributions in binary form must reproduce the above copyright
  //       notice, this list of conditions and the following disclaimer
  //       in the documentation and/or other materials provided with the
  //       distribution.
  //
  //    3. The names of its contributors may not be used to endorse or
  //       promote products derived from this software without specific
  //       prior written permission.
  //
  //  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  //  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  //  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  //  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
  //  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  //  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  //  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  //  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  //  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  //  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  //  OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  //
  // =============================   END   =================================
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

//
// Arguments    : const double a[3]
//                double varargin_1
//                double b_data[]
//                int b_size[2]
// Return Type  : void
//
static void b_repmat(const double a[3], double varargin_1, double b_data[], int
                     b_size[2])
{
  int b_size_tmp;
  int t;
  b_size[1] = 3;
  b_size_tmp = (int)varargin_1;
  b_size[0] = b_size_tmp;
  if (b_size_tmp != 0) {
    b_size_tmp--;
    for (t = 0; t <= b_size_tmp; t++) {
      b_data[3 * t] = a[0];
      b_data[1 + 3 * t] = a[1];
      b_data[2 + 3 * t] = a[2];
    }
  }
}

//
// Arguments    : double *x
// Return Type  : void
//
static void b_sind(double *x)
{
  double absx;
  signed char n;
  *x = rt_remd(*x, 360.0);
  absx = std::abs(*x);
  if (absx > 180.0) {
    if (*x > 0.0) {
      *x -= 360.0;
    } else {
      *x += 360.0;
    }

    absx = std::abs(*x);
  }

  if (absx <= 45.0) {
    *x *= 0.017453292519943295;
    n = 0;
  } else if (absx <= 135.0) {
    if (*x > 0.0) {
      *x = 0.017453292519943295 * (*x - 90.0);
      n = 1;
    } else {
      *x = 0.017453292519943295 * (*x + 90.0);
      n = -1;
    }
  } else if (*x > 0.0) {
    *x = 0.017453292519943295 * (*x - 180.0);
    n = 2;
  } else {
    *x = 0.017453292519943295 * (*x + 180.0);
    n = -2;
  }

  if (n == 0) {
    *x = std::sin(*x);
  } else if (n == 1) {
    *x = std::cos(*x);
  } else if (n == -1) {
    *x = -std::cos(*x);
  } else {
    *x = -std::sin(*x);
  }
}

//
// Arguments    : creal_T x[2]
//                int idx[2]
// Return Type  : void
//
static void b_sort(creal_T x[2], int idx[2])
{
  creal_T xwork[2];
  if (c_sortLE(x)) {
    idx[0] = 1;
    idx[1] = 2;
  } else {
    idx[0] = 2;
    idx[1] = 1;
  }

  xwork[0] = x[0];
  xwork[1] = x[1];
  x[0] = xwork[idx[0] - 1];
  x[1] = xwork[idx[1] - 1];
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float x_data[]
//                const int x_size[2]
//                int idx_data[]
//                int idx_size[1]
// Return Type  : void
//
static void b_sortIdx(PerceptionSmartLoaderStackData *SD, const float x_data[],
                      const int x_size[2], int idx_data[], int idx_size[1])
{
  idx_size[0] = x_size[0];
  if (0 <= x_size[0] - 1) {
    memset(&idx_data[0], 0, (unsigned int)(x_size[0] * (int)sizeof(int)));
  }

  c_mergesort(SD, idx_data, x_data, x_size[0]);
}

//
// Arguments    : const float v_data[]
//                int idx1
//                int idx2
// Return Type  : boolean_T
//
static boolean_T b_sortLE(const float v_data[], int idx1, int idx2)
{
  boolean_T p;
  int k;
  boolean_T exitg1;
  float v1;
  float v2;
  p = true;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 2)) {
    v1 = v_data[k + ((idx1 - 1) << 1)];
    v2 = v_data[k + ((idx2 - 1) << 1)];
    if (v1 != v2) {
      p = (v1 <= v2);
      exitg1 = true;
    } else {
      k++;
    }
  }

  return p;
}

//
// Arguments    : float x_data[]
//                int x_size[1]
// Return Type  : void
//
static void b_sqrt(float x_data[], int x_size[1])
{
  int i44;
  int k;
  int i45;
  i44 = x_size[0];
  for (k = 0; k < i44; k++) {
    i45 = (signed char)(1 + k) - 1;
    x_data[i45] = std::sqrt(x_data[i45]);
  }
}

//
// Arguments    : const float x_data[]
//                const int x_size[2]
//                float y_data[]
//                int y_size[1]
// Return Type  : void
//
static void b_sum(const float x_data[], const int x_size[2], float y_data[], int
                  y_size[1])
{
  if (x_size[0] == 0) {
    y_size[0] = 0;
  } else {
    nestedIter(x_data, x_size, y_data, y_size);
  }
}

//
// Arguments    : const float x_data[]
//                const int x_size[2]
//                float y_data[]
//                int y_size[1]
// Return Type  : void
//
static void b_vecnorm(const float x_data[], const int x_size[2], float y_data[],
                      int y_size[1])
{
  int hi;
  int k;
  float scale;
  int absxk_tmp;
  float absxk;
  float t;
  float yv;
  y_size[0] = x_size[0];
  if (0 <= x_size[0] - 1) {
    memset(&y_data[0], 0, (unsigned int)(x_size[0] * (int)sizeof(float)));
  }

  hi = x_size[0];
  for (k = 0; k < hi; k++) {
    scale = 1.29246971E-26F;
    absxk_tmp = k << 1;
    absxk = std::abs(x_data[absxk_tmp]);
    if (absxk > 1.29246971E-26F) {
      yv = 1.0F;
      scale = absxk;
    } else {
      t = absxk / 1.29246971E-26F;
      yv = t * t;
    }

    absxk = std::abs(x_data[1 + absxk_tmp]);
    if (absxk > scale) {
      t = scale / absxk;
      yv = 1.0F + yv * t * t;
      scale = absxk;
    } else {
      t = absxk / scale;
      yv += t * t;
    }

    y_data[k] = scale * std::sqrt(yv);
  }
}

//
// Arguments    : const float queryPt[2]
//                const double lowBounds_data[]
//                const double upBounds_data[]
//                float poweredRadius
// Return Type  : boolean_T
//
static boolean_T ball_within_bounds(const float queryPt[2], const double
  lowBounds_data[], const double upBounds_data[], float poweredRadius)
{
  boolean_T ballIsWithinBounds;
  float f0;
  float lowDist_idx_0;
  float f1;
  float upDist_idx_0;
  float b_lowDist_idx_0;
  float b_upDist_idx_0;
  f0 = queryPt[0] - (float)lowBounds_data[0];
  f0 *= f0;
  lowDist_idx_0 = f0;
  f1 = queryPt[0] - (float)upBounds_data[0];
  f1 *= f1;
  upDist_idx_0 = f1;
  f0 = queryPt[1] - (float)lowBounds_data[1];
  f0 *= f0;
  f1 = queryPt[1] - (float)upBounds_data[1];
  f1 *= f1;
  if (lowDist_idx_0 > f0) {
    b_lowDist_idx_0 = f0;
  } else {
    b_lowDist_idx_0 = lowDist_idx_0;
  }

  if (b_lowDist_idx_0 <= poweredRadius) {
    ballIsWithinBounds = false;
  } else {
    if (upDist_idx_0 > f1) {
      b_upDist_idx_0 = f1;
    } else {
      b_upDist_idx_0 = upDist_idx_0;
    }

    if (b_upDist_idx_0 <= poweredRadius) {
      ballIsWithinBounds = false;
    } else {
      ballIsWithinBounds = true;
    }
  }

  return ballIsWithinBounds;
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float X_data[]
//                const int X_size[2]
//                int idx_data[]
//                int idx_size[1]
//                float C[6]
//                float D_data[]
//                int D_size[2]
//                int counts[2]
//                boolean_T *converged
//                int *iter
// Return Type  : void
//
static void batchUpdate(PerceptionSmartLoaderStackData *SD, const float X_data[],
  const int X_size[2], int idx_data[], int idx_size[1], float C[6], float
  D_data[], int D_size[2], int counts[2], boolean_T *converged, int *iter)
{
  int n;
  int empties[2];
  int previdx_size_idx_0;
  int moved_size[1];
  int changed[2];
  int nchanged;
  float prevtotsumD;
  int exitg1;
  int nempty;
  int i;
  float maxd;
  int lonely;
  int nMoved;
  int from;
  float f7;
  boolean_T exitg2;
  int d_size[1];
  int nidx_size[1];
  n = X_size[0] - 1;
  empties[0] = 0;
  empties[1] = 0;
  previdx_size_idx_0 = X_size[0];
  if (0 <= X_size[0] - 1) {
    memset(&SD->u2.f13.previdx_data[0], 0, (unsigned int)(X_size[0] * (int)
            sizeof(int)));
  }

  moved_size[0] = X_size[0];
  if (0 <= X_size[0] - 1) {
    memset(&SD->u2.f13.moved_data[0], 0, (unsigned int)(X_size[0] * (int)sizeof
            (int)));
  }

  changed[0] = 1;
  changed[1] = 2;
  nchanged = 2;
  prevtotsumD = 3.402823466E+38F;
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
          f7 = D_data[(idx_data[nMoved] + (nMoved << 1)) - 1];
          if (f7 > maxd) {
            maxd = f7;
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

    maxd = 0.0F;
    for (i = 0; i <= n; i++) {
      maxd += D_data[(idx_data[i] + (i << 1)) - 1];
    }

    if (prevtotsumD <= maxd) {
      idx_size[0] = previdx_size_idx_0;
      if (0 <= previdx_size_idx_0 - 1) {
        memcpy(&idx_data[0], &SD->u2.f13.previdx_data[0], (unsigned int)
               (previdx_size_idx_0 * (int)sizeof(int)));
      }

      gcentroids(C, counts, X_data, X_size, SD->u2.f13.previdx_data, changed,
                 nchanged);
      (*iter)--;
      exitg1 = 1;
    } else if (*iter >= 100) {
      exitg1 = 1;
    } else {
      previdx_size_idx_0 = idx_size[0];
      nMoved = idx_size[0];
      if (0 <= nMoved - 1) {
        memcpy(&SD->u2.f13.previdx_data[0], &idx_data[0], (unsigned int)(nMoved *
                (int)sizeof(int)));
      }

      prevtotsumD = maxd;
      mindim2(D_data, D_size, SD->u2.f13.d_data, d_size, SD->u2.f13.nidx_data,
              nidx_size);
      nMoved = 0;
      for (i = 0; i <= n; i++) {
        if ((SD->u2.f13.nidx_data[i] != SD->u2.f13.previdx_data[i]) && (D_data
             [(SD->u2.f13.previdx_data[i] + (i << 1)) - 1] > SD->u2.f13.d_data[i]))
        {
          nMoved++;
          SD->u2.f13.moved_data[nMoved - 1] = i + 1;
          idx_data[i] = SD->u2.f13.nidx_data[i];
        }
      }

      if (nMoved == 0) {
        *converged = true;
        exitg1 = 1;
      } else {
        nchanged = findchanged(SD, changed, idx_data, SD->u2.f13.previdx_data,
          SD->u2.f13.moved_data, moved_size, nMoved);
      }
    }
  } while (exitg1 == 0);
}

//
// Arguments    : const float queryPt[2]
//                const double lowBounds_data[]
//                const double upBounds_data[]
//                float radius
// Return Type  : boolean_T
//
static boolean_T bounds_overlap_ball(const float queryPt[2], const double
  lowBounds_data[], const double upBounds_data[], float radius)
{
  boolean_T boundsOverlapBall;
  float sumDist;
  int c;
  boolean_T exitg1;
  float pRadIn;
  boundsOverlapBall = true;
  sumDist = 0.0F;
  c = 0;
  exitg1 = false;
  while ((!exitg1) && (c < 2)) {
    if (queryPt[c] < lowBounds_data[c]) {
      pRadIn = queryPt[c] - (float)lowBounds_data[c];
      sumDist += pRadIn * pRadIn;
    } else {
      if (queryPt[c] > upBounds_data[c]) {
        pRadIn = queryPt[c] - (float)upBounds_data[c];
        sumDist += pRadIn * pRadIn;
      }
    }

    if (sumDist > radius) {
      boundsOverlapBall = false;
      exitg1 = true;
    } else {
      c++;
    }
  }

  return boundsOverlapBall;
}

//
// Arguments    : const float a_data[]
//                const int a_size[2]
//                const float b[2]
//                float c_data[]
//                int c_size[2]
// Return Type  : void
//
static void bsxfun(const float a_data[], const int a_size[2], const float b[2],
                   float c_data[], int c_size[2])
{
  int acoef;
  int i5;
  int k;
  int varargin_2;
  int c_data_tmp;
  c_size[1] = 2;
  c_size[0] = a_size[0];
  if (a_size[0] != 0) {
    acoef = (a_size[0] != 1);
    i5 = a_size[0] - 1;
    for (k = 0; k <= i5; k++) {
      varargin_2 = acoef * k;
      c_data_tmp = k << 1;
      c_data[c_data_tmp] = a_data[varargin_2 << 1] - b[0];
      c_data[1 + c_data_tmp] = a_data[1 + (varargin_2 << 1)] - b[1];
    }
  }
}

//
// function [distanceFromPointToPlane, isPointAbovePlane] = CalcPlaneToPointDistance(planeModelParameters, srcPoints)
// The function calculate the plane to point or a vector of points distance
//  Input arguments:
//  model - plane model - type of XXX
//  srcPoints - matrix of Nx3 of 3d points
//  Output arguments:
//  distanceFromPointToPlane - distance for each point, size of Nx1
//  isPointAbovePlane - boolean - represet is the current point is above or below the plane - above or below is related to the normal vector
//  of the plane
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const double planeModelParameters[4]
//                const float srcPoints_data[]
//                const int srcPoints_size[2]
//                float distanceFromPointToPlane_data[]
//                int distanceFromPointToPlane_size[1]
//                boolean_T isPointAbovePlane_data[]
//                int isPointAbovePlane_size[1]
// Return Type  : void
//
static void c_CalcPlaneToPointDistance(PerceptionSmartLoaderStackData *SD, const
  double planeModelParameters[4], const float srcPoints_data[], const int
  srcPoints_size[2], float distanceFromPointToPlane_data[], int
  distanceFromPointToPlane_size[1], boolean_T isPointAbovePlane_data[], int
  isPointAbovePlane_size[1])
{
  int tmp_size[2];
  int b_tmp_size[2];
  int temp1_size[1];
  int loop_ub;
  int i21;
  double y;

  // assert(isa(planeModelParameters, 'double'));
  // 'CalcPlaneToPointDistance:14' assert(size(planeModelParameters,1) == 1);
  // 'CalcPlaneToPointDistance:15' assert(size(planeModelParameters,2) == 4);
  // 'CalcPlaneToPointDistance:18' modelParametersRepmat = repmat(planeModelParameters(1:3), size(srcPoints,1), 1); 
  // 'CalcPlaneToPointDistance:20' temp1 = sum(bsxfun(@times, srcPoints, modelParametersRepmat), 2) + planeModelParameters(4); 
  b_repmat(*(double (*)[3])&planeModelParameters[0], (double)srcPoints_size[0],
           SD->u1.f1.tmp_data, tmp_size);
  b_bsxfun(srcPoints_data, srcPoints_size, SD->u1.f1.tmp_data, tmp_size,
           SD->u1.f1.b_tmp_data, b_tmp_size);
  d_sum(SD->u1.f1.b_tmp_data, b_tmp_size, SD->u1.f1.temp1_data, temp1_size);
  loop_ub = temp1_size[0];
  for (i21 = 0; i21 < loop_ub; i21++) {
    SD->u1.f1.temp1_data[i21] += (float)planeModelParameters[3];
  }

  // 'CalcPlaneToPointDistance:22' distanceFromPointToPlane = abs(temp1) / norm(planeModelParameters(1:3)); 
  y = c_norm(*(double (*)[3])&planeModelParameters[0]);
  b_abs(SD->u1.f1.temp1_data, temp1_size, distanceFromPointToPlane_data,
        distanceFromPointToPlane_size);
  loop_ub = distanceFromPointToPlane_size[0];
  for (i21 = 0; i21 < loop_ub; i21++) {
    distanceFromPointToPlane_data[i21] /= (float)y;
  }

  // 'CalcPlaneToPointDistance:24' isPointAbovePlane = temp1 > 0;
  isPointAbovePlane_size[0] = temp1_size[0];
  loop_ub = temp1_size[0];
  for (i21 = 0; i21 < loop_ub; i21++) {
    isPointAbovePlane_data[i21] = (SD->u1.f1.temp1_data[i21] > 0.0F);
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                int idx_data[]
//                const float x_data[]
//                int n
// Return Type  : void
//
static void c_mergesort(PerceptionSmartLoaderStackData *SD, int idx_data[],
  const float x_data[], int n)
{
  int i37;
  int k;
  int i;
  int i2;
  int j;
  int pEnd;
  int p;
  int q;
  int qEnd;
  int kEnd;
  int i38;
  i37 = n - 1;
  for (k = 1; k <= i37; k += 2) {
    if (b_sortLE(x_data, k, k + 1)) {
      idx_data[k - 1] = k;
      idx_data[k] = k + 1;
    } else {
      idx_data[k - 1] = k + 1;
      idx_data[k] = k;
    }
  }

  if ((n & 1) != 0) {
    idx_data[n - 1] = n;
  }

  i = 2;
  while (i < n) {
    i2 = i << 1;
    j = 1;
    for (pEnd = 1 + i; pEnd < n + 1; pEnd = qEnd + i) {
      p = j;
      q = pEnd;
      qEnd = j + i2;
      if (qEnd > n + 1) {
        qEnd = n + 1;
      }

      k = 0;
      kEnd = qEnd - j;
      while (k + 1 <= kEnd) {
        i37 = idx_data[q - 1];
        i38 = idx_data[p - 1];
        if (b_sortLE(x_data, i38, i37)) {
          SD->u1.f7.iwork_data[k] = i38;
          p++;
          if (p == pEnd) {
            while (q < qEnd) {
              k++;
              SD->u1.f7.iwork_data[k] = idx_data[q - 1];
              q++;
            }
          }
        } else {
          SD->u1.f7.iwork_data[k] = i37;
          q++;
          if (q == qEnd) {
            while (p < pEnd) {
              k++;
              SD->u1.f7.iwork_data[k] = idx_data[p - 1];
              p++;
            }
          }
        }

        k++;
      }

      for (k = 0; k < kEnd; k++) {
        idx_data[(j + k) - 1] = SD->u1.f7.iwork_data[k];
      }

      j = qEnd;
    }

    i = i2;
  }
}

//
// Arguments    : const double x[3]
// Return Type  : double
//
static double c_norm(const double x[3])
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

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                float x_data[]
//                int x_size[2]
//                const int idx_data[]
// Return Type  : void
//
static void c_nullAssignment(PerceptionSmartLoaderStackData *SD, float x_data[],
  int x_size[2], const int idx_data[])
{
  int nrows;
  int i60;
  int i;
  nrows = x_size[0] - 1;
  i60 = idx_data[0];
  for (i = i60; i <= nrows; i++) {
    x_data[(i - 1) << 1] = x_data[i << 1];
  }

  for (i = i60; i <= nrows; i++) {
    x_data[1 + ((i - 1) << 1)] = x_data[1 + (i << 1)];
  }

  if (1 > nrows) {
    i = 0;
  } else {
    i = x_size[0] - 1;
  }

  for (i60 = 0; i60 < i; i60++) {
    nrows = i60 << 1;
    SD->u1.f2.x_data[nrows] = x_data[nrows];
    nrows++;
    SD->u1.f2.x_data[nrows] = x_data[nrows];
  }

  x_size[1] = 2;
  x_size[0] = i;
  for (i60 = 0; i60 < i; i60++) {
    nrows = i60 << 1;
    x_data[nrows] = SD->u1.f2.x_data[nrows];
    nrows++;
    x_data[nrows] = SD->u1.f2.x_data[nrows];
  }
}

//
// Arguments    : float x_data[]
//                int x_size[1]
//                int idx_data[]
//                int idx_size[1]
// Return Type  : void
//
static void c_sortIdx(float x_data[], int x_size[1], int idx_data[], int
                      idx_size[1])
{
  int unnamed_idx_0;
  int loop_ub;
  int i40;
  int i41;
  float x4[4];
  int idx4[4];
  int ib;
  int k;
  signed char perm[4];
  int i1;
  int i3;
  int i4;
  float f5;
  float f6;
  PerceptionSmartLoaderTLS *PerceptionSmartLoaderTLSThread;
  PerceptionSmartLoaderTLSThread = emlrtGetThreadStackData();
  unnamed_idx_0 = x_size[0];
  idx_size[0] = unnamed_idx_0;
  if (0 <= unnamed_idx_0 - 1) {
    memset(&idx_data[0], 0, (unsigned int)(unnamed_idx_0 * (int)sizeof(int)));
  }

  if (x_size[0] != 0) {
    if (0 <= unnamed_idx_0 - 1) {
      memset(&PerceptionSmartLoaderTLSThread->u1.f0.idx_data[0], 0, (unsigned
              int)(unnamed_idx_0 * (int)sizeof(int)));
    }

    loop_ub = x_size[0];
    if (0 <= loop_ub - 1) {
      memcpy(&PerceptionSmartLoaderTLSThread->u1.f0.x_data[0], &x_data[0],
             (unsigned int)(loop_ub * (int)sizeof(float)));
    }

    i40 = x_size[0];
    i41 = x_size[0] - 1;
    x4[0] = 0.0F;
    idx4[0] = 0;
    x4[1] = 0.0F;
    idx4[1] = 0;
    x4[2] = 0.0F;
    idx4[2] = 0;
    x4[3] = 0.0F;
    idx4[3] = 0;
    if (0 <= unnamed_idx_0 - 1) {
      memset(&PerceptionSmartLoaderTLSThread->u1.f0.iwork_data[0], 0, (unsigned
              int)(unnamed_idx_0 * (int)sizeof(int)));
    }

    ib = x_size[0];
    if (0 <= ib - 1) {
      memset(&PerceptionSmartLoaderTLSThread->u1.f0.xwork_data[0], 0, (unsigned
              int)(ib * (int)sizeof(float)));
    }

    ib = -1;
    for (k = 0; k <= i41; k++) {
      ib++;
      idx4[ib] = k + 1;
      x4[ib] = PerceptionSmartLoaderTLSThread->u1.f0.x_data[k];
      if (ib + 1 == 4) {
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

        f5 = x4[i1 - 1];
        f6 = x4[i3 - 1];
        if (f5 <= f6) {
          if (x4[ib - 1] <= f6) {
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
          f6 = x4[i4 - 1];
          if (f5 <= f6) {
            if (x4[ib - 1] <= f6) {
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

        i3 = perm[0] - 1;
        PerceptionSmartLoaderTLSThread->u1.f0.idx_data[k - 3] = idx4[i3];
        i4 = perm[1] - 1;
        PerceptionSmartLoaderTLSThread->u1.f0.idx_data[k - 2] = idx4[i4];
        ib = perm[2] - 1;
        PerceptionSmartLoaderTLSThread->u1.f0.idx_data[k - 1] = idx4[ib];
        i1 = perm[3] - 1;
        PerceptionSmartLoaderTLSThread->u1.f0.idx_data[k] = idx4[i1];
        PerceptionSmartLoaderTLSThread->u1.f0.x_data[k - 3] = x4[i3];
        PerceptionSmartLoaderTLSThread->u1.f0.x_data[k - 2] = x4[i4];
        PerceptionSmartLoaderTLSThread->u1.f0.x_data[k - 1] = x4[ib];
        PerceptionSmartLoaderTLSThread->u1.f0.x_data[k] = x4[i1];
        ib = -1;
      }
    }

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
        i3 = perm[k] - 1;
        i4 = (i41 - ib) + k;
        PerceptionSmartLoaderTLSThread->u1.f0.idx_data[i4] = idx4[i3];
        PerceptionSmartLoaderTLSThread->u1.f0.x_data[i4] = x4[i3];
      }
    }

    ib = 2;
    if (i40 > 1) {
      if (i40 >= 256) {
        ib = i40 >> 8;
        for (i1 = 0; i1 < ib; i1++) {
          merge_pow2_block(PerceptionSmartLoaderTLSThread->u1.f0.idx_data,
                           PerceptionSmartLoaderTLSThread->u1.f0.x_data, i1 << 8);
        }

        ib <<= 8;
        i1 = i40 - ib;
        if (i1 > 0) {
          merge_block(PerceptionSmartLoaderTLSThread->u1.f0.idx_data,
                      PerceptionSmartLoaderTLSThread->u1.f0.x_data, ib, i1, 2,
                      PerceptionSmartLoaderTLSThread->u1.f0.iwork_data,
                      PerceptionSmartLoaderTLSThread->u1.f0.xwork_data);
        }

        ib = 8;
      }

      merge_block(PerceptionSmartLoaderTLSThread->u1.f0.idx_data,
                  PerceptionSmartLoaderTLSThread->u1.f0.x_data, 0, i40, ib,
                  PerceptionSmartLoaderTLSThread->u1.f0.iwork_data,
                  PerceptionSmartLoaderTLSThread->u1.f0.xwork_data);
    }

    if (0 <= unnamed_idx_0 - 1) {
      memcpy(&idx_data[0], &PerceptionSmartLoaderTLSThread->u1.f0.idx_data[0],
             (unsigned int)(unnamed_idx_0 * (int)sizeof(int)));
    }

    if (0 <= loop_ub - 1) {
      memcpy(&x_data[0], &PerceptionSmartLoaderTLSThread->u1.f0.x_data[0],
             (unsigned int)(loop_ub * (int)sizeof(float)));
    }
  }
}

//
// Arguments    : const creal_T v[2]
// Return Type  : boolean_T
//
static boolean_T c_sortLE(const creal_T v[2])
{
  double ma;
  boolean_T SCALEA;
  double mb;
  boolean_T SCALEB;
  double x;
  double y;
  double absxk;
  int exponent;
  double Ma;
  int b_exponent;
  int c_exponent;
  int d_exponent;
  ma = std::abs(v[0].re);
  if ((ma > 8.9884656743115785E+307) || (std::abs(v[0].im) >
       8.9884656743115785E+307)) {
    SCALEA = true;
  } else {
    SCALEA = false;
  }

  mb = std::abs(v[1].re);
  if ((mb > 8.9884656743115785E+307) || (std::abs(v[1].im) >
       8.9884656743115785E+307)) {
    SCALEB = true;
  } else {
    SCALEB = false;
  }

  if (SCALEA || SCALEB) {
    x = rt_hypotd(v[0].re / 2.0, v[0].im / 2.0);
    y = rt_hypotd(v[1].re / 2.0, v[1].im / 2.0);
  } else {
    x = rt_hypotd(v[0].re, v[0].im);
    y = rt_hypotd(v[1].re, v[1].im);
  }

  absxk = y / 2.0;
  if (absxk <= 2.2250738585072014E-308) {
    absxk = 4.94065645841247E-324;
  } else {
    frexp(absxk, &exponent);
    absxk = std::ldexp(1.0, exponent - 53);
  }

  if (std::abs(y - x) < absxk) {
    absxk = std::abs(v[0].im);
    x = std::abs(v[1].im);
    if (ma > absxk) {
      Ma = ma;
      ma = absxk;
    } else {
      Ma = absxk;
    }

    if (mb > x) {
      absxk = mb;
      mb = x;
    } else {
      absxk = x;
    }

    if (Ma > absxk) {
      if (ma < mb) {
        x = Ma - absxk;
        y = (ma / 2.0 + mb / 2.0) / (Ma / 2.0 + absxk / 2.0) * (mb - ma);
      } else {
        x = Ma;
        y = absxk;
      }
    } else if (Ma < absxk) {
      if (ma > mb) {
        y = absxk - Ma;
        x = (ma / 2.0 + mb / 2.0) / (Ma / 2.0 + absxk / 2.0) * (ma - mb);
      } else {
        x = Ma;
        y = absxk;
      }
    } else {
      x = ma;
      y = mb;
    }

    absxk = std::abs(y / 2.0);
    if (absxk <= 2.2250738585072014E-308) {
      absxk = 4.94065645841247E-324;
    } else {
      frexp(absxk, &b_exponent);
      absxk = std::ldexp(1.0, b_exponent - 53);
    }

    if (std::abs(y - x) < absxk) {
      x = atan2(v[0].im, v[0].re);
      y = atan2(v[1].im, v[1].re);
      absxk = std::abs(y / 2.0);
      if (absxk <= 2.2250738585072014E-308) {
        absxk = 4.94065645841247E-324;
      } else {
        frexp(absxk, &c_exponent);
        absxk = std::ldexp(1.0, c_exponent - 53);
      }

      if (std::abs(y - x) < absxk) {
        if (x > 0.78539816339744828) {
          if (x > 2.3561944901923448) {
            x = -v[0].im;
            y = -v[1].im;
          } else {
            x = -v[0].re;
            y = -v[1].re;
          }
        } else if (x > -0.78539816339744828) {
          x = v[0].im;
          y = v[1].im;
        } else if (x > -2.3561944901923448) {
          x = v[0].re;
          y = v[1].re;
        } else {
          x = -v[0].im;
          y = -v[1].im;
        }

        absxk = std::abs(y / 2.0);
        if (absxk <= 2.2250738585072014E-308) {
          absxk = 4.94065645841247E-324;
        } else {
          frexp(absxk, &d_exponent);
          absxk = std::ldexp(1.0, d_exponent - 53);
        }

        if (std::abs(y - x) < absxk) {
          x = 0.0;
          y = 0.0;
        }
      }
    }
  }

  return x <= y;
}

//
// Arguments    : const float x_data[]
//                const int x_size[1]
// Return Type  : float
//
static float c_sum(const float x_data[], const int x_size[1])
{
  float y;
  int vlen;
  int k;
  vlen = x_size[0];
  if (x_size[0] == 0) {
    y = 0.0F;
  } else {
    y = x_data[0];
    for (k = 2; k <= vlen; k++) {
      if (vlen >= 2) {
        y += x_data[k - 1];
      }
    }
  }

  return y;
}

//
// Arguments    : double x1
//                double b_y1
//                double x2
//                double y2
//                signed char quad1
//                signed char quad2
//                double scale
//                signed char *diffQuad
//                boolean_T *onj
// Return Type  : void
//
static void contrib(double x1, double b_y1, double x2, double y2, signed char
                    quad1, signed char quad2, double scale, signed char
                    *diffQuad, boolean_T *onj)
{
  double cp;
  *onj = false;
  *diffQuad = (signed char)(quad2 - quad1);
  cp = x1 * y2 - x2 * b_y1;
  if (std::abs(cp) < scale) {
    *onj = (x1 * x2 + b_y1 * y2 <= 0.0);
    if ((*diffQuad == 2) || (*diffQuad == -2)) {
      *diffQuad = 0;
    } else if (*diffQuad == -3) {
      *diffQuad = 1;
    } else {
      if (*diffQuad == 3) {
        *diffQuad = -1;
      }
    }
  } else if (cp < 0.0) {
    if (*diffQuad == 2) {
      *diffQuad = -2;
    } else if (*diffQuad == -3) {
      *diffQuad = 1;
    } else {
      if (*diffQuad == 3) {
        *diffQuad = -1;
      }
    }
  } else if (*diffQuad == -2) {
    *diffQuad = 2;
  } else if (*diffQuad == -3) {
    *diffQuad = 1;
  } else {
    if (*diffQuad == 3) {
      *diffQuad = -1;
    }
  }
}

//
// Arguments    : int empties[2]
//                const int counts[2]
//                const int changed[2]
//                int nchanged
// Return Type  : int
//
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

//
// Arguments    : const double x[2]
// Return Type  : double
//
static double d_norm(const double x[2])
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

  return scale * std::sqrt(y);
}

//
// Arguments    : const float x_data[]
//                const int x_size[2]
//                float y_data[]
//                int y_size[1]
// Return Type  : void
//
static void d_sum(const float x_data[], const int x_size[2], float y_data[], int
                  y_size[1])
{
  if (x_size[0] == 0) {
    y_size[0] = 0;
  } else {
    b_nestedIter(x_data, x_size, y_data, y_size);
  }
}

//
// Arguments    : const creal_T v[4]
//                creal_T d[2]
// Return Type  : void
//
static void diag(const creal_T v[4], creal_T d[2])
{
  d[0] = v[0];
  d[1] = v[3];
}

//
// Arguments    : float D_data[]
//                const float X_data[]
//                const int X_size[2]
//                const float C[6]
//                int crows
// Return Type  : void
//
static void distfun(float D_data[], const float X_data[], const int X_size[2],
                    const float C[6], int crows)
{
  int n;
  int r;
  int i50;
  n = X_size[0] - 1;
  for (r = 0; r <= n; r++) {
    D_data[(crows + (r << 1)) - 1] = std::pow(X_data[3 * r] - C[3 * (crows - 1)],
      2.0F);
  }

  for (r = 0; r <= n; r++) {
    i50 = (crows + (r << 1)) - 1;
    D_data[i50] += std::pow(X_data[3 * r + 1] - C[3 * (crows - 1) + 1], 2.0F);
  }

  for (r = 0; r <= n; r++) {
    D_data[(crows + (r << 1)) - 1] += std::pow(X_data[3 * r + 2] - C[3 * (crows
      - 1) + 2], 2.0F);
  }
}

//
// Arguments    : int numerator
//                int denominator
// Return Type  : int
//
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

//
// Arguments    : const creal_T a[2]
//                const double b[2]
// Return Type  : creal_T
//
static creal_T dot(const creal_T a[2], const double b[2])
{
  creal_T c;
  c.re = a[0].re * b[0];
  c.im = a[0].im * -b[0];
  c.re += a[1].re * b[1];
  c.im += a[1].im * -b[1];
  return c;
}

//
// Arguments    : const boolean_T x_data[]
//                const int x_size[1]
// Return Type  : double
//
static double e_sum(const boolean_T x_data[], const int x_size[1])
{
  double y;
  int vlen;
  int k;
  vlen = x_size[0];
  if (x_size[0] == 0) {
    y = 0.0;
  } else {
    y = x_data[0];
    for (k = 2; k <= vlen; k++) {
      if (vlen >= 2) {
        y += (double)x_data[k - 1];
      }
    }
  }

  return y;
}

//
// Arguments    : const double A[4]
//                creal_T V[4]
//                creal_T D[4]
// Return Type  : void
//
static void eig(const double A[4], creal_T V[4], creal_T D[4])
{
  boolean_T p;
  int info;
  boolean_T exitg2;
  int i;
  creal_T At[4];
  double b_A[4];
  int exitg1;
  double absxk;
  double Vr[4];
  creal_T alpha1[2];
  creal_T beta1[2];
  double colnorm;
  double scale;
  int k;
  double t;
  double ba;
  int i30;
  double s;
  double d;
  double cs;
  double sn;
  double rt1i;
  p = true;
  info = 0;
  exitg2 = false;
  while ((!exitg2) && (info < 2)) {
    i = 0;
    do {
      exitg1 = 0;
      if (i <= info) {
        if (A[i + (info << 1)] != A[info + (i << 1)]) {
          p = false;
          exitg1 = 1;
        } else {
          i++;
        }
      } else {
        info++;
        exitg1 = 2;
      }
    } while (exitg1 == 0);

    if (exitg1 == 1) {
      exitg2 = true;
    }
  }

  if (p) {
    b_A[0] = A[0];
    b_A[1] = A[1];
    b_A[2] = A[2];
    b_A[3] = A[3];
    absxk = xgehrd(b_A);
    Vr[2] = 0.0;
    Vr[1] = 0.0;
    Vr[0] = 1.0;
    Vr[3] = 1.0 - absxk;
    for (i = 1; i + 1 >= 1; i = k - 2) {
      k = i + 1;
      exitg2 = false;
      while ((!exitg2) && (k > 1)) {
        ba = std::abs(b_A[1]);
        if (ba <= 2.0041683600089728E-292) {
          exitg2 = true;
        } else {
          t = std::abs(b_A[3]);
          if (ba <= 2.2204460492503131E-16 * (std::abs(b_A[0]) + t)) {
            scale = std::abs(b_A[2]);
            if (ba > scale) {
              colnorm = ba;
              ba = scale;
            } else {
              colnorm = scale;
            }

            scale = std::abs(b_A[0] - b_A[3]);
            if (t > scale) {
              absxk = t;
              t = scale;
            } else {
              absxk = scale;
            }

            s = absxk + colnorm;
            scale = 2.2204460492503131E-16 * (t * (absxk / s));
            if (2.0041683600089728E-292 > scale) {
              scale = 2.0041683600089728E-292;
            }

            if (ba * (colnorm / s) <= scale) {
              exitg2 = true;
            } else {
              k = 1;
            }
          } else {
            k = 1;
          }
        }
      }

      if (k > 1) {
        b_A[1] = 0.0;
      }

      if ((k != i + 1) && (k == i)) {
        info = i << 1;
        scale = b_A[info];
        absxk = b_A[i];
        i30 = i + info;
        t = b_A[i30];
        xdlanv2(&b_A[0], &scale, &absxk, &t, &colnorm, &ba, &s, &d, &cs, &sn);
        b_A[info] = scale;
        b_A[i] = absxk;
        b_A[i30] = t;
        if (2 > i + 1) {
          b_A[2] = cs * b_A[2] + sn * b_A[2];
        }

        info = (i - 1) << 1;
        absxk = cs * Vr[info] + sn * Vr[2];
        Vr[2] = cs * Vr[2] - sn * Vr[info];
        Vr[info] = absxk;
        info++;
        absxk = cs * Vr[info] + sn * Vr[3];
        Vr[3] = cs * Vr[3] - sn * Vr[info];
        Vr[info] = absxk;
      }
    }

    D[0].re = b_A[0];
    V[0].re = Vr[0];
    V[0].im = 0.0;
    V[1].re = Vr[1];
    V[1].im = 0.0;
    V[2].re = Vr[2];
    V[2].im = 0.0;
    D[3].re = b_A[3];
    V[3].re = Vr[3];
    V[3].im = 0.0;
    if (b_A[1] != 0.0) {
      scale = b_A[0];
      absxk = b_A[2];
      t = b_A[1];
      d = b_A[3];
      xdlanv2(&scale, &absxk, &t, &d, &colnorm, &rt1i, &ba, &s, &cs, &sn);
      t = colnorm - b_A[3];
      absxk = rt_hypotd(rt_hypotd(t, rt1i), b_A[1]);
      if (rt1i == 0.0) {
        t /= absxk;
        scale = 0.0;
      } else if (t == 0.0) {
        t = 0.0;
        scale = rt1i / absxk;
      } else {
        t /= absxk;
        scale = rt1i / absxk;
      }

      s = b_A[1] / absxk;
      D[0].re = t * b_A[0] + s * b_A[1];
      D[0].im = 0.0 - scale * b_A[0];
      D[1].re = t * b_A[1] - s * b_A[0];
      D[3].re = t * b_A[3] - s * b_A[2];
      D[3].im = scale * b_A[3];
      D[0].re = (t * D[0].re - scale * D[0].im) + s * (t * b_A[2] + s * b_A[3]);
      D[3].re = (t * D[3].re + scale * D[3].im) - s * D[1].re;
      V[0].re = t * Vr[0] + s * Vr[2];
      V[0].im = scale * Vr[0];
      V[2].re = t * Vr[2] - s * Vr[0];
      V[2].im = 0.0 - scale * Vr[2];
      V[1].re = t * Vr[1] + s * Vr[3];
      V[1].im = scale * Vr[1];
      V[3].re = t * Vr[3] - s * Vr[1];
      V[3].im = 0.0 - scale * Vr[3];
    }

    D[0].im = 0.0;
    D[3].im = 0.0;
    D[1].re = 0.0;
    D[1].im = 0.0;
    D[2].re = 0.0;
    D[2].im = 0.0;
  } else {
    At[0].re = A[0];
    At[0].im = 0.0;
    At[1].re = A[1];
    At[1].im = 0.0;
    At[2].re = A[2];
    At[2].im = 0.0;
    At[3].re = A[3];
    At[3].im = 0.0;
    xzggev(At, &info, alpha1, beta1, V);
    colnorm = 0.0;
    scale = 3.3121686421112381E-170;
    for (k = 1; k < 3; k++) {
      absxk = std::abs(V[k - 1].re);
      if (absxk > scale) {
        t = scale / absxk;
        colnorm = 1.0 + colnorm * t * t;
        scale = absxk;
      } else {
        t = absxk / scale;
        colnorm += t * t;
      }

      absxk = std::abs(V[k - 1].im);
      if (absxk > scale) {
        t = scale / absxk;
        colnorm = 1.0 + colnorm * t * t;
        scale = absxk;
      } else {
        t = absxk / scale;
        colnorm += t * t;
      }
    }

    colnorm = scale * std::sqrt(colnorm);
    for (info = 1; info < 3; info++) {
      absxk = V[info - 1].re;
      scale = V[info - 1].im;
      if (scale == 0.0) {
        V[info - 1].re = absxk / colnorm;
        V[info - 1].im = 0.0;
      } else if (absxk == 0.0) {
        V[info - 1].re = 0.0;
        V[info - 1].im = scale / colnorm;
      } else {
        V[info - 1].re = absxk / colnorm;
        V[info - 1].im = scale / colnorm;
      }
    }

    colnorm = 0.0;
    scale = 3.3121686421112381E-170;
    for (k = 3; k < 5; k++) {
      absxk = std::abs(V[k - 1].re);
      if (absxk > scale) {
        t = scale / absxk;
        colnorm = 1.0 + colnorm * t * t;
        scale = absxk;
      } else {
        t = absxk / scale;
        colnorm += t * t;
      }

      absxk = std::abs(V[k - 1].im);
      if (absxk > scale) {
        t = scale / absxk;
        colnorm = 1.0 + colnorm * t * t;
        scale = absxk;
      } else {
        t = absxk / scale;
        colnorm += t * t;
      }
    }

    colnorm = scale * std::sqrt(colnorm);
    for (info = 3; info < 5; info++) {
      absxk = V[info - 1].re;
      scale = V[info - 1].im;
      if (scale == 0.0) {
        V[info - 1].re = absxk / colnorm;
        V[info - 1].im = 0.0;
      } else if (absxk == 0.0) {
        V[info - 1].re = 0.0;
        V[info - 1].im = scale / colnorm;
      } else {
        V[info - 1].re = absxk / colnorm;
        V[info - 1].im = scale / colnorm;
      }
    }

    D[1].re = 0.0;
    D[1].im = 0.0;
    D[2].re = 0.0;
    D[2].im = 0.0;
    if (beta1[0].im == 0.0) {
      if (alpha1[0].im == 0.0) {
        D[0].re = alpha1[0].re / beta1[0].re;
        D[0].im = 0.0;
      } else if (alpha1[0].re == 0.0) {
        D[0].re = 0.0;
        D[0].im = alpha1[0].im / beta1[0].re;
      } else {
        D[0].re = alpha1[0].re / beta1[0].re;
        D[0].im = alpha1[0].im / beta1[0].re;
      }
    } else if (beta1[0].re == 0.0) {
      if (alpha1[0].re == 0.0) {
        D[0].re = alpha1[0].im / beta1[0].im;
        D[0].im = 0.0;
      } else if (alpha1[0].im == 0.0) {
        D[0].re = 0.0;
        D[0].im = -(alpha1[0].re / beta1[0].im);
      } else {
        D[0].re = alpha1[0].im / beta1[0].im;
        D[0].im = -(alpha1[0].re / beta1[0].im);
      }
    } else {
      t = std::abs(beta1[0].re);
      absxk = std::abs(beta1[0].im);
      if (t > absxk) {
        s = beta1[0].im / beta1[0].re;
        d = beta1[0].re + s * beta1[0].im;
        D[0].re = (alpha1[0].re + s * alpha1[0].im) / d;
        D[0].im = (alpha1[0].im - s * alpha1[0].re) / d;
      } else if (absxk == t) {
        if (beta1[0].re > 0.0) {
          absxk = 0.5;
        } else {
          absxk = -0.5;
        }

        if (beta1[0].im > 0.0) {
          scale = 0.5;
        } else {
          scale = -0.5;
        }

        D[0].re = (alpha1[0].re * absxk + alpha1[0].im * scale) / t;
        D[0].im = (alpha1[0].im * absxk - alpha1[0].re * scale) / t;
      } else {
        s = beta1[0].re / beta1[0].im;
        d = beta1[0].im + s * beta1[0].re;
        D[0].re = (s * alpha1[0].re + alpha1[0].im) / d;
        D[0].im = (s * alpha1[0].im - alpha1[0].re) / d;
      }
    }

    if (beta1[1].im == 0.0) {
      if (alpha1[1].im == 0.0) {
        D[3].re = alpha1[1].re / beta1[1].re;
        D[3].im = 0.0;
      } else if (alpha1[1].re == 0.0) {
        D[3].re = 0.0;
        D[3].im = alpha1[1].im / beta1[1].re;
      } else {
        D[3].re = alpha1[1].re / beta1[1].re;
        D[3].im = alpha1[1].im / beta1[1].re;
      }
    } else if (beta1[1].re == 0.0) {
      if (alpha1[1].re == 0.0) {
        D[3].re = alpha1[1].im / beta1[1].im;
        D[3].im = 0.0;
      } else if (alpha1[1].im == 0.0) {
        D[3].re = 0.0;
        D[3].im = -(alpha1[1].re / beta1[1].im);
      } else {
        D[3].re = alpha1[1].im / beta1[1].im;
        D[3].im = -(alpha1[1].re / beta1[1].im);
      }
    } else {
      t = std::abs(beta1[1].re);
      absxk = std::abs(beta1[1].im);
      if (t > absxk) {
        s = beta1[1].im / beta1[1].re;
        d = beta1[1].re + s * beta1[1].im;
        D[3].re = (alpha1[1].re + s * alpha1[1].im) / d;
        D[3].im = (alpha1[1].im - s * alpha1[1].re) / d;
      } else if (absxk == t) {
        if (beta1[1].re > 0.0) {
          absxk = 0.5;
        } else {
          absxk = -0.5;
        }

        if (beta1[1].im > 0.0) {
          scale = 0.5;
        } else {
          scale = -0.5;
        }

        D[3].re = (alpha1[1].re * absxk + alpha1[1].im * scale) / t;
        D[3].im = (alpha1[1].im * absxk - alpha1[1].re * scale) / t;
      } else {
        s = beta1[1].re / beta1[1].im;
        d = beta1[1].im + s * beta1[1].re;
        D[3].re = (s * alpha1[1].re + alpha1[1].im) / d;
        D[3].im = (s * alpha1[1].im - alpha1[1].re) / d;
      }
    }
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
// Return Type  : void
//
static void eml_rand_mt19937ar_stateful_init(PerceptionSmartLoaderStackData *SD)
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

//
// Arguments    : void
// Return Type  : void
//
static void emlrtFreeThreadStackData()
{
  int i;

#pragma omp parallel for schedule(static)\
 num_threads(omp_get_max_threads())

  for (i = 1; i <= omp_get_max_threads(); i++) {
    free(PerceptionSmartLoaderTLSGlobal);
  }
}

//
// Arguments    : void
// Return Type  : PerceptionSmartLoaderTLS *
//
static PerceptionSmartLoaderTLS *emlrtGetThreadStackData()
{
  return PerceptionSmartLoaderTLSGlobal;
}

//
// Arguments    : void
// Return Type  : void
//
static void emlrtInitThreadStackData()
{
  int i;

#pragma omp parallel for schedule(static)\
 num_threads(omp_get_max_threads())

  for (i = 1; i <= omp_get_max_threads(); i++) {
    PerceptionSmartLoaderTLSGlobal = (PerceptionSmartLoaderTLS *)malloc(1U *
      sizeof(PerceptionSmartLoaderTLS));
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                int changed[2]
//                const int idx_data[]
//                const int previdx_data[]
//                const int moved_data[]
//                const int moved_size[1]
//                int nmoved
// Return Type  : int
//
static int findchanged(PerceptionSmartLoaderStackData *SD, int changed[2], const
  int idx_data[], const int previdx_data[], const int moved_data[], const int
  moved_size[1], int nmoved)
{
  int nchanged;
  int j;
  int i58;
  if (0 <= moved_size[0] - 1) {
    memset(&SD->u1.f5.b_data[0], 0, (unsigned int)(moved_size[0] * (int)sizeof
            (boolean_T)));
  }

  for (j = 0; j < nmoved; j++) {
    SD->u1.f5.b_data[idx_data[moved_data[j] - 1] - 1] = true;
    SD->u1.f5.b_data[previdx_data[moved_data[j] - 1] - 1] = true;
  }

  nchanged = 0;
  i58 = moved_size[0];
  for (j = 0; j < i58; j++) {
    if (SD->u1.f5.b_data[j]) {
      nchanged++;
      changed[nchanged - 1] = j + 1;
    }
  }

  return nchanged;
}

//
// Arguments    : float C[6]
//                int counts[2]
//                const float X_data[]
//                const int X_size[2]
//                const int idx_data[]
//                const int clusters[2]
//                int nclusters
// Return Type  : void
//
static void gcentroids(float C[6], int counts[2], const float X_data[], const
  int X_size[2], const int idx_data[], const int clusters[2], int nclusters)
{
  int n;
  int ic;
  int i52;
  int cc;
  int i53;
  int i54;
  int i;
  n = X_size[0];
  for (ic = 0; ic < nclusters; ic++) {
    counts[clusters[ic] - 1] = 0;
    i52 = 3 * (clusters[ic] - 1);
    C[i52] = 0.0F;
    C[1 + i52] = 0.0F;
    C[2 + i52] = 0.0F;
  }

  for (ic = 0; ic < nclusters; ic++) {
    cc = 0;
    i52 = 3 * (clusters[ic] - 1);
    C[i52] = 0.0F;
    i53 = 1 + i52;
    C[i53] = 0.0F;
    i54 = 2 + i52;
    C[i54] = 0.0F;
    for (i = 0; i < n; i++) {
      if (idx_data[i] == clusters[ic]) {
        cc++;
        C[i52] += X_data[3 * i];
        C[i53] += X_data[1 + 3 * i];
        C[i54] += X_data[2 + 3 * i];
      }
    }

    counts[clusters[ic] - 1] = cc;
    C[i52] /= (float)cc;
    C[i53] /= (float)cc;
    C[i54] /= (float)cc;
  }
}

//
// Arguments    : const unsigned int idxAll_data[]
//                const double idxDim_data[]
//                double this_node
//                unsigned int node_idx_this_data[]
//                int node_idx_this_size[1]
// Return Type  : void
//
static void getNodeFromArray(const unsigned int idxAll_data[], const double
  idxDim_data[], double this_node, unsigned int node_idx_this_data[], int
  node_idx_this_size[1])
{
  double d1;
  int loop_ub_tmp;
  int loop_ub;
  int i4;
  int idxDim_size[1];
  short tmp_data[32766];
  double nIdx;
  PerceptionSmartLoaderTLS *PerceptionSmartLoaderTLSThread;
  PerceptionSmartLoaderTLSThread = emlrtGetThreadStackData();
  d1 = idxDim_data[(int)this_node - 1];
  if (d1 == 0.0) {
    node_idx_this_size[0] = 0;
  } else if (this_node == 1.0) {
    if (1.0 > idxDim_data[0]) {
      loop_ub = 0;
    } else {
      loop_ub = (int)idxDim_data[0];
    }

    for (i4 = 0; i4 < loop_ub; i4++) {
      PerceptionSmartLoaderTLSThread->u1.f1.b_tmp_data[i4] = i4;
    }

    node_idx_this_size[0] = loop_ub;
    for (i4 = 0; i4 < loop_ub; i4++) {
      node_idx_this_data[i4] = idxAll_data
        [PerceptionSmartLoaderTLSThread->u1.f1.b_tmp_data[i4]];
    }
  } else {
    loop_ub_tmp = (short)(int)(this_node - 1.0);
    loop_ub = loop_ub_tmp - 1;
    for (i4 = 0; i4 <= loop_ub; i4++) {
      tmp_data[i4] = (short)i4;
    }

    idxDim_size[0] = loop_ub_tmp;
    for (i4 = 0; i4 < loop_ub_tmp; i4++) {
      PerceptionSmartLoaderTLSThread->u1.f1.idxDim_data[i4] =
        idxDim_data[tmp_data[i4]];
    }

    nIdx = sum(PerceptionSmartLoaderTLSThread->u1.f1.idxDim_data, idxDim_size);
    loop_ub_tmp = (int)std::floor(d1 - 1.0);
    loop_ub = loop_ub_tmp + 1;
    for (i4 = 0; i4 <= loop_ub_tmp; i4++) {
      PerceptionSmartLoaderTLSThread->u1.f1.tmp_data[i4] = nIdx + (1.0 + (double)
        i4);
    }

    node_idx_this_size[0] = loop_ub;
    for (i4 = 0; i4 < loop_ub; i4++) {
      node_idx_this_data[i4] = idxAll_data[(int)
        PerceptionSmartLoaderTLSThread->u1.f1.tmp_data[i4] - 1];
    }
  }
}

//
// Arguments    : const float queryPt[2]
//                const double cutDim_data[]
//                const double cutVal_data[]
//                const boolean_T leafNode_data[]
//                const double leftChild_data[]
//                const double rightChild_data[]
// Return Type  : double
//
static double get_starting_node(const float queryPt[2], const double
  cutDim_data[], const double cutVal_data[], const boolean_T leafNode_data[],
  const double leftChild_data[], const double rightChild_data[])
{
  double node;
  node = 1.0;
  while (!leafNode_data[(int)node - 1]) {
    if (queryPt[(int)cutDim_data[(int)node - 1] - 1] <= cutVal_data[(int)node -
        1]) {
      node = leftChild_data[(int)node - 1];
    } else {
      node = rightChild_data[(int)node - 1];
    }
  }

  return node;
}

//
// Arguments    : const boolean_T x_data[]
// Return Type  : boolean_T
//
static boolean_T ifWhileCond(const boolean_T x_data[])
{
  boolean_T y;
  y = true;
  if (!x_data[0]) {
    y = false;
  }

  return y;
}

//
// Arguments    : const float x_data[]
//                const int x_size[1]
//                const float y_data[]
//                const double xv[4]
//                const double yv[4]
//                boolean_T in_data[]
//                int in_size[1]
// Return Type  : void
//
static void inpolygon(const float x_data[], const int x_size[1], const float
                      y_data[], const double xv[4], const double yv[4],
                      boolean_T in_data[], int in_size[1])
{
  signed char last[4];
  double minxv;
  double maxxv;
  int i23;
  int j;
  double minyv;
  double a;
  double maxyv;
  double scale[4];
  double b;
  double b_a;
  int b_j;
  int c_j;
  float xj;
  float yj;
  signed char sdq;
  double xvFirst;
  double yvFirst;
  signed char quadFirst;
  double xv2;
  double yv2;
  signed char quad2;
  int exitg1;
  signed char dquad;
  boolean_T onj;
  double xv1;
  double yv1;
  signed char quad1;
  in_size[0] = x_size[0];
  if (0 <= x_size[0] - 1) {
    memset(&in_data[0], 0, (unsigned int)(x_size[0] * (int)sizeof(boolean_T)));
  }

  if (x_size[0] != 0) {
    if ((xv[3] == xv[0]) && (yv[3] == yv[0])) {
      last[0] = 3;
    } else {
      last[0] = 4;
    }

    minxv = xv[0];
    maxxv = xv[0];
    i23 = last[0];
    for (j = 1; j <= i23; j++) {
      a = xv[j - 1];
      if (a < minxv) {
        minxv = a;
      } else {
        if (a > maxxv) {
          maxxv = a;
        }
      }
    }

    minyv = yv[0];
    maxyv = yv[0];
    i23 = last[0];
    for (j = 1; j <= i23; j++) {
      a = yv[j - 1];
      if (a < minyv) {
        minyv = a;
      } else {
        if (a > maxyv) {
          maxyv = a;
        }
      }
    }

    scale[0] = 0.0;
    scale[1] = 0.0;
    scale[2] = 0.0;
    scale[3] = 0.0;
    i23 = last[0] - 1;
    a = std::abs(0.5 * (xv[i23] + xv[0]));
    b = std::abs(0.5 * (yv[i23] + yv[0]));
    if ((a > 1.0) && (b > 1.0)) {
      a *= b;
    } else {
      if (b > a) {
        a = b;
      }
    }

    for (j = 1; j <= i23; j++) {
      b_a = std::abs(0.5 * (xv[j - 1] + xv[j]));
      b = std::abs(0.5 * (yv[j - 1] + yv[j]));
      if ((b_a > 1.0) && (b > 1.0)) {
        b_a *= b;
      } else {
        if (b > b_a) {
          b_a = b;
        }
      }

      scale[j - 1] = b_a * 6.6613381477509392E-16;
    }

    scale[i23] = a * 6.6613381477509392E-16;
    j = x_size[0] - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(c_j,xj,yj,sdq,xvFirst,yvFirst,quadFirst,xv2,yv2,quad2,exitg1,xv1,yv1,quad1) \
 firstprivate(dquad,onj)

    for (b_j = 0; b_j <= j; b_j++) {
      xj = x_data[b_j];
      yj = y_data[b_j];
      in_data[b_j] = false;
      if ((xj >= minxv) && (xj <= maxxv) && (yj >= minyv) && (yj <= maxyv)) {
        sdq = 0;
        xvFirst = xv[0] - xj;
        yvFirst = yv[0] - yj;
        if (xvFirst > 0.0) {
          if (yvFirst > 0.0) {
            quadFirst = 0;
          } else {
            quadFirst = 3;
          }
        } else if (yvFirst > 0.0) {
          quadFirst = 1;
        } else {
          quadFirst = 2;
        }

        xv2 = xvFirst;
        yv2 = yvFirst;
        quad2 = quadFirst;
        c_j = 1;
        do {
          exitg1 = 0;
          if (c_j <= last[0] - 1) {
            xv1 = xv2;
            yv1 = yv2;
            xv2 = xv[c_j] - xj;
            yv2 = yv[c_j] - yj;
            quad1 = quad2;
            if (xv2 > 0.0) {
              if (yv2 > 0.0) {
                quad2 = 0;
              } else {
                quad2 = 3;
              }
            } else if (yv2 > 0.0) {
              quad2 = 1;
            } else {
              quad2 = 2;
            }

            contrib(xv1, yv1, xv2, yv2, quad1, quad2, scale[c_j - 1], &dquad,
                    &onj);
            if (onj) {
              in_data[b_j] = true;
              exitg1 = 1;
            } else {
              sdq += dquad;
              c_j++;
            }
          } else {
            contrib(xv2, yv2, xvFirst, yvFirst, quad2, quadFirst, scale[last[0]
                    - 1], &dquad, &onj);
            if (onj) {
              in_data[b_j] = true;
            } else {
              sdq += dquad;
              in_data[b_j] = (sdq != 0);
            }

            exitg1 = 1;
          }
        } while (exitg1 == 0);
      }
    }
  }
}

//
// Arguments    : const double x[16]
//                double y[16]
// Return Type  : void
//
static void inv(const double x[16], double y[16])
{
  double b_x[16];
  double b_y[16];
  signed char ipiv[4];
  int j;
  signed char p[4];
  int jA;
  int jj;
  int jp1j;
  int n;
  int a;
  int ix;
  int iy;
  double smax;
  double s;
  int i2;
  int i;
  b_x[0] = x[0];
  b_x[1] = x[4];
  b_x[2] = x[8];
  b_x[3] = x[12];
  b_x[4] = x[1];
  b_x[5] = x[5];
  b_x[6] = x[9];
  b_x[7] = x[13];
  b_x[8] = x[2];
  b_x[9] = x[6];
  b_x[10] = x[10];
  b_x[11] = x[14];
  b_x[12] = x[3];
  b_x[13] = x[7];
  b_x[14] = x[11];
  b_x[15] = x[15];
  memset(&b_y[0], 0, sizeof(double) << 4);
  ipiv[0] = 1;
  ipiv[1] = 2;
  ipiv[2] = 3;
  for (j = 0; j < 3; j++) {
    jA = j * 5;
    jj = j * 5;
    jp1j = jA + 2;
    n = 4 - j;
    a = 0;
    ix = jA;
    smax = std::abs(b_x[jA]);
    for (iy = 2; iy <= n; iy++) {
      ix++;
      s = std::abs(b_x[ix]);
      if (s > smax) {
        a = iy - 1;
        smax = s;
      }
    }

    if (b_x[jj + a] != 0.0) {
      if (a != 0) {
        a += j;
        ipiv[j] = (signed char)(a + 1);
        smax = b_x[j];
        b_x[j] = b_x[a];
        b_x[a] = smax;
        ix = j + 4;
        iy = a + 4;
        smax = b_x[ix];
        b_x[ix] = b_x[iy];
        b_x[iy] = smax;
        ix += 4;
        iy += 4;
        smax = b_x[ix];
        b_x[ix] = b_x[iy];
        b_x[iy] = smax;
        ix += 4;
        iy += 4;
        smax = b_x[ix];
        b_x[ix] = b_x[iy];
        b_x[iy] = smax;
      }

      i2 = jj - j;
      for (i = jp1j; i <= i2 + 4; i++) {
        b_x[i - 1] /= b_x[jj];
      }
    }

    n = 2 - j;
    iy = jA + 4;
    jA = jj + 5;
    for (a = 0; a <= n; a++) {
      smax = b_x[iy];
      if (b_x[iy] != 0.0) {
        ix = jj + 1;
        i2 = jA + 1;
        i = (jA - j) + 3;
        for (jp1j = i2; jp1j <= i; jp1j++) {
          b_x[jp1j - 1] += b_x[ix] * -smax;
          ix++;
        }
      }

      iy += 4;
      jA += 4;
    }
  }

  p[0] = 1;
  p[1] = 2;
  p[2] = 3;
  p[3] = 4;
  if (ipiv[0] > 1) {
    a = ipiv[0] - 1;
    iy = p[a];
    p[a] = 1;
    p[0] = (signed char)iy;
  }

  if (ipiv[1] > 2) {
    a = ipiv[1] - 1;
    iy = p[a];
    p[a] = p[1];
    p[1] = (signed char)iy;
  }

  if (ipiv[2] > 3) {
    a = ipiv[2] - 1;
    iy = p[a];
    p[a] = p[2];
    p[2] = (signed char)iy;
  }

  a = p[0] - 1;
  iy = a << 2;
  b_y[iy] = 1.0;
  for (j = 1; j < 5; j++) {
    if (b_y[(j + iy) - 1] != 0.0) {
      i2 = j + 1;
      for (i = i2; i < 5; i++) {
        jA = (i + iy) - 1;
        b_y[jA] -= b_y[(j + (a << 2)) - 1] * b_x[(i + ((j - 1) << 2)) - 1];
      }
    }
  }

  a = p[1] - 1;
  iy = a << 2;
  b_y[1 + iy] = 1.0;
  for (j = 2; j < 5; j++) {
    if (b_y[(j + iy) - 1] != 0.0) {
      i2 = j + 1;
      for (i = i2; i < 5; i++) {
        jA = (i + iy) - 1;
        b_y[jA] -= b_y[(j + (a << 2)) - 1] * b_x[(i + ((j - 1) << 2)) - 1];
      }
    }
  }

  a = p[2] - 1;
  iy = a << 2;
  b_y[2 + iy] = 1.0;
  for (j = 3; j < 5; j++) {
    if (b_y[(j + iy) - 1] != 0.0) {
      i2 = j + 1;
      for (i = i2; i < 5; i++) {
        jA = iy + 3;
        b_y[jA] -= b_y[(j + (a << 2)) - 1] * b_x[((j - 1) << 2) + 3];
      }
    }
  }

  b_y[3 + ((p[3] - 1) << 2)] = 1.0;
  if (b_y[3] != 0.0) {
    b_y[3] /= b_x[15];
    for (i = 0; i < 3; i++) {
      b_y[i] -= b_y[3] * b_x[i + 12];
    }
  }

  if (b_y[2] != 0.0) {
    b_y[2] /= b_x[10];
    for (i = 0; i < 2; i++) {
      b_y[i] -= b_y[2] * b_x[i + 8];
    }
  }

  if (b_y[1] != 0.0) {
    b_y[1] /= b_x[5];
    for (i = 0; i < 1; i++) {
      b_y[0] -= b_y[1] * b_x[4];
    }
  }

  if (b_y[0] != 0.0) {
    b_y[0] /= b_x[0];
  }

  if (b_y[7] != 0.0) {
    b_y[7] /= b_x[15];
    for (i = 0; i < 3; i++) {
      b_y[i + 4] -= b_y[7] * b_x[i + 12];
    }
  }

  if (b_y[6] != 0.0) {
    b_y[6] /= b_x[10];
    for (i = 0; i < 2; i++) {
      b_y[i + 4] -= b_y[6] * b_x[i + 8];
    }
  }

  if (b_y[5] != 0.0) {
    b_y[5] /= b_x[5];
    for (i = 0; i < 1; i++) {
      b_y[4] -= b_y[5] * b_x[4];
    }
  }

  if (b_y[4] != 0.0) {
    b_y[4] /= b_x[0];
  }

  if (b_y[11] != 0.0) {
    b_y[11] /= b_x[15];
    for (i = 0; i < 3; i++) {
      b_y[i + 8] -= b_y[11] * b_x[i + 12];
    }
  }

  if (b_y[10] != 0.0) {
    b_y[10] /= b_x[10];
    for (i = 0; i < 2; i++) {
      b_y[i + 8] -= b_y[10] * b_x[i + 8];
    }
  }

  if (b_y[9] != 0.0) {
    b_y[9] /= b_x[5];
    for (i = 0; i < 1; i++) {
      b_y[8] -= b_y[9] * b_x[4];
    }
  }

  if (b_y[8] != 0.0) {
    b_y[8] /= b_x[0];
  }

  if (b_y[15] != 0.0) {
    b_y[15] /= b_x[15];
    for (i = 0; i < 3; i++) {
      b_y[i + 12] -= b_y[15] * b_x[i + 12];
    }
  }

  if (b_y[14] != 0.0) {
    b_y[14] /= b_x[10];
    for (i = 0; i < 2; i++) {
      b_y[i + 12] -= b_y[14] * b_x[i + 8];
    }
  }

  if (b_y[13] != 0.0) {
    b_y[13] /= b_x[5];
    for (i = 0; i < 1; i++) {
      b_y[12] -= b_y[13] * b_x[4];
    }
  }

  if (b_y[12] != 0.0) {
    b_y[12] /= b_x[0];
  }

  y[0] = b_y[0];
  y[1] = b_y[4];
  y[2] = b_y[8];
  y[3] = b_y[12];
  y[4] = b_y[1];
  y[5] = b_y[5];
  y[6] = b_y[9];
  y[7] = b_y[13];
  y[8] = b_y[2];
  y[9] = b_y[6];
  y[10] = b_y[10];
  y[11] = b_y[14];
  y[12] = b_y[3];
  y[13] = b_y[7];
  y[14] = b_y[11];
  y[15] = b_y[15];
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const struct_T *obj
//                const emxArray_real_T *Y
//                emxArray_real_T *idx
//                emxArray_real32_T *dist
// Return Type  : void
//
static void kdsearchfun(PerceptionSmartLoaderStackData *SD, const struct_T *obj,
  const emxArray_real_T *Y, emxArray_real_T *idx, emxArray_real32_T *dist)
{
  int yk;
  int k;
  int numNN1;
  int numNN;
  int noNanCol_size[2];
  int j;
  float distTemp_data[1];
  int distTemp_size[1];
  int b_j;
  unsigned int t1_I_data[1];
  int t1_I_size[1];
  float tmp_data[64];
  int i3;
  PerceptionSmartLoaderTLS *PerceptionSmartLoaderTLSThread;
  PerceptionSmartLoaderTLSThread = emlrtGetThreadStackData();
  yk = Y->size[1] * Y->size[0];
  for (k = 0; k < yk; k++) {
    SD->u5.f19.Y_data[k] = (float)Y->data[k];
  }

  yk = obj->X.size[1] * obj->X.size[0];
  if (0 <= yk - 1) {
    memcpy(&SD->u5.f19.X_data[0], &obj->X.data[0], (unsigned int)(yk * (int)
            sizeof(float)));
  }

  if (1 > obj->X.size[0]) {
    numNN1 = obj->X.size[0];
  } else {
    numNN1 = 1;
  }

  if ((Y->size[0] == 0) || (numNN1 == 0)) {
    k = idx->size[0] * idx->size[1];
    idx->size[1] = numNN1;
    idx->size[0] = Y->size[0];
    emxEnsureCapacity_real_T(idx, k);
    yk = numNN1 * Y->size[0];
    for (k = 0; k < yk; k++) {
      idx->data[k] = 0.0;
    }

    k = dist->size[0] * dist->size[1];
    dist->size[1] = numNN1;
    dist->size[0] = Y->size[0];
    emxEnsureCapacity_real32_T(dist, k);
    yk = numNN1 * Y->size[0];
    for (k = 0; k < yk; k++) {
      dist->data[k] = 0.0F;
    }
  } else {
    if (numNN1 > obj->nx_nonan) {
      numNN = (int)obj->nx_nonan;
    } else {
      numNN = numNN1;
    }

    if (numNN > 0) {
      k = idx->size[0] * idx->size[1];
      idx->size[1] = numNN1;
      idx->size[0] = Y->size[0];
      emxEnsureCapacity_real_T(idx, k);
      yk = numNN1 * Y->size[0];
      for (k = 0; k < yk; k++) {
        idx->data[k] = 0.0;
      }

      k = dist->size[0] * dist->size[1];
      dist->size[1] = numNN1;
      dist->size[0] = Y->size[0];
      emxEnsureCapacity_real32_T(dist, k);
      yk = numNN1 * Y->size[0];
      for (k = 0; k < yk; k++) {
        dist->data[k] = 0.0F;
      }

      noNanCol_size[1] = numNN;
      SD->u5.f19.noNanCol_data[0] = 1;
      yk = 1;
      for (k = 2; k <= numNN; k++) {
        yk++;
        SD->u5.f19.noNanCol_data[k - 1] = yk;
      }

      k = Y->size[0];
      yk = k - 1;

#pragma omp parallel \
 num_threads(omp_get_max_threads()) \
 private(PerceptionSmartLoaderTLSThread,distTemp_data,distTemp_size,b_j,i3) \
 firstprivate(t1_I_data,t1_I_size,tmp_data)

      {
        PerceptionSmartLoaderTLSThread = emlrtGetThreadStackData();

#pragma omp for nowait

        for (j = 0; j <= yk; j++) {
          search_kdtree(obj->cutDim.data, obj->cutVal.data,
                        obj->lowerBounds.data, obj->lowerBounds.size,
                        obj->upperBounds.data, obj->upperBounds.size,
                        obj->leftChild.data, obj->rightChild.data,
                        obj->leafNode.data, obj->idxAll.data, obj->idxDim.data,
                        SD->u5.f19.X_data, *(float (*)[2])&SD->u5.f19.Y_data[j <<
                        1], numNN, distTemp_data, distTemp_size, t1_I_data,
                        t1_I_size);
          t1_I_size[0] = distTemp_size[0];
          if (0 <= distTemp_size[0] - 1) {
            memcpy(&tmp_data[0], &distTemp_data[0], (unsigned int)
                   (distTemp_size[0] * (int)sizeof(float)));
          }

          b_sqrt(tmp_data, t1_I_size);
          if (0 <= t1_I_size[0] - 1) {
            memcpy(&distTemp_data[0], &tmp_data[0], (unsigned int)(t1_I_size[0] *
                    (int)sizeof(float)));
          }

          b_j = noNanCol_size[1];
          for (i3 = 0; i3 < b_j; i3++) {
            PerceptionSmartLoaderTLSThread->u5.f6.tmp_data[i3] =
              SD->u5.f19.noNanCol_data[i3] - 1;
          }

          b_j = noNanCol_size[1];
          for (i3 = 0; i3 < b_j; i3++) {
            dist->data[PerceptionSmartLoaderTLSThread->u5.f6.tmp_data[i3] +
              dist->size[1] * j] = distTemp_data[i3];
          }

          b_j = noNanCol_size[1];
          for (i3 = 0; i3 < b_j; i3++) {
            PerceptionSmartLoaderTLSThread->u5.f6.tmp_data[i3] =
              SD->u5.f19.noNanCol_data[i3] - 1;
          }

          b_j = noNanCol_size[1];
          for (i3 = 0; i3 < b_j; i3++) {
            idx->data[PerceptionSmartLoaderTLSThread->u5.f6.tmp_data[i3] +
              idx->size[1] * j] = t1_I_data[i3];
          }
        }
      }
    } else {
      idx->size[1] = 0;
      idx->size[0] = Y->size[0];
      dist->size[1] = 0;
      dist->size[0] = Y->size[0];
    }
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float X_data[]
//                const int X_size[2]
//                double idxbest_data[]
//                int idxbest_size[1]
//                float Cbest[6]
//                float varargout_1[2]
//                float varargout_2_data[]
//                int varargout_2_size[2]
// Return Type  : void
//
static void kmeans(PerceptionSmartLoaderStackData *SD, const float X_data[],
                   const int X_size[2], double idxbest_data[], int idxbest_size
                   [1], float Cbest[6], float varargout_1[2], float
                   varargout_2_data[], int varargout_2_size[2])
{
  int idx_size[1];
  int loop_ub;
  int i7;
  local_kmeans(SD, X_data, X_size, SD->u5.f18.idx_data, idx_size, Cbest,
               varargout_1, varargout_2_data, varargout_2_size);
  idxbest_size[0] = idx_size[0];
  loop_ub = idx_size[0];
  for (i7 = 0; i7 < loop_ub; i7++) {
    idxbest_data[i7] = SD->u5.f18.idx_data[i7];
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float X_data[]
//                const int X_size[2]
//                const emxArray_real_T *Y
//                emxArray_real_T *Idx
// Return Type  : void
//
static void knnsearch(PerceptionSmartLoaderStackData *SD, const float X_data[],
                      const int X_size[2], const emxArray_real_T *Y,
                      emxArray_real_T *Idx)
{
  emxArray_int32_T *y;
  int loop_ub;
  int upperBoundsTemp_data_tmp;
  int m;
  double M;
  int lowerBoundsTemp_size[2];
  boolean_T leafNodeTemp_data[32767];
  emxArray_cell_wrap_13 *idxTemp;
  int currentNode;
  int nextUnusedNode;
  int loop_ub_tmp;
  float p;
  float maxval_idx_1;
  signed char cgstruct_cutDim_data[32767];
  float minval_idx_0;
  float minval_idx_1;
  int iidx;
  int x_size[1];
  int iidx_size[1];
  double d0;
  int half;
  int b_upperBoundsTemp_data_tmp;
  double temp[2];
  boolean_T cgstruct_leafNode_data[32767];
  emxArray_real32_T *D;
  emxInit_int32_T(&y, 2);
  if (X_size[0] < 1) {
    y->size[1] = 0;
    y->size[0] = 1;
  } else {
    loop_ub = X_size[0];
    upperBoundsTemp_data_tmp = y->size[0] * y->size[1];
    y->size[1] = X_size[0];
    y->size[0] = 1;
    emxEnsureCapacity_int32_T(y, upperBoundsTemp_data_tmp);
    for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
         upperBoundsTemp_data_tmp++) {
      y->data[upperBoundsTemp_data_tmp] = 1 + upperBoundsTemp_data_tmp;
    }
  }

  m = y->size[1];
  loop_ub = y->size[1] * y->size[0];
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.notnan_data[upperBoundsTemp_data_tmp] = y->
      data[upperBoundsTemp_data_tmp];
  }

  emxFree_int32_T(&y);
  M = (double)X_size[0] / 50.0;
  if (M <= 1.0) {
    M = 1.0;
  }

  M = b_log2(M);
  M = std::ceil(M);
  M = mpower(M + 1.0) - 1.0;
  loop_ub = (int)M;
  if (0 <= loop_ub - 1) {
    memset(&SD->u6.f21.cutDimTemp_data[0], 0, (unsigned int)(loop_ub * (int)
            sizeof(int)));
  }

  loop_ub = (int)M;
  if (0 <= loop_ub - 1) {
    memset(&SD->u6.f21.cutValTemp_data[0], 0, (unsigned int)(loop_ub * (int)
            sizeof(double)));
  }

  lowerBoundsTemp_size[0] = (int)M;
  loop_ub = 2 * lowerBoundsTemp_size[0];
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.lowerBoundsTemp_data[upperBoundsTemp_data_tmp] =
      1.7976931348623157E+308;
  }

  loop_ub = (lowerBoundsTemp_size[0] << 1) - 1;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp <= loop_ub;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.lowerBoundsTemp_data[upperBoundsTemp_data_tmp] =
      -SD->u6.f21.lowerBoundsTemp_data[upperBoundsTemp_data_tmp];
  }

  loop_ub = 2 * (int)M;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.upperBoundsTemp_data[upperBoundsTemp_data_tmp] =
      1.7976931348623157E+308;
  }

  loop_ub = (int)M;
  if (0 <= loop_ub - 1) {
    memset(&SD->u6.f21.leftChildTemp_data[0], 0, (unsigned int)(loop_ub * (int)
            sizeof(double)));
  }

  loop_ub = (int)M;
  if (0 <= loop_ub - 1) {
    memset(&SD->u6.f21.rightChildTemp_data[0], 0, (unsigned int)(loop_ub * (int)
            sizeof(double)));
  }

  loop_ub = (int)M;
  if (0 <= loop_ub - 1) {
    memset(&leafNodeTemp_data[0], 0, (unsigned int)(loop_ub * (int)sizeof
            (boolean_T)));
  }

  emxInit_cell_wrap_13(&idxTemp, 1);
  upperBoundsTemp_data_tmp = idxTemp->size[0];
  idxTemp->size[0] = (int)M;
  emxEnsureCapacity_cell_wrap_13(idxTemp, upperBoundsTemp_data_tmp);
  idxTemp->data[0].f1.size[1] = m;
  idxTemp->data[0].f1.size[0] = 1;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < m;
       upperBoundsTemp_data_tmp++) {
    idxTemp->data[0].f1.data[upperBoundsTemp_data_tmp] = SD->
      u6.f21.notnan_data[upperBoundsTemp_data_tmp];
  }

  currentNode = 0;
  nextUnusedNode = 1;
  while (currentNode + 1 < nextUnusedNode + 1) {
    if (idxTemp->data[currentNode].f1.size[1] <= 50) {
      leafNodeTemp_data[currentNode] = true;
    } else {
      m = idxTemp->data[currentNode].f1.size[1];
      p = X_data[((int)idxTemp->data[currentNode].f1.data[0] - 1) << 1];
      maxval_idx_1 = X_data[1 + (((int)idxTemp->data[currentNode].f1.data[0] - 1)
        << 1)];
      for (loop_ub = 2; loop_ub <= m; loop_ub++) {
        if (p < X_data[((int)idxTemp->data[currentNode].f1.data[loop_ub - 1] - 1)
            << 1]) {
          p = X_data[((int)idxTemp->data[currentNode].f1.data[loop_ub - 1] - 1) <<
            1];
        }

        if (maxval_idx_1 < X_data[1 + (((int)idxTemp->data[currentNode]
              .f1.data[loop_ub - 1] - 1) << 1)]) {
          maxval_idx_1 = X_data[1 + (((int)idxTemp->data[currentNode]
            .f1.data[loop_ub - 1] - 1) << 1)];
        }
      }

      m = idxTemp->data[currentNode].f1.size[1];
      minval_idx_0 = X_data[((int)idxTemp->data[currentNode].f1.data[0] - 1) <<
        1];
      minval_idx_1 = X_data[1 + (((int)idxTemp->data[currentNode].f1.data[0] - 1)
        << 1)];
      for (loop_ub = 2; loop_ub <= m; loop_ub++) {
        if (minval_idx_0 > X_data[((int)idxTemp->data[currentNode]
             .f1.data[loop_ub - 1] - 1) << 1]) {
          minval_idx_0 = X_data[((int)idxTemp->data[currentNode].f1.data[loop_ub
            - 1] - 1) << 1];
        }

        if (minval_idx_1 > X_data[1 + (((int)idxTemp->data[currentNode]
              .f1.data[loop_ub - 1] - 1) << 1)]) {
          minval_idx_1 = X_data[1 + (((int)idxTemp->data[currentNode]
            .f1.data[loop_ub - 1] - 1) << 1)];
        }
      }

      p -= minval_idx_0;
      maxval_idx_1 -= minval_idx_1;
      iidx = (p < maxval_idx_1);
      m = idxTemp->data[currentNode].f1.size[1];
      x_size[0] = m;
      for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < m;
           upperBoundsTemp_data_tmp++) {
        SD->u6.f21.x_data[upperBoundsTemp_data_tmp] = X_data[iidx + (((int)
          idxTemp->data[currentNode].f1.data[upperBoundsTemp_data_tmp] - 1) << 1)];
      }

      sort(SD->u6.f21.x_data, x_size, SD->u6.f21.iidx_data, iidx_size);
      loop_ub = iidx_size[0];
      for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
           upperBoundsTemp_data_tmp++) {
        SD->u6.f21.tmp_data[upperBoundsTemp_data_tmp] = (int)idxTemp->
          data[currentNode].f1.data[SD->
          u6.f21.iidx_data[upperBoundsTemp_data_tmp] - 1];
      }

      half = (int)std::ceil((double)x_size[0] / 2.0);
      p = (SD->u6.f21.x_data[half - 1] + SD->u6.f21.x_data[half]) / 2.0F;
      SD->u6.f21.cutDimTemp_data[currentNode] = iidx + 1;
      SD->u6.f21.cutValTemp_data[currentNode] = p;
      M = nextUnusedNode + 1;
      SD->u6.f21.leftChildTemp_data[currentNode] = M;
      SD->u6.f21.rightChildTemp_data[currentNode] = M + 1.0;
      m = currentNode << 1;
      temp[0] = SD->u6.f21.upperBoundsTemp_data[m];
      loop_ub = 1 + m;
      M = SD->u6.f21.upperBoundsTemp_data[loop_ub];
      temp[1] = SD->u6.f21.upperBoundsTemp_data[loop_ub];
      b_upperBoundsTemp_data_tmp = (nextUnusedNode + 1) << 1;
      SD->u6.f21.upperBoundsTemp_data[b_upperBoundsTemp_data_tmp] =
        SD->u6.f21.upperBoundsTemp_data[m];
      loop_ub_tmp = 1 + b_upperBoundsTemp_data_tmp;
      SD->u6.f21.upperBoundsTemp_data[loop_ub_tmp] = M;
      temp[iidx] = p;
      SD->u6.f21.upperBoundsTemp_data[nextUnusedNode << 1] = temp[0];
      temp[0] = SD->u6.f21.lowerBoundsTemp_data[currentNode << 1];
      upperBoundsTemp_data_tmp = 1 + (nextUnusedNode << 1);
      SD->u6.f21.upperBoundsTemp_data[upperBoundsTemp_data_tmp] = temp[1];
      M = SD->u6.f21.lowerBoundsTemp_data[loop_ub];
      temp[1] = SD->u6.f21.lowerBoundsTemp_data[1 + (currentNode << 1)];
      SD->u6.f21.lowerBoundsTemp_data[nextUnusedNode << 1] =
        SD->u6.f21.lowerBoundsTemp_data[m];
      SD->u6.f21.lowerBoundsTemp_data[upperBoundsTemp_data_tmp] = M;
      temp[iidx] = p;
      SD->u6.f21.lowerBoundsTemp_data[b_upperBoundsTemp_data_tmp] = temp[0];
      SD->u6.f21.lowerBoundsTemp_data[loop_ub_tmp] = temp[1];
      idxTemp->data[currentNode].f1.size[1] = 0;
      idxTemp->data[currentNode].f1.size[0] = 1;
      for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < half;
           upperBoundsTemp_data_tmp++) {
        SD->u6.f21.b_tmp_data[upperBoundsTemp_data_tmp] =
          upperBoundsTemp_data_tmp;
      }

      idxTemp->data[nextUnusedNode].f1.size[1] = half;
      idxTemp->data[nextUnusedNode].f1.size[0] = 1;
      for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < half;
           upperBoundsTemp_data_tmp++) {
        idxTemp->data[nextUnusedNode].f1.data[upperBoundsTemp_data_tmp] =
          SD->u6.f21.tmp_data[SD->u6.f21.b_tmp_data[upperBoundsTemp_data_tmp]];
      }

      if (half + 1 > iidx_size[0]) {
        upperBoundsTemp_data_tmp = 1;
        b_upperBoundsTemp_data_tmp = 0;
      } else {
        upperBoundsTemp_data_tmp = half + 1;
        b_upperBoundsTemp_data_tmp = iidx_size[0];
      }

      m = (b_upperBoundsTemp_data_tmp - upperBoundsTemp_data_tmp) + 1;
      for (b_upperBoundsTemp_data_tmp = 0; b_upperBoundsTemp_data_tmp < m;
           b_upperBoundsTemp_data_tmp++) {
        SD->u6.f21.iidx_data[b_upperBoundsTemp_data_tmp] =
          (upperBoundsTemp_data_tmp + b_upperBoundsTemp_data_tmp) - 1;
      }

      idxTemp->data[nextUnusedNode + 1].f1.size[1] = m;
      idxTemp->data[nextUnusedNode + 1].f1.size[0] = 1;
      for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < m;
           upperBoundsTemp_data_tmp++) {
        idxTemp->data[nextUnusedNode + 1].f1.data[upperBoundsTemp_data_tmp] =
          SD->u6.f21.tmp_data[SD->u6.f21.iidx_data[upperBoundsTemp_data_tmp]];
      }

      nextUnusedNode += 2;
    }

    currentNode++;
  }

  loop_ub_tmp = (short)nextUnusedNode - 1;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp <= loop_ub_tmp;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.c_tmp_data[upperBoundsTemp_data_tmp] = (short)
      upperBoundsTemp_data_tmp;
  }

  loop_ub = (short)nextUnusedNode;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    cgstruct_cutDim_data[upperBoundsTemp_data_tmp] = (signed char)
      SD->u6.f21.cutDimTemp_data[SD->u6.f21.c_tmp_data[upperBoundsTemp_data_tmp]];
  }

  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp <= loop_ub_tmp;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.c_tmp_data[upperBoundsTemp_data_tmp] = (short)
      upperBoundsTemp_data_tmp;
  }

  loop_ub = (short)nextUnusedNode;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.cgstruct_cutVal_data[upperBoundsTemp_data_tmp] =
      SD->u6.f21.cutValTemp_data[SD->u6.f21.c_tmp_data[upperBoundsTemp_data_tmp]];
  }

  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < nextUnusedNode;
       upperBoundsTemp_data_tmp++) {
    m = upperBoundsTemp_data_tmp << 1;
    SD->u6.f21.cgstruct_lowerBounds_data[m] = SD->u6.f21.lowerBoundsTemp_data[m];
    m++;
    SD->u6.f21.cgstruct_lowerBounds_data[m] = SD->u6.f21.lowerBoundsTemp_data[m];
  }

  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < nextUnusedNode;
       upperBoundsTemp_data_tmp++) {
    m = upperBoundsTemp_data_tmp << 1;
    SD->u6.f21.cgstruct_upperBounds_data[m] = SD->u6.f21.upperBoundsTemp_data[m];
    m++;
    SD->u6.f21.cgstruct_upperBounds_data[m] = SD->u6.f21.upperBoundsTemp_data[m];
  }

  if (0 <= X_size[0] - 1) {
    memset(&SD->u6.f21.tempIdx_data[0], 0, (unsigned int)(X_size[0] * (int)
            sizeof(unsigned int)));
  }

  M = 1.0;
  for (m = 0; m < nextUnusedNode; m++) {
    SD->u6.f21.cutDimTemp_data[m] = idxTemp->data[m].f1.size[1];
    if (SD->u6.f21.cutDimTemp_data[m] > 0) {
      d0 = M + (double)SD->u6.f21.cutDimTemp_data[m];
      if (M > d0 - 1.0) {
        upperBoundsTemp_data_tmp = 0;
        b_upperBoundsTemp_data_tmp = 0;
      } else {
        upperBoundsTemp_data_tmp = (int)M - 1;
        b_upperBoundsTemp_data_tmp = (int)(d0 - 1.0);
      }

      loop_ub = b_upperBoundsTemp_data_tmp - upperBoundsTemp_data_tmp;
      for (b_upperBoundsTemp_data_tmp = 0; b_upperBoundsTemp_data_tmp < loop_ub;
           b_upperBoundsTemp_data_tmp++) {
        SD->u6.f21.iidx_data[b_upperBoundsTemp_data_tmp] =
          upperBoundsTemp_data_tmp + b_upperBoundsTemp_data_tmp;
      }

      for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
           upperBoundsTemp_data_tmp++) {
        SD->u6.f21.tempIdx_data[SD->u6.f21.iidx_data[upperBoundsTemp_data_tmp]] =
          (unsigned int)idxTemp->data[m].f1.data[upperBoundsTemp_data_tmp];
      }

      M = d0;
    }
  }

  emxFree_cell_wrap_13(&idxTemp);
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp <= loop_ub_tmp;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.c_tmp_data[upperBoundsTemp_data_tmp] = (short)
      upperBoundsTemp_data_tmp;
  }

  loop_ub = (short)nextUnusedNode;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.cgstruct_leftChild_data[upperBoundsTemp_data_tmp] =
      SD->u6.f21.leftChildTemp_data[SD->
      u6.f21.c_tmp_data[upperBoundsTemp_data_tmp]];
  }

  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp <= loop_ub_tmp;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.c_tmp_data[upperBoundsTemp_data_tmp] = (short)
      upperBoundsTemp_data_tmp;
  }

  loop_ub = (short)nextUnusedNode;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.cutValTemp_data[upperBoundsTemp_data_tmp] =
      SD->u6.f21.rightChildTemp_data[SD->
      u6.f21.c_tmp_data[upperBoundsTemp_data_tmp]];
  }

  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp <= loop_ub_tmp;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.c_tmp_data[upperBoundsTemp_data_tmp] = (short)
      upperBoundsTemp_data_tmp;
  }

  loop_ub = (short)nextUnusedNode;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    cgstruct_leafNode_data[upperBoundsTemp_data_tmp] = leafNodeTemp_data
      [SD->u6.f21.c_tmp_data[upperBoundsTemp_data_tmp]];
  }

  SD->u6.f21.expl_temp.idxDim.size[0] = nextUnusedNode;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < nextUnusedNode;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.expl_temp.idxDim.data[upperBoundsTemp_data_tmp] =
      SD->u6.f21.cutDimTemp_data[upperBoundsTemp_data_tmp];
  }

  SD->u6.f21.expl_temp.idxAll.size[0] = X_size[0];
  if (0 <= X_size[0] - 1) {
    memcpy(&SD->u6.f21.expl_temp.idxAll.data[0], &SD->u6.f21.tempIdx_data[0],
           (unsigned int)(X_size[0] * (int)sizeof(unsigned int)));
  }

  SD->u6.f21.expl_temp.nx_nonan = X_size[0];
  SD->u6.f21.expl_temp.leafNode.size[0] = (short)nextUnusedNode;
  loop_ub = (short)nextUnusedNode;
  if (0 <= loop_ub - 1) {
    memcpy(&SD->u6.f21.expl_temp.leafNode.data[0], &cgstruct_leafNode_data[0],
           (unsigned int)(loop_ub * (int)sizeof(boolean_T)));
  }

  SD->u6.f21.expl_temp.rightChild.size[0] = (short)nextUnusedNode;
  loop_ub = (short)nextUnusedNode;
  if (0 <= loop_ub - 1) {
    memcpy(&SD->u6.f21.expl_temp.rightChild.data[0], &SD->
           u6.f21.cutValTemp_data[0], (unsigned int)(loop_ub * (int)sizeof
            (double)));
  }

  SD->u6.f21.expl_temp.leftChild.size[0] = (short)nextUnusedNode;
  loop_ub = (short)nextUnusedNode;
  if (0 <= loop_ub - 1) {
    memcpy(&SD->u6.f21.expl_temp.leftChild.data[0],
           &SD->u6.f21.cgstruct_leftChild_data[0], (unsigned int)(loop_ub * (int)
            sizeof(double)));
  }

  SD->u6.f21.expl_temp.upperBounds.size[1] = 2;
  SD->u6.f21.expl_temp.upperBounds.size[0] = nextUnusedNode;
  loop_ub_tmp = nextUnusedNode << 1;
  if (0 <= loop_ub_tmp - 1) {
    memcpy(&SD->u6.f21.expl_temp.upperBounds.data[0],
           &SD->u6.f21.cgstruct_upperBounds_data[0], (unsigned int)(loop_ub_tmp *
            (int)sizeof(double)));
  }

  SD->u6.f21.expl_temp.lowerBounds.size[1] = 2;
  SD->u6.f21.expl_temp.lowerBounds.size[0] = nextUnusedNode;
  if (0 <= loop_ub_tmp - 1) {
    memcpy(&SD->u6.f21.expl_temp.lowerBounds.data[0],
           &SD->u6.f21.cgstruct_lowerBounds_data[0], (unsigned int)(loop_ub_tmp *
            (int)sizeof(double)));
  }

  SD->u6.f21.expl_temp.cutVal.size[0] = (short)nextUnusedNode;
  loop_ub = (short)nextUnusedNode;
  if (0 <= loop_ub - 1) {
    memcpy(&SD->u6.f21.expl_temp.cutVal.data[0],
           &SD->u6.f21.cgstruct_cutVal_data[0], (unsigned int)(loop_ub * (int)
            sizeof(double)));
  }

  SD->u6.f21.expl_temp.cutDim.size[0] = (short)nextUnusedNode;
  loop_ub = (short)nextUnusedNode;
  for (upperBoundsTemp_data_tmp = 0; upperBoundsTemp_data_tmp < loop_ub;
       upperBoundsTemp_data_tmp++) {
    SD->u6.f21.expl_temp.cutDim.data[upperBoundsTemp_data_tmp] =
      cgstruct_cutDim_data[upperBoundsTemp_data_tmp];
  }

  SD->u6.f21.expl_temp.numNodes = nextUnusedNode;
  SD->u6.f21.expl_temp.wasnanIdx.size[1] = 0;
  SD->u6.f21.expl_temp.wasnanIdx.size[0] = 1;
  SD->u6.f21.expl_temp.BucketSize = 50.0;
  SD->u6.f21.expl_temp.Distance[0] = 'e';
  SD->u6.f21.expl_temp.Distance[1] = 'u';
  SD->u6.f21.expl_temp.Distance[2] = 'c';
  SD->u6.f21.expl_temp.Distance[3] = 'l';
  SD->u6.f21.expl_temp.Distance[4] = 'i';
  SD->u6.f21.expl_temp.Distance[5] = 'd';
  SD->u6.f21.expl_temp.Distance[6] = 'e';
  SD->u6.f21.expl_temp.Distance[7] = 'a';
  SD->u6.f21.expl_temp.Distance[8] = 'n';
  SD->u6.f21.expl_temp.X.size[1] = 2;
  SD->u6.f21.expl_temp.X.size[0] = X_size[0];
  loop_ub = X_size[1] * X_size[0];
  if (0 <= loop_ub - 1) {
    memcpy(&SD->u6.f21.expl_temp.X.data[0], &X_data[0], (unsigned int)(loop_ub *
            (int)sizeof(float)));
  }

  emxInit_real32_T(&D, 2);
  kdsearchfun(SD, &SD->u6.f21.expl_temp, Y, Idx, D);
  emxFree_real32_T(&D);
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float X_data[]
//                const int X_size[2]
//                int idxbest_data[]
//                int idxbest_size[1]
//                float Cbest[6]
//                float varargout_1[2]
//                float varargout_2_data[]
//                int varargout_2_size[2]
// Return Type  : void
//
static void local_kmeans(PerceptionSmartLoaderStackData *SD, const float X_data[],
  const int X_size[2], int idxbest_data[], int idxbest_size[1], float Cbest[6],
  float varargout_1[2], float varargout_2_data[], int varargout_2_size[2])
{
  float totsumDbest;
  int rep;
  float totsumD;
  int idx_size[1];
  float C[6];
  float sumD[2];
  int D_size[2];
  int loop_ub;
  loopBody(SD, X_data, X_size, &totsumDbest, idxbest_data, idxbest_size, Cbest,
           varargout_1, varargout_2_data, varargout_2_size);
  for (rep = 0; rep < 4; rep++) {
    loopBody(SD, X_data, X_size, &totsumD, SD->u4.f17.idx_data, idx_size, C,
             sumD, SD->u4.f17.D_data, D_size);
    if (totsumD < totsumDbest) {
      totsumDbest = totsumD;
      idxbest_size[0] = idx_size[0];
      if (0 <= idx_size[0] - 1) {
        memcpy(&idxbest_data[0], &SD->u4.f17.idx_data[0], (unsigned int)
               (idx_size[0] * (int)sizeof(int)));
      }

      Cbest[0] = C[0];
      Cbest[1] = C[1];
      Cbest[2] = C[2];
      Cbest[3] = C[3];
      Cbest[4] = C[4];
      Cbest[5] = C[5];
      varargout_1[0] = sumD[0];
      varargout_1[1] = sumD[1];
      varargout_2_size[1] = 2;
      varargout_2_size[0] = D_size[0];
      loop_ub = D_size[1] * D_size[0];
      if (0 <= loop_ub - 1) {
        memcpy(&varargout_2_data[0], &SD->u4.f17.D_data[0], (unsigned int)
               (loop_ub * (int)sizeof(float)));
      }
    }
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float X_data[]
//                const int X_size[2]
//                float *totsumD
//                int idx_data[]
//                int idx_size[1]
//                float C[6]
//                float sumD[2]
//                float D_data[]
//                int D_size[2]
// Return Type  : void
//
static void loopBody(PerceptionSmartLoaderStackData *SD, const float X_data[],
                     const int X_size[2], float *totsumD, int idx_data[], int
                     idx_size[1], float C[6], float sumD[2], float D_data[], int
                     D_size[2])
{
  int n;
  double b_index;
  int pidx;
  int nNonEmpty;
  int sampleDist_size[1];
  boolean_T DNeedsComputing;
  float denominator;
  float f2;
  int crows[2];
  int nonEmpties[2];
  n = X_size[0] - 1;
  b_index = b_rand(SD);
  C[3] = 0.0F;
  C[4] = 0.0F;
  C[5] = 0.0F;
  pidx = (int)(1.0 + std::floor(b_index * (double)X_size[0]));
  C[0] = X_data[3 * (pidx - 1)];
  C[1] = X_data[1 + 3 * (pidx - 1)];
  C[2] = X_data[2 + 3 * (pidx - 1)];
  D_size[1] = 2;
  D_size[0] = X_size[0];
  nNonEmpty = X_size[0] << 1;
  if (0 <= nNonEmpty - 1) {
    memset(&D_data[0], 0, (unsigned int)(nNonEmpty * (int)sizeof(float)));
  }

  distfun(D_data, X_data, X_size, C, 1);
  nNonEmpty = X_size[0];
  for (pidx = 0; pidx < nNonEmpty; pidx++) {
    SD->u3.f16.d_data[pidx] = D_data[pidx << 1];
  }

  idx_size[0] = X_size[0];
  nNonEmpty = X_size[0];
  for (pidx = 0; pidx < nNonEmpty; pidx++) {
    idx_data[pidx] = 1;
  }

  sampleDist_size[0] = X_size[0] + 1;
  if (0 <= X_size[0]) {
    memset(&SD->u3.f16.sampleDist_data[0], 0, (unsigned int)((X_size[0] + 1) *
            (int)sizeof(float)));
  }

  DNeedsComputing = false;
  denominator = 0.0F;
  SD->u3.f16.sampleDist_data[0] = 0.0F;
  for (pidx = 0; pidx <= n; pidx++) {
    f2 = D_data[pidx << 1];
    SD->u3.f16.sampleDist_data[pidx + 1] = SD->u3.f16.sampleDist_data[pidx] + f2;
    denominator += f2;
  }

  if (denominator == 0.0F) {
    simpleRandperm(SD, X_size[0], idx_data, idx_size);
    pidx = 3 * (idx_data[0] - 1);
    C[3] = X_data[pidx];
    C[4] = X_data[1 + pidx];
    C[5] = X_data[2 + pidx];
    DNeedsComputing = true;
  } else {
    nNonEmpty = X_size[0] + 1;
    for (pidx = 0; pidx < nNonEmpty; pidx++) {
      SD->u3.f16.sampleDist_data[pidx] /= denominator;
    }

    pidx = b_bsearch(SD->u3.f16.sampleDist_data, sampleDist_size, b_rand(SD));
    denominator = SD->u3.f16.sampleDist_data[pidx - 1];
    if (SD->u3.f16.sampleDist_data[pidx - 1] < 1.0F) {
      while ((pidx <= n + 1) && (SD->u3.f16.sampleDist_data[pidx] <= denominator))
      {
        pidx++;
      }
    } else {
      while ((pidx >= 2) && (SD->u3.f16.sampleDist_data[pidx - 2] >= denominator))
      {
        pidx--;
      }
    }

    pidx = 3 * (pidx - 1);
    C[3] = X_data[pidx];
    C[4] = X_data[1 + pidx];
    C[5] = X_data[2 + pidx];
    distfun(D_data, X_data, X_size, C, 2);
    for (pidx = 0; pidx <= n; pidx++) {
      f2 = D_data[1 + (pidx << 1)];
      if (f2 < SD->u3.f16.d_data[pidx]) {
        SD->u3.f16.d_data[pidx] = f2;
        idx_data[pidx] = 2;
      }
    }
  }

  if (DNeedsComputing) {
    crows[0] = 1;
    crows[1] = 2;
    b_distfun(D_data, X_data, X_size, C, crows, 2);
    mindim2(D_data, D_size, SD->u3.f16.d_data, sampleDist_size, idx_data,
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
    SD->u3.f16.d_data[pidx] = D_data[(idx_data[pidx] + (pidx << 1)) - 1];
  }

  sumD[0] = 0.0F;
  sumD[1] = 0.0F;
  for (pidx = 0; pidx <= n; pidx++) {
    sumD[idx_data[pidx] - 1] += SD->u3.f16.d_data[pidx];
  }

  *totsumD = 0.0F;
  for (pidx = 0; pidx <= nNonEmpty; pidx++) {
    *totsumD += sumD[nonEmpties[pidx] - 1];
  }
}

//
// Arguments    : const float x_data[]
//                const int x_size[2]
//                float y[3]
// Return Type  : void
//
static void mean(const float x_data[], const int x_size[2], float y[3])
{
  int vlen;
  int firstBlockLength;
  int nblocks;
  int lastBlockLength;
  int k;
  int offset;
  float bsum[3];
  int hi;
  int varargin_1;
  vlen = x_size[0];
  if (x_size[0] == 0) {
    y[0] = 0.0F;
    y[1] = 0.0F;
    y[2] = 0.0F;
  } else {
    if (x_size[0] <= 1024) {
      firstBlockLength = x_size[0];
      lastBlockLength = 0;
      nblocks = 1;
    } else {
      firstBlockLength = 1024;
      nblocks = x_size[0] / 1024;
      lastBlockLength = x_size[0] - (nblocks << 10);
      if (lastBlockLength > 0) {
        nblocks++;
      } else {
        lastBlockLength = 1024;
      }
    }

    y[0] = x_data[0];
    y[1] = x_data[1];
    y[2] = x_data[2];
    for (k = 2; k <= firstBlockLength; k++) {
      if (vlen >= 2) {
        y[0] += x_data[3 * (k - 1)];
        y[1] += x_data[1 + 3 * (k - 1)];
        y[2] += x_data[2 + 3 * (k - 1)];
      }
    }

    for (firstBlockLength = 2; firstBlockLength <= nblocks; firstBlockLength++)
    {
      offset = (firstBlockLength - 1) << 10;
      bsum[0] = x_data[3 * offset];
      bsum[1] = x_data[3 * offset + 1];
      bsum[2] = x_data[3 * offset + 2];
      if (firstBlockLength == nblocks) {
        hi = lastBlockLength;
      } else {
        hi = 1024;
      }

      for (k = 2; k <= hi; k++) {
        varargin_1 = offset + k;
        if (vlen >= 2) {
          bsum[0] += x_data[3 * (varargin_1 - 1)];
          bsum[1] += x_data[3 * (varargin_1 - 1) + 1];
          bsum[2] += x_data[3 * (varargin_1 - 1) + 2];
        }
      }

      y[0] += bsum[0];
      y[1] += bsum[1];
      y[2] += bsum[2];
    }
  }

  y[0] /= (float)x_size[0];
  y[1] /= (float)x_size[0];
  y[2] /= (float)x_size[0];
}

//
// Arguments    : float v_data[]
//                int nv
//                int ia
//                int ib
// Return Type  : void
//
static void med3(float v_data[], int nv, int ia, int ib)
{
  int ic;
  float tmp;
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

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float x_data[]
//                const int x_size[1]
// Return Type  : float
//
static float median(PerceptionSmartLoaderStackData *SD, const float x_data[],
                    const int x_size[1])
{
  float y;
  int b_x_size[1];
  if (x_size[0] == 0) {
    y = 0.0F;
  } else {
    b_x_size[0] = x_size[0];
    if (0 <= x_size[0] - 1) {
      memcpy(&SD->u2.f12.x_data[0], &x_data[0], (unsigned int)(x_size[0] * (int)
              sizeof(float)));
    }

    y = vmedian(SD, SD->u2.f12.x_data, b_x_size, x_size[0]);
  }

  return y;
}

//
// Arguments    : float v_data[]
//                int nv
//                int ia
// Return Type  : void
//
static void medmed(float v_data[], int nv, int ia)
{
  int ngroupsof5;
  int nlast;
  int k;
  int i1;
  int destidx;
  float tmp;
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

//
// Arguments    : int idx_data[]
//                float x_data[]
//                int offset
//                int np
//                int nq
//                int iwork_data[]
//                float xwork_data[]
// Return Type  : void
//
static void merge(int idx_data[], float x_data[], int offset, int np, int nq,
                  int iwork_data[], float xwork_data[])
{
  int n_tmp;
  int iout;
  int p;
  int i42;
  int q;
  int exitg1;
  if (nq != 0) {
    n_tmp = np + nq;
    for (iout = 0; iout < n_tmp; iout++) {
      i42 = offset + iout;
      iwork_data[iout] = idx_data[i42];
      xwork_data[iout] = x_data[i42];
    }

    p = 0;
    q = np;
    iout = offset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork_data[p] <= xwork_data[q]) {
        idx_data[iout] = iwork_data[p];
        x_data[iout] = xwork_data[p];
        if (p + 1 < np) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork_data[q];
        x_data[iout] = xwork_data[q];
        if (q + 1 < n_tmp) {
          q++;
        } else {
          q = iout - p;
          for (iout = p + 1; iout <= np; iout++) {
            i42 = q + iout;
            idx_data[i42] = iwork_data[iout - 1];
            x_data[i42] = xwork_data[iout - 1];
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

//
// Arguments    : const float D1_data[]
//                const int D1_size[1]
//                const float D2_data[]
//                const int D2_size[1]
//                const unsigned int I1_data[]
//                const unsigned int I2_data[]
//                int N
//                float dOut_data[]
//                int dOut_size[1]
//                unsigned int iOut_data[]
//                int iOut_size[1]
// Return Type  : void
//
static void mergeSort(const float D1_data[], const int D1_size[1], const float
                      D2_data[], const int D2_size[1], const unsigned int
                      I1_data[], const unsigned int I2_data[], int N, float
                      dOut_data[], int dOut_size[1], unsigned int iOut_data[],
                      int iOut_size[1])
{
  int cD2;
  int uBound;
  int c;
  boolean_T exitg1;
  int cc;
  cD2 = 0;
  uBound = D1_size[0] + D2_size[0];
  if (uBound > N) {
    uBound = N;
  }

  dOut_size[0] = uBound;
  if (0 <= uBound - 1) {
    memset(&dOut_data[0], 0, (unsigned int)(uBound * (int)sizeof(float)));
  }

  iOut_size[0] = uBound;
  if (0 <= uBound - 1) {
    memset(&iOut_data[0], 0, (unsigned int)(uBound * (int)sizeof(unsigned int)));
  }

  c = 0;
  exitg1 = false;
  while ((!exitg1) && (c <= uBound - 1)) {
    if (D1_data[0] <= D2_data[cD2]) {
      dOut_data[c] = D1_data[0];
      iOut_data[c] = I1_data[0];
      for (cc = c + 2; cc <= uBound; cc++) {
        dOut_data[cc - 1] = D2_data[cD2];
        iOut_data[cc - 1] = I2_data[cD2];
        cD2++;
      }

      exitg1 = true;
    } else {
      dOut_data[c] = D2_data[cD2];
      iOut_data[c] = I2_data[cD2];
      cD2++;
      if (cD2 + 1 > D2_size[0]) {
        for (cc = c + 2; cc <= uBound; cc++) {
          dOut_data[cc - 1] = D1_data[0];
          iOut_data[cc - 1] = I1_data[0];
        }

        exitg1 = true;
      } else {
        c++;
      }
    }
  }
}

//
// Arguments    : int idx_data[]
//                float x_data[]
//                int offset
//                int n
//                int preSortLevel
//                int iwork_data[]
//                float xwork_data[]
// Return Type  : void
//
static void merge_block(int idx_data[], float x_data[], int offset, int n, int
  preSortLevel, int iwork_data[], float xwork_data[])
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
        merge(idx_data, x_data, offset + tailOffset, bLen, nTail - bLen,
              iwork_data, xwork_data);
      }
    }

    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 0; nTail < nPairs; nTail++) {
      merge(idx_data, x_data, offset + nTail * tailOffset, bLen, bLen,
            iwork_data, xwork_data);
    }

    bLen = tailOffset;
  }

  if (n > bLen) {
    merge(idx_data, x_data, offset, bLen, n - bLen, iwork_data, xwork_data);
  }
}

//
// Arguments    : int idx_data[]
//                float x_data[]
//                int offset
// Return Type  : void
//
static void merge_pow2_block(int idx_data[], float x_data[], int offset)
{
  int k;
  int blockOffset;
  int j;
  int p;
  int iout;
  int q;
  int iwork[256];
  float xwork[256];
  int exitg1;
  for (k = 0; k < 32; k++) {
    blockOffset = offset + k * 8;
    for (j = 0; j < 8; j++) {
      iout = blockOffset + j;
      iwork[j] = idx_data[iout];
      xwork[j] = x_data[iout];
    }

    p = 0;
    q = 4;
    iout = blockOffset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork[p] <= xwork[q]) {
        idx_data[iout] = iwork[p];
        x_data[iout] = xwork[p];
        if (p + 1 < 4) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork[q];
        x_data[iout] = xwork[q];
        if (q + 1 < 8) {
          q++;
        } else {
          iout -= p;
          for (j = p + 1; j < 5; j++) {
            q = iout + j;
            idx_data[q] = iwork[j - 1];
            x_data[q] = xwork[j - 1];
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }

  for (k = 0; k < 16; k++) {
    blockOffset = offset + k * 16;
    for (j = 0; j < 16; j++) {
      iout = blockOffset + j;
      iwork[j] = idx_data[iout];
      xwork[j] = x_data[iout];
    }

    p = 0;
    q = 8;
    iout = blockOffset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork[p] <= xwork[q]) {
        idx_data[iout] = iwork[p];
        x_data[iout] = xwork[p];
        if (p + 1 < 8) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork[q];
        x_data[iout] = xwork[q];
        if (q + 1 < 16) {
          q++;
        } else {
          iout -= p;
          for (j = p + 1; j < 9; j++) {
            q = iout + j;
            idx_data[q] = iwork[j - 1];
            x_data[q] = xwork[j - 1];
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }

  for (k = 0; k < 8; k++) {
    blockOffset = offset + k * 32;
    for (j = 0; j < 32; j++) {
      iout = blockOffset + j;
      iwork[j] = idx_data[iout];
      xwork[j] = x_data[iout];
    }

    p = 0;
    q = 16;
    iout = blockOffset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork[p] <= xwork[q]) {
        idx_data[iout] = iwork[p];
        x_data[iout] = xwork[p];
        if (p + 1 < 16) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork[q];
        x_data[iout] = xwork[q];
        if (q + 1 < 32) {
          q++;
        } else {
          iout -= p;
          for (j = p + 1; j < 17; j++) {
            q = iout + j;
            idx_data[q] = iwork[j - 1];
            x_data[q] = xwork[j - 1];
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }

  for (k = 0; k < 4; k++) {
    blockOffset = offset + k * 64;
    for (j = 0; j < 64; j++) {
      iout = blockOffset + j;
      iwork[j] = idx_data[iout];
      xwork[j] = x_data[iout];
    }

    p = 0;
    q = 32;
    iout = blockOffset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork[p] <= xwork[q]) {
        idx_data[iout] = iwork[p];
        x_data[iout] = xwork[p];
        if (p + 1 < 32) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork[q];
        x_data[iout] = xwork[q];
        if (q + 1 < 64) {
          q++;
        } else {
          iout -= p;
          for (j = p + 1; j < 33; j++) {
            q = iout + j;
            idx_data[q] = iwork[j - 1];
            x_data[q] = xwork[j - 1];
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }

  for (k = 0; k < 2; k++) {
    blockOffset = offset + k * 128;
    for (j = 0; j < 128; j++) {
      iout = blockOffset + j;
      iwork[j] = idx_data[iout];
      xwork[j] = x_data[iout];
    }

    p = 0;
    q = 64;
    iout = blockOffset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork[p] <= xwork[q]) {
        idx_data[iout] = iwork[p];
        x_data[iout] = xwork[p];
        if (p + 1 < 64) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork[q];
        x_data[iout] = xwork[q];
        if (q + 1 < 128) {
          q++;
        } else {
          iout -= p;
          for (j = p + 1; j < 65; j++) {
            q = iout + j;
            idx_data[q] = iwork[j - 1];
            x_data[q] = xwork[j - 1];
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }

  for (k = 0; k < 1; k++) {
    for (j = 0; j < 256; j++) {
      iout = offset + j;
      iwork[j] = idx_data[iout];
      xwork[j] = x_data[iout];
    }

    p = 0;
    q = 128;
    iout = offset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork[p] <= xwork[q]) {
        idx_data[iout] = iwork[p];
        x_data[iout] = xwork[p];
        if (p + 1 < 128) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork[q];
        x_data[iout] = xwork[q];
        if (q + 1 < 256) {
          q++;
        } else {
          iout -= p;
          for (j = p + 1; j < 129; j++) {
            q = iout + j;
            idx_data[q] = iwork[j - 1];
            x_data[q] = xwork[j - 1];
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

//
// Arguments    : const float D_data[]
//                const int D_size[2]
//                float d_data[]
//                int d_size[1]
//                int idx_data[]
//                int idx_size[1]
// Return Type  : void
//
static void mindim2(const float D_data[], const int D_size[2], float d_data[],
                    int d_size[1], int idx_data[], int idx_size[1])
{
  int n;
  int loop_ub;
  int i8;
  float f3;
  n = D_size[0];
  repmat(D_size[0], d_data, d_size);
  idx_size[0] = D_size[0];
  loop_ub = D_size[0];
  for (i8 = 0; i8 < loop_ub; i8++) {
    idx_data[i8] = 1;
  }

  for (loop_ub = 0; loop_ub < n; loop_ub++) {
    i8 = loop_ub << 1;
    if (D_data[i8] < d_data[loop_ub]) {
      idx_data[loop_ub] = 1;
      d_data[loop_ub] = D_data[i8];
    }
  }

  for (loop_ub = 0; loop_ub < n; loop_ub++) {
    f3 = D_data[1 + (loop_ub << 1)];
    if (f3 < d_data[loop_ub]) {
      idx_data[loop_ub] = 2;
      d_data[loop_ub] = f3;
    }
  }
}

//
// Arguments    : double b
// Return Type  : double
//
static double mpower(double b)
{
  return pow(2.0, b);
}

//
// Arguments    : const float x_data[]
//                const int x_size[2]
//                float y_data[]
//                int y_size[1]
// Return Type  : void
//
static void nestedIter(const float x_data[], const int x_size[2], float y_data[],
  int y_size[1])
{
  int i6;
  int k;
  int y_data_tmp;
  y_size[0] = x_size[0];
  i6 = x_size[0];
  for (k = 0; k < i6; k++) {
    y_data_tmp = k << 1;
    y_data[k] = x_data[y_data_tmp];
    y_data[k] += x_data[1 + y_data_tmp];
  }
}

//
// Arguments    : const int x_size[1]
// Return Type  : int
//
static int nonSingletonDim(const int x_size[1])
{
  int dim;
  dim = 2;
  if (x_size[0] != 1) {
    dim = 1;
  }

  return dim;
}

//
// Arguments    : const cell_wrap_4 x_data[]
//                const boolean_T idx_data[]
//                cell_wrap_4 b_x_data[]
//                int x_size[2]
// Return Type  : void
//
static void nullAssignment(const cell_wrap_4 x_data[], const boolean_T idx_data[],
  cell_wrap_4 b_x_data[], int x_size[2])
{
  int n;
  int k;
  int i16;
  int bidx;
  int loop_ub;
  n = 0;
  for (k = 0; k < 64; k++) {
    n += idx_data[k];
  }

  i16 = x_size[0] * x_size[1];
  x_size[1] = 64 - n;
  x_size[0] = 1;
  emxEnsureCapacity_cell_wrap_4(b_x_data, x_size, i16);
  bidx = 0;
  i16 = 63 - n;
  for (k = 0; k <= i16; k++) {
    while ((bidx + 1 <= 64) && idx_data[bidx]) {
      bidx++;
    }

    n = b_x_data[k].f1->size[0] * b_x_data[k].f1->size[1];
    b_x_data[k].f1->size[1] = x_data[bidx].f1->size[1];
    b_x_data[k].f1->size[0] = x_data[bidx].f1->size[0];
    emxEnsureCapacity_real32_T(b_x_data[k].f1, n);
    loop_ub = x_data[bidx].f1->size[1] * x_data[bidx].f1->size[0];
    for (n = 0; n < loop_ub; n++) {
      b_x_data[k].f1->data[n] = x_data[bidx].f1->data[n];
    }

    bidx++;
  }
}

//
// Arguments    : const emxArray_real32_T *Xin
//                emxArray_real32_T *Y
// Return Type  : void
//
static void pdist(const emxArray_real32_T *Xin, emxArray_real32_T *Y)
{
  emxArray_real32_T *X;
  int px;
  int nx;
  int i17;
  int loop_ub;
  int b_loop_ub;
  int i18;
  double d2;
  int kk;
  int jj;
  double qq;
  double ii;
  float tempSum;
  emxInit_real32_T(&X, 2);
  px = Xin->size[1];
  nx = Xin->size[0];
  i17 = X->size[0] * X->size[1];
  X->size[1] = Xin->size[0];
  X->size[0] = Xin->size[1];
  emxEnsureCapacity_real32_T(X, i17);
  loop_ub = Xin->size[1];
  for (i17 = 0; i17 < loop_ub; i17++) {
    b_loop_ub = Xin->size[0];
    for (i18 = 0; i18 < b_loop_ub; i18++) {
      X->data[i18 + X->size[1] * i17] = Xin->data[i17 + Xin->size[1] * i18];
    }
  }

  if (Xin->size[0] == 0) {
    Y->size[1] = 0;
    Y->size[0] = 1;
  } else {
    loop_ub = Xin->size[0] * (Xin->size[0] - 1) / 2;
    i17 = Y->size[0] * Y->size[1];
    Y->size[1] = loop_ub;
    Y->size[0] = 1;
    emxEnsureCapacity_real32_T(Y, i17);
    d2 = (double)Xin->size[0] * ((double)Xin->size[0] - 1.0) / 2.0;
    loop_ub = (int)d2 - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(jj,qq,ii,tempSum)

    for (kk = 0; kk <= loop_ub; kk++) {
      tempSum = 0.0F;
      ii = (((double)nx - 2.0) - std::floor(std::sqrt((-8.0 * ((1.0 + (double)kk)
                - 1.0) + 4.0 * (double)nx * ((double)nx - 1.0)) - 7.0) / 2.0 -
             0.5)) + 1.0;
      qq = (double)nx - ii;
      qq = (((1.0 + (double)kk) + ii) - (double)nx * ((double)nx - 1.0) / 2.0) +
        qq * (qq + 1.0) / 2.0;
      for (jj = 0; jj < px; jj++) {
        tempSum += (X->data[((int)qq + X->size[1] * jj) - 1] - X->data[((int)ii
          + X->size[1] * jj) - 1]) * (X->data[((int)qq + X->size[1] * jj) - 1] -
          X->data[((int)ii + X->size[1] * jj) - 1]);
      }

      Y->data[kk] = std::sqrt(tempSum);
    }
  }

  emxFree_real32_T(&X);
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const emxArray_real32_T *Xin
//                const float Yin_data[]
//                const int Yin_size[2]
//                emxArray_real32_T *D
// Return Type  : void
//
static void pdist2(PerceptionSmartLoaderStackData *SD, const emxArray_real32_T
                   *Xin, const float Yin_data[], const int Yin_size[2],
                   emxArray_real32_T *D)
{
  emxArray_real32_T *X;
  int nx;
  int ub_loop;
  int loop_ub;
  int Y_size[2];
  int b_loop_ub;
  int i15;
  unsigned int Xin_idx_0;
  int ii;
  int jj;
  float tempSum;
  emxInit_real32_T(&X, 2);
  nx = Xin->size[0];
  ub_loop = X->size[0] * X->size[1];
  X->size[1] = Xin->size[0];
  X->size[0] = Xin->size[1];
  emxEnsureCapacity_real32_T(X, ub_loop);
  loop_ub = Xin->size[1];
  for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
    b_loop_ub = Xin->size[0];
    for (i15 = 0; i15 < b_loop_ub; i15++) {
      X->data[i15 + X->size[1] * ub_loop] = Xin->data[ub_loop + Xin->size[1] *
        i15];
    }
  }

  Y_size[1] = Yin_size[0];
  loop_ub = Yin_size[0];
  for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
    SD->u1.f3.Y_data[ub_loop] = Yin_data[ub_loop << 1];
  }

  for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
    SD->u1.f3.Y_data[ub_loop + Y_size[1]] = Yin_data[1 + (ub_loop << 1)];
  }

  if ((Xin->size[0] == 0) || (Yin_size[0] == 0)) {
    ub_loop = D->size[0] * D->size[1];
    D->size[1] = Yin_size[0];
    D->size[0] = Xin->size[0];
    emxEnsureCapacity_real32_T(D, ub_loop);
    loop_ub = Yin_size[0] * Xin->size[0];
    for (ub_loop = 0; ub_loop < loop_ub; ub_loop++) {
      D->data[ub_loop] = 0.0F;
    }
  } else {
    Xin_idx_0 = (unsigned int)Xin->size[0];
    ub_loop = D->size[0] * D->size[1];
    D->size[1] = Yin_size[0];
    D->size[0] = (int)Xin_idx_0;
    emxEnsureCapacity_real32_T(D, ub_loop);
    ub_loop = Yin_size[0] - 1;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(jj,tempSum)

    for (ii = 0; ii <= ub_loop; ii++) {
      for (jj = 0; jj < nx; jj++) {
        tempSum = std::pow(X->data[jj] - SD->u1.f3.Y_data[ii], 2.0F);
        tempSum += std::pow(X->data[jj + X->size[1]] - SD->u1.f3.Y_data[ii +
                            Y_size[1]], 2.0F);
        D->data[ii + D->size[1] * jj] = std::sqrt(tempSum);
      }
    }
  }

  emxFree_real32_T(&X);
}

//
// Arguments    : float v_data[]
//                int *ip
//                int ia
//                int ib
// Return Type  : int
//
static int pivot(float v_data[], int *ip, int ia, int ib)
{
  int reps;
  float vref;
  int i59;
  int k;
  float vk_tmp;
  vref = v_data[*ip - 1];
  v_data[*ip - 1] = v_data[ib - 1];
  v_data[ib - 1] = vref;
  *ip = ia;
  reps = 0;
  i59 = ib - 1;
  for (k = ia; k <= i59; k++) {
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

//
// Arguments    : float v_data[]
//                int n
//                int vlen
//                float *vn
//                int *nfirst
//                int *nlast
// Return Type  : void
//
static void quickselect(float v_data[], int n, int vlen, float *vn, int *nfirst,
  int *nlast)
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
    *vn = 0.0F;
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

//
// Arguments    : int varargin_1
//                float b_data[]
//                int b_size[1]
// Return Type  : void
//
static void repmat(int varargin_1, float b_data[], int b_size[1])
{
  int i9;
  b_size[0] = varargin_1;
  for (i9 = 0; i9 < varargin_1; i9++) {
    b_data[i9] = 3.402823466E+38F;
  }
}

//
// Arguments    : const float b_data[]
//                int k0
//                int k
// Return Type  : boolean_T
//
static boolean_T rows_differ(const float b_data[], int k0, int k)
{
  boolean_T p;
  int j;
  boolean_T exitg1;
  float b;
  float absxk;
  int exponent;
  p = false;
  j = 0;
  exitg1 = false;
  while ((!exitg1) && (j < 2)) {
    b = b_data[j + ((k - 1) << 1)];
    absxk = std::abs(b / 2.0F);
    if (absxk <= 1.17549435E-38F) {
      absxk = 1.4013E-45F;
    } else {
      std::frexp(absxk, &exponent);
      absxk = std::ldexp(1.0F, exponent - 24);
    }

    if (std::abs(b - b_data[j + ((k0 - 1) << 1)]) >= absxk) {
      p = true;
      exitg1 = true;
    } else {
      j++;
    }
  }

  return p;
}

//
// Arguments    : double u0
//                double u1
// Return Type  : double
//
static double rt_hypotd(double u0, double u1)
{
  double y;
  double a;
  double b;
  a = std::abs(u0);
  b = std::abs(u1);
  if (a < b) {
    a /= b;
    y = b * std::sqrt(a * a + 1.0);
  } else if (a > b) {
    b /= a;
    y = a * std::sqrt(b * b + 1.0);
  } else {
    y = a * 1.4142135623730951;
  }

  return y;
}

//
// Arguments    : double u0
//                double u1
// Return Type  : double
//
static double rt_remd(double u0, double u1)
{
  double y;
  double b_u1;
  double q;
  if (u1 < 0.0) {
    b_u1 = std::ceil(u1);
  } else {
    b_u1 = std::floor(u1);
  }

  if ((u1 != 0.0) && (u1 != b_u1)) {
    q = std::abs(u0 / u1);
    if (std::abs(q - std::floor(q + 0.5)) <= DBL_EPSILON * q) {
      y = 0.0;
    } else {
      y = std::fmod(u0, u1);
    }
  } else {
    y = std::fmod(u0, u1);
  }

  return y;
}

//
// Arguments    : double u
// Return Type  : double
//
static double rt_roundd(double u)
{
  double y;
  if (std::abs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = std::floor(u + 0.5);
    } else if (u > -0.5) {
      y = 0.0;
    } else {
      y = std::ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

//
// Arguments    : float u
// Return Type  : float
//
static float rt_roundf(float u)
{
  float y;
  if (std::abs(u) < 8.388608E+6F) {
    if (u >= 0.5F) {
      y = std::floor(u + 0.5F);
    } else if (u > -0.5F) {
      y = 0.0F;
    } else {
      y = std::ceil(u - 0.5F);
    }
  } else {
    y = u;
  }

  return y;
}

//
// Arguments    : const double obj_cutDim_data[]
//                const double obj_cutVal_data[]
//                const double obj_lowerBounds_data[]
//                const int obj_lowerBounds_size[2]
//                const double obj_upperBounds_data[]
//                const int obj_upperBounds_size[2]
//                const double obj_leftChild_data[]
//                const double obj_rightChild_data[]
//                const boolean_T obj_leafNode_data[]
//                const unsigned int obj_idxAll_data[]
//                const double obj_idxDim_data[]
//                const float X_data[]
//                const float queryPt[2]
//                int numNN
//                float pq_D_data[]
//                int pq_D_size[1]
//                unsigned int pq_I_data[]
//                int pq_I_size[1]
// Return Type  : void
//
static void search_kdtree(const double obj_cutDim_data[], const double
  obj_cutVal_data[], const double obj_lowerBounds_data[], const int
  obj_lowerBounds_size[2], const double obj_upperBounds_data[], const int
  obj_upperBounds_size[2], const double obj_leftChild_data[], const double
  obj_rightChild_data[], const boolean_T obj_leafNode_data[], const unsigned int
  obj_idxAll_data[], const double obj_idxDim_data[], const float X_data[], const
  float queryPt[2], int numNN, float pq_D_data[], int pq_D_size[1], unsigned int
  pq_I_data[], int pq_I_size[1])
{
  double start_node;
  int node_idx_this_size[1];
  boolean_T ballIsWithinBounds;
  int obj_lowerBounds_data_tmp;
  int b_obj_lowerBounds_data_tmp;
  double b_obj_lowerBounds_data[2];
  int nodeStack_size_idx_0;
  double b_obj_upperBounds_data[2];
  int exitg1;
  double currentNode;
  int tmp_size[1];
  boolean_T guard1 = false;
  PerceptionSmartLoaderTLS *PerceptionSmartLoaderTLSThread;
  PerceptionSmartLoaderTLSThread = emlrtGetThreadStackData();
  start_node = get_starting_node(queryPt, obj_cutDim_data, obj_cutVal_data,
    obj_leafNode_data, obj_leftChild_data, obj_rightChild_data);
  getNodeFromArray(obj_idxAll_data, obj_idxDim_data, start_node,
                   PerceptionSmartLoaderTLSThread->u4.f5.node_idx_this_data,
                   node_idx_this_size);
  PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0] = 0;
  PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.size[0] = 0;
  search_node(X_data, queryPt,
              PerceptionSmartLoaderTLSThread->u4.f5.node_idx_this_data,
              node_idx_this_size, numNN,
              &PerceptionSmartLoaderTLSThread->u4.f5.r1);
  pq_D_size[0] = PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0];
  if (0 <= PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0] - 1) {
    memcpy(&pq_D_data[0], &PerceptionSmartLoaderTLSThread->u4.f5.r1.D.data[0],
           (unsigned int)(PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0] *
                          (int)sizeof(float)));
  }

  pq_I_size[0] = PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.size[0];
  if (0 <= PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.size[0] - 1) {
    memcpy(&pq_I_data[0], &PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.data[0],
           (unsigned int)(PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.size[0] *
                          (int)sizeof(unsigned int)));
  }

  if (PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0] != 0) {
    obj_lowerBounds_data_tmp = (int)start_node - 1;
    b_obj_lowerBounds_data_tmp = obj_lowerBounds_size[1] *
      obj_lowerBounds_data_tmp;
    b_obj_lowerBounds_data[0] = obj_lowerBounds_data[b_obj_lowerBounds_data_tmp];
    obj_lowerBounds_data_tmp *= obj_upperBounds_size[1];
    b_obj_upperBounds_data[0] = obj_upperBounds_data[obj_lowerBounds_data_tmp];
    b_obj_lowerBounds_data[1] = obj_lowerBounds_data[1 +
      b_obj_lowerBounds_data_tmp];
    b_obj_upperBounds_data[1] = obj_upperBounds_data[1 +
      obj_lowerBounds_data_tmp];
    ballIsWithinBounds = ball_within_bounds(queryPt, b_obj_lowerBounds_data,
      b_obj_upperBounds_data, PerceptionSmartLoaderTLSThread->u4.f5.r1.D.data[0]);
  } else {
    ballIsWithinBounds = false;
  }

  if ((PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0] == numNN) &&
      ballIsWithinBounds) {
  } else {
    nodeStack_size_idx_0 = 1;
    PerceptionSmartLoaderTLSThread->u4.f5.nodeStack_data[0] = 1.0;
    do {
      exitg1 = 0;
      if (nodeStack_size_idx_0 != 0) {
        currentNode = PerceptionSmartLoaderTLSThread->u4.f5.nodeStack_data[0];
        tmp_size[0] = nodeStack_size_idx_0;
        if (0 <= nodeStack_size_idx_0 - 1) {
          memcpy(&PerceptionSmartLoaderTLSThread->u4.f5.tmp_data[0],
                 &PerceptionSmartLoaderTLSThread->u4.f5.nodeStack_data[0],
                 (unsigned int)(nodeStack_size_idx_0 * (int)sizeof(double)));
        }

        b_nullAssignment(PerceptionSmartLoaderTLSThread->u4.f5.tmp_data,
                         tmp_size);
        nodeStack_size_idx_0 = tmp_size[0];
        if (0 <= tmp_size[0] - 1) {
          memcpy(&PerceptionSmartLoaderTLSThread->u4.f5.nodeStack_data[0],
                 &PerceptionSmartLoaderTLSThread->u4.f5.tmp_data[0], (unsigned
                  int)(tmp_size[0] * (int)sizeof(double)));
        }

        guard1 = false;
        if (pq_D_size[0] < numNN) {
          guard1 = true;
        } else {
          obj_lowerBounds_data_tmp = (int)currentNode - 1;
          b_obj_lowerBounds_data_tmp = obj_lowerBounds_size[1] *
            obj_lowerBounds_data_tmp;
          b_obj_lowerBounds_data[0] =
            obj_lowerBounds_data[b_obj_lowerBounds_data_tmp];
          obj_lowerBounds_data_tmp *= obj_upperBounds_size[1];
          b_obj_upperBounds_data[0] =
            obj_upperBounds_data[obj_lowerBounds_data_tmp];
          b_obj_lowerBounds_data[1] = obj_lowerBounds_data[1 +
            b_obj_lowerBounds_data_tmp];
          b_obj_upperBounds_data[1] = obj_upperBounds_data[1 +
            obj_lowerBounds_data_tmp];
          if (bounds_overlap_ball(queryPt, b_obj_lowerBounds_data,
                                  b_obj_upperBounds_data, pq_D_data[0])) {
            guard1 = true;
          }
        }

        if (guard1) {
          if (!obj_leafNode_data[(int)currentNode - 1]) {
            if (queryPt[(int)obj_cutDim_data[(int)currentNode - 1] - 1] <=
                obj_cutVal_data[(int)currentNode - 1]) {
              PerceptionSmartLoaderTLSThread->u4.f5.obj_rightChild_data[0] =
                obj_leftChild_data[(int)currentNode - 1];
              PerceptionSmartLoaderTLSThread->u4.f5.obj_rightChild_data[1] =
                obj_rightChild_data[(int)currentNode - 1];
              if (0 <= tmp_size[0] - 1) {
                memcpy
                  (&PerceptionSmartLoaderTLSThread->u4.f5.obj_rightChild_data[2],
                   &PerceptionSmartLoaderTLSThread->u4.f5.nodeStack_data[0],
                   (unsigned int)(tmp_size[0] * (int)sizeof(double)));
              }

              nodeStack_size_idx_0 = 2 + tmp_size[0];
              obj_lowerBounds_data_tmp = 2 + tmp_size[0];
              if (0 <= obj_lowerBounds_data_tmp - 1) {
                memcpy(&PerceptionSmartLoaderTLSThread->u4.f5.nodeStack_data[0],
                       &PerceptionSmartLoaderTLSThread->u4.f5.obj_rightChild_data
                       [0], (unsigned int)(obj_lowerBounds_data_tmp * (int)
                        sizeof(double)));
              }
            } else {
              PerceptionSmartLoaderTLSThread->u4.f5.obj_rightChild_data[0] =
                obj_rightChild_data[(int)currentNode - 1];
              PerceptionSmartLoaderTLSThread->u4.f5.obj_rightChild_data[1] =
                obj_leftChild_data[(int)currentNode - 1];
              if (0 <= tmp_size[0] - 1) {
                memcpy
                  (&PerceptionSmartLoaderTLSThread->u4.f5.obj_rightChild_data[2],
                   &PerceptionSmartLoaderTLSThread->u4.f5.nodeStack_data[0],
                   (unsigned int)(tmp_size[0] * (int)sizeof(double)));
              }

              nodeStack_size_idx_0 = 2 + tmp_size[0];
              obj_lowerBounds_data_tmp = 2 + tmp_size[0];
              if (0 <= obj_lowerBounds_data_tmp - 1) {
                memcpy(&PerceptionSmartLoaderTLSThread->u4.f5.nodeStack_data[0],
                       &PerceptionSmartLoaderTLSThread->u4.f5.obj_rightChild_data
                       [0], (unsigned int)(obj_lowerBounds_data_tmp * (int)
                        sizeof(double)));
              }
            }
          } else {
            if (currentNode != start_node) {
              getNodeFromArray(obj_idxAll_data, obj_idxDim_data, currentNode,
                               PerceptionSmartLoaderTLSThread->u4.f5.node_idx_this_data,
                               node_idx_this_size);
              PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0] = pq_D_size[0];
              if (0 <= pq_D_size[0] - 1) {
                memcpy(&PerceptionSmartLoaderTLSThread->u4.f5.r1.D.data[0],
                       &pq_D_data[0], (unsigned int)(pq_D_size[0] * (int)sizeof
                        (float)));
              }

              PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.size[0] = pq_I_size[0];
              if (0 <= pq_I_size[0] - 1) {
                memcpy(&PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.data[0],
                       &pq_I_data[0], (unsigned int)(pq_I_size[0] * (int)sizeof
                        (unsigned int)));
              }

              search_node(X_data, queryPt,
                          PerceptionSmartLoaderTLSThread->u4.f5.node_idx_this_data,
                          node_idx_this_size, numNN,
                          &PerceptionSmartLoaderTLSThread->u4.f5.r1);
              pq_D_size[0] = PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0];
              if (0 <= PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0] - 1) {
                memcpy(&pq_D_data[0],
                       &PerceptionSmartLoaderTLSThread->u4.f5.r1.D.data[0],
                       (unsigned int)
                       (PerceptionSmartLoaderTLSThread->u4.f5.r1.D.size[0] *
                        (int)sizeof(float)));
              }

              pq_I_size[0] = PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.size[0];
              if (0 <= PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.size[0] - 1)
              {
                memcpy(&pq_I_data[0],
                       &PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.data[0],
                       (unsigned int)
                       (PerceptionSmartLoaderTLSThread->u4.f5.r1.b_I.size[0] *
                        (int)sizeof(unsigned int)));
              }
            }
          }
        }
      } else {
        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }
}

//
// Arguments    : const float X_data[]
//                const float queryPt[2]
//                const unsigned int node_idx_start_data[]
//                const int node_idx_start_size[1]
//                int numNN
//                d_struct_T *pq
// Return Type  : void
//
static void search_node(const float X_data[], const float queryPt[2], const
  unsigned int node_idx_start_data[], const int node_idx_start_size[1], int
  numNN, d_struct_T *pq)
{
  int X_size[2];
  int loop_ub;
  int i43;
  int diffAllDim_size[2];
  int X_data_tmp;
  int b_diffAllDim_size[2];
  int b_X_data_tmp;
  int aDistOut_size[1];
  int distInP_size[1];
  int tmp_size[1];
  PerceptionSmartLoaderTLS *PerceptionSmartLoaderTLSThread;
  PerceptionSmartLoaderTLSThread = emlrtGetThreadStackData();
  X_size[1] = 2;
  X_size[0] = node_idx_start_size[0];
  loop_ub = node_idx_start_size[0];
  for (i43 = 0; i43 < loop_ub; i43++) {
    X_data_tmp = ((int)node_idx_start_data[i43] - 1) << 1;
    b_X_data_tmp = i43 << 1;
    PerceptionSmartLoaderTLSThread->u3.f4.X_data[b_X_data_tmp] =
      X_data[X_data_tmp];
    PerceptionSmartLoaderTLSThread->u3.f4.X_data[1 + b_X_data_tmp] = X_data[1 +
      X_data_tmp];
  }

  bsxfun(PerceptionSmartLoaderTLSThread->u3.f4.X_data, X_size, queryPt,
         PerceptionSmartLoaderTLSThread->u3.f4.diffAllDim_data, diffAllDim_size);
  b_diffAllDim_size[1] = 2;
  b_diffAllDim_size[0] = diffAllDim_size[0];
  loop_ub = diffAllDim_size[1] * diffAllDim_size[0];
  for (i43 = 0; i43 < loop_ub; i43++) {
    PerceptionSmartLoaderTLSThread->u3.f4.X_data[i43] =
      PerceptionSmartLoaderTLSThread->u3.f4.diffAllDim_data[i43] *
      PerceptionSmartLoaderTLSThread->u3.f4.diffAllDim_data[i43];
  }

  b_sum(PerceptionSmartLoaderTLSThread->u3.f4.X_data, b_diffAllDim_size,
        PerceptionSmartLoaderTLSThread->u3.f4.aDistOut_data, aDistOut_size);
  distInP_size[0] = aDistOut_size[0];
  if (0 <= aDistOut_size[0] - 1) {
    memcpy(&PerceptionSmartLoaderTLSThread->u3.f4.distInP_data[0],
           &PerceptionSmartLoaderTLSThread->u3.f4.aDistOut_data[0], (unsigned
            int)(aDistOut_size[0] * (int)sizeof(float)));
  }

  sort(PerceptionSmartLoaderTLSThread->u3.f4.distInP_data, distInP_size,
       PerceptionSmartLoaderTLSThread->u3.f4.iidx_data, aDistOut_size);
  if (pq->D.size[0] == 0) {
    if (distInP_size[0] <= numNN) {
      pq->D.size[0] = distInP_size[0];
      if (0 <= distInP_size[0] - 1) {
        memcpy(&pq->D.data[0],
               &PerceptionSmartLoaderTLSThread->u3.f4.distInP_data[0], (unsigned
                int)(distInP_size[0] * (int)sizeof(float)));
      }

      pq->b_I.size[0] = aDistOut_size[0];
      loop_ub = aDistOut_size[0];
      for (i43 = 0; i43 < loop_ub; i43++) {
        pq->b_I.data[i43] = node_idx_start_data
          [PerceptionSmartLoaderTLSThread->u3.f4.iidx_data[i43] - 1];
      }
    } else {
      for (i43 = 0; i43 < numNN; i43++) {
        PerceptionSmartLoaderTLSThread->u3.f4.c_tmp_data[i43] = i43;
      }

      pq->D.size[0] = numNN;
      for (i43 = 0; i43 < numNN; i43++) {
        pq->D.data[i43] = PerceptionSmartLoaderTLSThread->
          u3.f4.distInP_data[PerceptionSmartLoaderTLSThread->
          u3.f4.c_tmp_data[i43]];
      }

      for (i43 = 0; i43 < numNN; i43++) {
        PerceptionSmartLoaderTLSThread->u3.f4.c_tmp_data[i43] = i43;
      }

      pq->b_I.size[0] = numNN;
      for (i43 = 0; i43 < numNN; i43++) {
        pq->b_I.data[i43] = node_idx_start_data
          [PerceptionSmartLoaderTLSThread->
          u3.f4.iidx_data[PerceptionSmartLoaderTLSThread->u3.f4.c_tmp_data[i43]]
          - 1];
      }
    }
  } else {
    loop_ub = aDistOut_size[0];
    for (i43 = 0; i43 < loop_ub; i43++) {
      PerceptionSmartLoaderTLSThread->u3.f4.node_idx_start_data[i43] =
        node_idx_start_data[PerceptionSmartLoaderTLSThread->u3.f4.iidx_data[i43]
        - 1];
    }

    mergeSort(pq->D.data, pq->D.size,
              PerceptionSmartLoaderTLSThread->u3.f4.distInP_data, distInP_size,
              pq->b_I.data,
              PerceptionSmartLoaderTLSThread->u3.f4.node_idx_start_data, numNN,
              PerceptionSmartLoaderTLSThread->u3.f4.tmp_data, aDistOut_size,
              PerceptionSmartLoaderTLSThread->u3.f4.b_tmp_data, tmp_size);
    pq->D.size[0] = aDistOut_size[0];
    if (0 <= aDistOut_size[0] - 1) {
      memcpy(&pq->D.data[0], &PerceptionSmartLoaderTLSThread->u3.f4.tmp_data[0],
             (unsigned int)(aDistOut_size[0] * (int)sizeof(float)));
    }

    pq->b_I.size[0] = tmp_size[0];
    if (0 <= tmp_size[0] - 1) {
      memcpy(&pq->b_I.data[0], &PerceptionSmartLoaderTLSThread->
             u3.f4.b_tmp_data[0], (unsigned int)(tmp_size[0] * (int)sizeof
              (unsigned int)));
    }
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                int n
//                int idx_data[]
//                int idx_size[1]
// Return Type  : void
//
static void simpleRandperm(PerceptionSmartLoaderStackData *SD, int n, int
  idx_data[], int idx_size[1])
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

//
// Arguments    : float x_data[]
//                int x_size[1]
//                int idx_data[]
//                int idx_size[1]
// Return Type  : void
//
static void sort(float x_data[], int x_size[1], int idx_data[], int idx_size[1])
{
  int dim;
  int i39;
  int vlen;
  int vwork_size[1];
  int vstride;
  int k;
  int iidx_size[1];
  PerceptionSmartLoaderTLS *PerceptionSmartLoaderTLSThread;
  PerceptionSmartLoaderTLSThread = emlrtGetThreadStackData();
  dim = nonSingletonDim(x_size);
  if (dim <= 1) {
    i39 = x_size[0];
  } else {
    i39 = 1;
  }

  vlen = i39 - 1;
  vwork_size[0] = i39;
  idx_size[0] = x_size[0];
  vstride = 1;
  for (k = 0; k <= dim - 2; k++) {
    vstride *= x_size[0];
  }

  for (dim = 0; dim < vstride; dim++) {
    for (k = 0; k <= vlen; k++) {
      PerceptionSmartLoaderTLSThread->u2.f3.vwork_data[k] = x_data[dim + k *
        vstride];
    }

    c_sortIdx(PerceptionSmartLoaderTLSThread->u2.f3.vwork_data, vwork_size,
              PerceptionSmartLoaderTLSThread->u2.f3.iidx_data, iidx_size);
    for (k = 0; k <= vlen; k++) {
      i39 = dim + k * vstride;
      x_data[i39] = PerceptionSmartLoaderTLSThread->u2.f3.vwork_data[k];
      idx_data[i39] = PerceptionSmartLoaderTLSThread->u2.f3.iidx_data[k];
    }
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const float x_data[]
//                const int x_size[2]
//                const int col_data[]
//                const int col_size[2]
//                int idx_data[]
//                int idx_size[1]
// Return Type  : void
//
static void sortIdx(PerceptionSmartLoaderStackData *SD, const float x_data[],
                    const int x_size[2], const int col_data[], const int
                    col_size[2], int idx_data[], int idx_size[1])
{
  int n;
  int k;
  n = x_size[0];
  idx_size[0] = x_size[0];
  if (0 <= x_size[0] - 1) {
    memset(&idx_data[0], 0, (unsigned int)(x_size[0] * (int)sizeof(int)));
  }

  if ((x_size[0] == 0) || (x_size[1] == 0)) {
    for (k = 0; k < n; k++) {
      idx_data[k] = k + 1;
    }
  } else {
    b_mergesort(SD, idx_data, x_data, x_size, col_data, col_size, x_size[0]);
  }
}

//
// Arguments    : const float v_data[]
//                const int v_size[2]
//                const int dir_data[]
//                const int dir_size[2]
//                int idx1
//                int idx2
// Return Type  : boolean_T
//
static boolean_T sortLE(const float v_data[], const int v_size[2], const int
  dir_data[], const int dir_size[2], int idx1, int idx2)
{
  boolean_T p;
  int k;
  boolean_T exitg1;
  float v1;
  float v2;
  p = true;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k <= dir_size[1] - 1)) {
    v1 = v_data[(dir_data[k] + v_size[1] * (idx1 - 1)) - 1];
    v2 = v_data[(dir_data[k] + v_size[1] * (idx2 - 1)) - 1];
    if (v1 != v2) {
      p = (v1 <= v2);
      exitg1 = true;
    } else {
      k++;
    }
  }

  return p;
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                float y_data[]
//                int y_size[2]
//                double ndx_data[]
//                int ndx_size[1]
// Return Type  : void
//
static void sortrows(PerceptionSmartLoaderStackData *SD, float y_data[], int
                     y_size[2], double ndx_data[], int ndx_size[1])
{
  int idx_size[1];
  int loop_ub;
  int i36;
  b_sortIdx(SD, y_data, y_size, SD->u2.f14.idx_data, idx_size);
  b_apply_row_permutation(SD, y_data, y_size, SD->u2.f14.idx_data);
  ndx_size[0] = idx_size[0];
  loop_ub = idx_size[0];
  for (i36 = 0; i36 < loop_ub; i36++) {
    ndx_data[i36] = SD->u2.f14.idx_data[i36];
  }
}

//
// Arguments    : const emxArray_real32_T *Y
//                emxArray_real32_T *Z
// Return Type  : void
//
static void squareform(const emxArray_real32_T *Y, emxArray_real32_T *Z)
{
  int m;
  int i19;
  int loop_ub;
  unsigned int k;
  int i;
  int b_i;
  int i20;
  m = (int)std::ceil(std::sqrt(2.0 * (double)Y->size[1]));
  i19 = Z->size[0] * Z->size[1];
  Z->size[1] = m;
  Z->size[0] = m;
  emxEnsureCapacity_real32_T(Z, i19);
  loop_ub = m * m;
  for (i19 = 0; i19 < loop_ub; i19++) {
    Z->data[i19] = 0.0F;
  }

  if (m > 1) {
    k = 1U;
    for (loop_ub = 0; loop_ub <= m - 2; loop_ub++) {
      i19 = m - loop_ub;
      for (i = 0; i <= i19 - 2; i++) {
        b_i = (loop_ub + i) + 1;
        i20 = (int)k - 1;
        Z->data[loop_ub + Z->size[1] * b_i] = Y->data[i20];
        Z->data[b_i + Z->size[1] * loop_ub] = Y->data[i20];
        k++;
      }
    }
  }
}

//
// Arguments    : const double x_data[]
//                const int x_size[1]
// Return Type  : double
//
static double sum(const double x_data[], const int x_size[1])
{
  double y;
  int vlen;
  int k;
  vlen = x_size[0];
  y = x_data[0];
  for (k = 2; k <= vlen; k++) {
    if (vlen >= 2) {
      y += x_data[k - 1];
    }
  }

  return y;
}

//
// Arguments    : const float v_data[]
//                int ia
//                int ib
// Return Type  : int
//
static int thirdOfFive(const float v_data[], int ia, int ib)
{
  int im;
  float v4;
  float v5_tmp;
  int b_j1;
  int j2;
  int j3;
  int j4;
  int j5;
  float v5;
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

//
// Arguments    : const float x[6]
//                float y[2]
// Return Type  : void
//
static void vecnorm(const float x[6], float y[2])
{
  float scale;
  float absxk;
  float t;
  float yv;
  scale = 1.29246971E-26F;
  absxk = std::abs(x[0]);
  if (absxk > 1.29246971E-26F) {
    yv = 1.0F;
    scale = absxk;
  } else {
    t = absxk / 1.29246971E-26F;
    yv = t * t;
  }

  absxk = std::abs(x[2]);
  if (absxk > scale) {
    t = scale / absxk;
    yv = 1.0F + yv * t * t;
    scale = absxk;
  } else {
    t = absxk / scale;
    yv += t * t;
  }

  absxk = std::abs(x[4]);
  if (absxk > scale) {
    t = scale / absxk;
    yv = 1.0F + yv * t * t;
    scale = absxk;
  } else {
    t = absxk / scale;
    yv += t * t;
  }

  y[0] = scale * std::sqrt(yv);
  scale = 1.29246971E-26F;
  absxk = std::abs(x[1]);
  if (absxk > 1.29246971E-26F) {
    yv = 1.0F;
    scale = absxk;
  } else {
    t = absxk / 1.29246971E-26F;
    yv = t * t;
  }

  absxk = std::abs(x[3]);
  if (absxk > scale) {
    t = scale / absxk;
    yv = 1.0F + yv * t * t;
    scale = absxk;
  } else {
    t = absxk / scale;
    yv += t * t;
  }

  absxk = std::abs(x[5]);
  if (absxk > scale) {
    t = scale / absxk;
    yv = 1.0F + yv * t * t;
    scale = absxk;
  } else {
    t = absxk / scale;
    yv += t * t;
  }

  y[1] = scale * std::sqrt(yv);
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
//                float v_data[]
//                int v_size[1]
//                int n
// Return Type  : float
//
static float vmedian(PerceptionSmartLoaderStackData *SD, float v_data[], int
                     v_size[1], int n)
{
  float m;
  int midm1;
  int loop_ub;
  int ilast;
  int unusedU5;
  int b_v_data;
  float b;
  if (n <= 4) {
    if (n == 0) {
      m = 0.0F;
    } else if (n == 1) {
      m = v_data[0];
    } else if (n == 2) {
      m = v_data[0] + (v_data[1] - v_data[0]) / 2.0F;
    } else if (n == 3) {
      if (v_data[0] < v_data[1]) {
        if (v_data[1] < v_data[2]) {
          b_v_data = 1;
        } else if (v_data[0] < v_data[2]) {
          b_v_data = 2;
        } else {
          b_v_data = 0;
        }
      } else if (v_data[0] < v_data[2]) {
        b_v_data = 0;
      } else if (v_data[1] < v_data[2]) {
        b_v_data = 2;
      } else {
        b_v_data = 1;
      }

      m = v_data[b_v_data];
    } else {
      if (v_data[0] < v_data[1]) {
        if (v_data[1] < v_data[2]) {
          loop_ub = 0;
          unusedU5 = 1;
          ilast = 2;
        } else if (v_data[0] < v_data[2]) {
          loop_ub = 0;
          unusedU5 = 2;
          ilast = 1;
        } else {
          loop_ub = 2;
          unusedU5 = 0;
          ilast = 1;
        }
      } else if (v_data[0] < v_data[2]) {
        loop_ub = 1;
        unusedU5 = 0;
        ilast = 2;
      } else if (v_data[1] < v_data[2]) {
        loop_ub = 1;
        unusedU5 = 2;
        ilast = 0;
      } else {
        loop_ub = 2;
        unusedU5 = 1;
        ilast = 0;
      }

      if (v_data[loop_ub] < v_data[3]) {
        if (v_data[3] < v_data[ilast]) {
          m = v_data[unusedU5] + (v_data[3] - v_data[unusedU5]) / 2.0F;
        } else {
          m = v_data[unusedU5] + (v_data[ilast] - v_data[unusedU5]) / 2.0F;
        }
      } else {
        m = v_data[loop_ub] + (v_data[unusedU5] - v_data[loop_ub]) / 2.0F;
      }
    }
  } else {
    midm1 = n >> 1;
    if ((n & 1) == 0) {
      quickselect(v_data, midm1 + 1, n, &m, &loop_ub, &ilast);
      if (midm1 < loop_ub) {
        loop_ub = v_size[0];
        if (0 <= loop_ub - 1) {
          memcpy(&SD->u1.f4.unusedU3_data[0], &v_data[0], (unsigned int)(loop_ub
                  * (int)sizeof(float)));
        }

        quickselect(SD->u1.f4.unusedU3_data, midm1, ilast - 1, &b, &loop_ub,
                    &unusedU5);
        m += (b - m) / 2.0F;
      }
    } else {
      loop_ub = v_size[0];
      if (0 <= loop_ub - 1) {
        memcpy(&SD->u1.f4.unusedU3_data[0], &v_data[0], (unsigned int)(loop_ub *
                (int)sizeof(float)));
      }

      quickselect(SD->u1.f4.unusedU3_data, midm1 + 1, n, &m, &loop_ub, &unusedU5);
    }
  }

  return m;
}

//
// Arguments    : double *a
//                double *b
//                double *c
//                double *d
//                double *rt1r
//                double *rt1i
//                double *rt2r
//                double *rt2i
//                double *cs
//                double *sn
// Return Type  : void
//
static void xdlanv2(double *a, double *b, double *c, double *d, double *rt1r,
                    double *rt1i, double *rt2r, double *rt2i, double *cs, double
                    *sn)
{
  double tau;
  double p;
  double bcmax;
  double bcmis;
  double scale;
  int b_b;
  int b_c;
  double z;
  double b_p;
  int b_bcmis;
  if (*c == 0.0) {
    *cs = 1.0;
    *sn = 0.0;
  } else if (*b == 0.0) {
    *cs = 0.0;
    *sn = 1.0;
    bcmax = *d;
    *d = *a;
    *a = bcmax;
    *b = -*c;
    *c = 0.0;
  } else {
    tau = *a - *d;
    if ((tau == 0.0) && ((*b < 0.0) != (*c < 0.0))) {
      *cs = 1.0;
      *sn = 0.0;
    } else {
      p = 0.5 * tau;
      bcmis = std::abs(*b);
      scale = std::abs(*c);
      if (bcmis > scale) {
        bcmax = bcmis;
      } else {
        bcmax = scale;
      }

      if (bcmis < scale) {
        scale = bcmis;
      }

      if (*b >= 0.0) {
        b_b = 1;
      } else {
        b_b = -1;
      }

      if (*c >= 0.0) {
        b_c = 1;
      } else {
        b_c = -1;
      }

      bcmis = scale * (double)b_b * (double)b_c;
      scale = std::abs(p);
      if (scale <= bcmax) {
        scale = bcmax;
      }

      z = p / scale * p + bcmax / scale * bcmis;
      if (z >= 8.8817841970012523E-16) {
        *a = std::sqrt(scale) * std::sqrt(z);
        if (p >= 0.0) {
          b_p = *a;
        } else {
          b_p = -*a;
        }

        z = p + b_p;
        *a = *d + z;
        *d -= bcmax / z * bcmis;
        tau = rt_hypotd(*c, z);
        *cs = z / tau;
        *sn = *c / tau;
        *b -= *c;
        *c = 0.0;
      } else {
        bcmis = *b + *c;
        tau = rt_hypotd(bcmis, tau);
        *cs = std::sqrt(0.5 * (1.0 + std::abs(bcmis) / tau));
        if (bcmis >= 0.0) {
          b_bcmis = 1;
        } else {
          b_bcmis = -1;
        }

        *sn = -(p / (tau * *cs)) * (double)b_bcmis;
        bcmax = *a * *cs + *b * *sn;
        scale = -*a * *sn + *b * *cs;
        z = *c * *cs + *d * *sn;
        bcmis = -*c * *sn + *d * *cs;
        *b = scale * *cs + bcmis * *sn;
        *c = -bcmax * *sn + z * *cs;
        bcmax = 0.5 * ((bcmax * *cs + z * *sn) + (-scale * *sn + bcmis * *cs));
        *a = bcmax;
        *d = bcmax;
        if (*c != 0.0) {
          if (*b != 0.0) {
            if ((*b < 0.0) == (*c < 0.0)) {
              bcmis = std::sqrt(std::abs(*b));
              z = std::sqrt(std::abs(*c));
              *a = bcmis * z;
              if (*c >= 0.0) {
                p = *a;
              } else {
                p = -*a;
              }

              tau = 1.0 / std::sqrt(std::abs(*b + *c));
              *a = bcmax + p;
              *d = bcmax - p;
              *b -= *c;
              *c = 0.0;
              scale = bcmis * tau;
              bcmis = z * tau;
              bcmax = *cs * scale - *sn * bcmis;
              *sn = *cs * bcmis + *sn * scale;
              *cs = bcmax;
            }
          } else {
            *b = -*c;
            *c = 0.0;
            bcmax = *cs;
            *cs = -*sn;
            *sn = bcmax;
          }
        }
      }
    }
  }

  *rt1r = *a;
  *rt2r = *d;
  if (*c == 0.0) {
    *rt1i = 0.0;
    *rt2i = 0.0;
  } else {
    *rt1i = std::sqrt(std::abs(*b)) * std::sqrt(std::abs(*c));
    *rt2i = -*rt1i;
  }
}

//
// Arguments    : double a[4]
// Return Type  : double
//
static double xgehrd(double a[4])
{
  double tau;
  double alpha1;
  alpha1 = a[1];
  tau = 0.0;
  a[1] = 1.0;
  a[1] = alpha1;
  return tau;
}

//
// Arguments    : creal_T A[4]
//                int *ilo
//                int *ihi
//                int rscale[2]
// Return Type  : void
//
static void xzggbal(creal_T A[4], int *ilo, int *ihi, int rscale[2])
{
  int i;
  int j;
  boolean_T found;
  int ii;
  boolean_T exitg1;
  int nzcount;
  double atmp_re;
  double atmp_im;
  int jj;
  boolean_T exitg2;
  rscale[0] = 1;
  rscale[1] = 1;
  *ilo = 1;
  *ihi = 2;
  i = 0;
  j = 0;
  found = false;
  ii = 2;
  exitg1 = false;
  while ((!exitg1) && (ii > 0)) {
    nzcount = 0;
    i = ii;
    j = 2;
    jj = 0;
    exitg2 = false;
    while ((!exitg2) && (jj < 2)) {
      if ((A[(ii + (jj << 1)) - 1].re != 0.0) || (A[(ii + (jj << 1)) - 1].im !=
           0.0) || (ii == jj + 1)) {
        if (nzcount == 0) {
          j = jj + 1;
          nzcount = 1;
          jj++;
        } else {
          nzcount = 2;
          exitg2 = true;
        }
      } else {
        jj++;
      }
    }

    if (nzcount < 2) {
      found = true;
      exitg1 = true;
    } else {
      ii--;
    }
  }

  if (!found) {
    i = 0;
    j = 0;
    found = false;
    jj = 1;
    exitg1 = false;
    while ((!exitg1) && (jj < 3)) {
      nzcount = 0;
      i = 2;
      j = jj;
      ii = 1;
      exitg2 = false;
      while ((!exitg2) && (ii < 3)) {
        if ((A[(ii + ((jj - 1) << 1)) - 1].re != 0.0) || (A[(ii + ((jj - 1) << 1))
             - 1].im != 0.0) || (ii == jj)) {
          if (nzcount == 0) {
            i = ii;
            nzcount = 1;
            ii++;
          } else {
            nzcount = 2;
            exitg2 = true;
          }
        } else {
          ii++;
        }
      }

      if (nzcount < 2) {
        found = true;
        exitg1 = true;
      } else {
        jj++;
      }
    }

    if (found) {
      if (i != 1) {
        atmp_re = A[i - 1].re;
        atmp_im = A[i - 1].im;
        A[i - 1] = A[0];
        A[0].re = atmp_re;
        A[0].im = atmp_im;
        atmp_re = A[i + 1].re;
        atmp_im = A[i + 1].im;
        A[i + 1] = A[2];
        A[2].re = atmp_re;
        A[2].im = atmp_im;
      }

      if (j != 1) {
        i = (j - 1) << 1;
        atmp_re = A[i].re;
        atmp_im = A[(j - 1) << 1].im;
        A[i] = A[0];
        A[0].re = atmp_re;
        A[0].im = atmp_im;
        i++;
        atmp_re = A[i].re;
        atmp_im = A[1 + ((j - 1) << 1)].im;
        A[i] = A[1];
        A[1].re = atmp_re;
        A[1].im = atmp_im;
      }

      rscale[0] = j;
      *ilo = 2;
      rscale[1] = 2;
    }
  } else {
    if (i != 2) {
      atmp_re = A[i - 1].re;
      atmp_im = A[i - 1].im;
      A[i - 1] = A[1];
      A[1].re = atmp_re;
      A[1].im = atmp_im;
      atmp_re = A[i + 1].re;
      atmp_im = A[i + 1].im;
      A[i + 1] = A[3];
      A[3].re = atmp_re;
      A[3].im = atmp_im;
    }

    if (j != 2) {
      i = (j - 1) << 1;
      atmp_re = A[i].re;
      atmp_im = A[(j - 1) << 1].im;
      A[i] = A[2];
      A[2].re = atmp_re;
      A[2].im = atmp_im;
      i++;
      atmp_re = A[i].re;
      atmp_im = A[1 + ((j - 1) << 1)].im;
      A[i] = A[3];
      A[3].re = atmp_re;
      A[3].im = atmp_im;
    }

    rscale[1] = j;
    *ihi = 1;
    rscale[0] = 1;
  }
}

//
// Arguments    : creal_T A[4]
//                int *info
//                creal_T alpha1[2]
//                creal_T beta1[2]
//                creal_T V[4]
// Return Type  : void
//
static void xzggev(creal_T A[4], int *info, creal_T alpha1[2], creal_T beta1[2],
                   creal_T V[4])
{
  double anrm;
  double absxk;
  boolean_T ilascl;
  double anrmto;
  int ilo;
  int ihi;
  int rscale[2];
  double ctoc;
  boolean_T notdone;
  double scale;
  double d;
  double y_tmp;
  int jcol;
  int jrow;
  int i;
  int f_re_tmp;
  int rescaledir;
  double f_re;
  double f_im;
  double fs_re;
  double fs_im;
  int b_y_tmp;
  double gs_re;
  double gs_im;
  int count;
  boolean_T guard1 = false;
  double c;
  double b_gs_re;
  anrm = 0.0;
  absxk = rt_hypotd(A[0].re, A[0].im);
  if (absxk > 0.0) {
    anrm = absxk;
  }

  absxk = rt_hypotd(A[1].re, A[1].im);
  if (absxk > anrm) {
    anrm = absxk;
  }

  absxk = rt_hypotd(A[2].re, A[2].im);
  if (absxk > anrm) {
    anrm = absxk;
  }

  absxk = rt_hypotd(A[3].re, A[3].im);
  if (absxk > anrm) {
    anrm = absxk;
  }

  ilascl = false;
  anrmto = anrm;
  if ((anrm > 0.0) && (anrm < 6.7178761075670888E-139)) {
    anrmto = 6.7178761075670888E-139;
    ilascl = true;
  } else {
    if (anrm > 1.4885657073574029E+138) {
      anrmto = 1.4885657073574029E+138;
      ilascl = true;
    }
  }

  if (ilascl) {
    absxk = anrm;
    ctoc = anrmto;
    notdone = true;
    while (notdone) {
      scale = absxk * 2.0041683600089728E-292;
      d = ctoc / 4.9896007738368E+291;
      if ((scale > ctoc) && (ctoc != 0.0)) {
        y_tmp = 2.0041683600089728E-292;
        absxk = scale;
      } else if (d > absxk) {
        y_tmp = 4.9896007738368E+291;
        ctoc = d;
      } else {
        y_tmp = ctoc / absxk;
        notdone = false;
      }

      A[0].re *= y_tmp;
      A[0].im *= y_tmp;
      A[1].re *= y_tmp;
      A[1].im *= y_tmp;
      A[2].re *= y_tmp;
      A[2].im *= y_tmp;
      A[3].re *= y_tmp;
      A[3].im *= y_tmp;
    }
  }

  xzggbal(A, &ilo, &ihi, rscale);
  V[0].re = 1.0;
  V[0].im = 0.0;
  V[1].re = 0.0;
  V[1].im = 0.0;
  V[2].re = 0.0;
  V[2].im = 0.0;
  V[3].re = 1.0;
  V[3].im = 0.0;
  if (ihi >= ilo + 2) {
    for (jcol = ilo - 1; jcol + 1 < ihi - 1; jcol++) {
      for (jrow = ihi - 2; jrow + 2 > jcol + 2; jrow--) {
        f_re_tmp = jrow + (jcol << 1);
        f_re = A[f_re_tmp].re;
        f_im = A[jrow + (jcol << 1)].im;
        d = std::abs(A[jrow + (jcol << 1)].re);
        scale = d;
        y_tmp = std::abs(A[jrow + (jcol << 1)].im);
        if (y_tmp > d) {
          scale = y_tmp;
        }

        b_y_tmp = f_re_tmp + 1;
        ctoc = std::abs(A[b_y_tmp].re);
        absxk = std::abs(A[(jrow + (jcol << 1)) + 1].im);
        if (absxk > ctoc) {
          ctoc = absxk;
        }

        if (ctoc > scale) {
          scale = ctoc;
        }

        fs_re = A[jrow + (jcol << 1)].re;
        fs_im = A[jrow + (jcol << 1)].im;
        gs_re = A[(jrow + (jcol << 1)) + 1].re;
        gs_im = A[(jrow + (jcol << 1)) + 1].im;
        count = -1;
        rescaledir = 0;
        guard1 = false;
        if (scale >= 7.4428285367870146E+137) {
          do {
            count++;
            fs_re *= 1.3435752215134178E-138;
            fs_im *= 1.3435752215134178E-138;
            gs_re *= 1.3435752215134178E-138;
            gs_im *= 1.3435752215134178E-138;
            scale *= 1.3435752215134178E-138;
          } while (!(scale < 7.4428285367870146E+137));

          rescaledir = 1;
          guard1 = true;
        } else if (scale <= 1.3435752215134178E-138) {
          if ((A[(jrow + (jcol << 1)) + 1].re == 0.0) && (A[(jrow + (jcol << 1))
               + 1].im == 0.0)) {
            c = 1.0;
            gs_re = 0.0;
            gs_im = 0.0;
          } else {
            do {
              count++;
              fs_re *= 7.4428285367870146E+137;
              fs_im *= 7.4428285367870146E+137;
              gs_re *= 7.4428285367870146E+137;
              gs_im *= 7.4428285367870146E+137;
              scale *= 7.4428285367870146E+137;
            } while (!(scale > 1.3435752215134178E-138));

            rescaledir = -1;
            guard1 = true;
          }
        } else {
          guard1 = true;
        }

        if (guard1) {
          ctoc = fs_re * fs_re + fs_im * fs_im;
          scale = gs_re * gs_re + gs_im * gs_im;
          absxk = scale;
          if (1.0 > scale) {
            absxk = 1.0;
          }

          if (ctoc <= absxk * 2.0041683600089728E-292) {
            if ((A[jrow + (jcol << 1)].re == 0.0) && (A[jrow + (jcol << 1)].im ==
                 0.0)) {
              c = 0.0;
              f_re = rt_hypotd(A[(jrow + (jcol << 1)) + 1].re, A[(jrow + (jcol <<
                1)) + 1].im);
              f_im = 0.0;
              d = rt_hypotd(gs_re, gs_im);
              gs_re /= d;
              gs_im = -gs_im / d;
            } else {
              scale = std::sqrt(scale);
              c = rt_hypotd(fs_re, fs_im) / scale;
              if (y_tmp > d) {
                d = y_tmp;
              }

              if (d > 1.0) {
                d = rt_hypotd(A[jrow + (jcol << 1)].re, A[jrow + (jcol << 1)].im);
                fs_re = A[jrow + (jcol << 1)].re / d;
                fs_im = A[jrow + (jcol << 1)].im / d;
              } else {
                absxk = 7.4428285367870146E+137 * A[jrow + (jcol << 1)].re;
                ctoc = 7.4428285367870146E+137 * A[jrow + (jcol << 1)].im;
                d = rt_hypotd(absxk, ctoc);
                fs_re = absxk / d;
                fs_im = ctoc / d;
              }

              b_gs_re = gs_re / scale;
              gs_im = -gs_im / scale;
              gs_re = fs_re * b_gs_re - fs_im * gs_im;
              gs_im = fs_re * gs_im + fs_im * b_gs_re;
              f_re = c * A[jrow + (jcol << 1)].re + (gs_re * A[(jrow + (jcol <<
                1)) + 1].re - gs_im * A[(jrow + (jcol << 1)) + 1].im);
              f_im = c * A[jrow + (jcol << 1)].im + (gs_re * A[(jrow + (jcol <<
                1)) + 1].im + gs_im * A[(jrow + (jcol << 1)) + 1].re);
            }
          } else {
            absxk = std::sqrt(1.0 + scale / ctoc);
            f_re = absxk * fs_re;
            f_im = absxk * fs_im;
            c = 1.0 / absxk;
            d = ctoc + scale;
            ctoc = f_re / d;
            absxk = f_im / d;
            b_gs_re = gs_re;
            gs_re = ctoc * gs_re - absxk * -gs_im;
            gs_im = ctoc * -gs_im + absxk * b_gs_re;
            if (rescaledir > 0) {
              for (i = 0; i <= count; i++) {
                f_re *= 7.4428285367870146E+137;
                f_im *= 7.4428285367870146E+137;
              }
            } else {
              if (rescaledir < 0) {
                for (i = 0; i <= count; i++) {
                  f_re *= 1.3435752215134178E-138;
                  f_im *= 1.3435752215134178E-138;
                }
              }
            }
          }
        }

        A[f_re_tmp].re = f_re;
        A[f_re_tmp].im = f_im;
        A[b_y_tmp].re = 0.0;
        A[b_y_tmp].im = 0.0;
        b_gs_re = gs_re * A[1].re - gs_im * A[1].im;
        y_tmp = gs_re * A[1].im + gs_im * A[1].re;
        absxk = c * A[0].re;
        ctoc = c * A[0].im;
        scale = A[0].im;
        d = A[0].re;
        A[1].re = c * A[1].re - (gs_re * A[0].re + gs_im * A[0].im);
        A[1].im = c * A[1].im - (gs_re * scale - gs_im * d);
        A[0].re = absxk + b_gs_re;
        A[0].im = ctoc + y_tmp;
        b_gs_re = gs_re * A[3].re - gs_im * A[3].im;
        y_tmp = gs_re * A[3].im + gs_im * A[3].re;
        absxk = c * A[2].re;
        ctoc = c * A[2].im;
        scale = A[2].im;
        d = A[2].re;
        A[3].re = c * A[3].re - (gs_re * A[2].re + gs_im * A[2].im);
        A[3].im = c * A[3].im - (gs_re * scale - gs_im * d);
        A[2].re = absxk + b_gs_re;
        A[2].im = ctoc + y_tmp;
        gs_re = -gs_re;
        gs_im = -gs_im;
        b_gs_re = gs_re * A[0].re - gs_im * A[0].im;
        y_tmp = gs_re * A[0].im + gs_im * A[0].re;
        absxk = c * A[2].re;
        ctoc = c * A[2].im;
        scale = A[2].im;
        d = A[2].re;
        A[0].re = c * A[0].re - (gs_re * A[2].re + gs_im * A[2].im);
        A[0].im = c * A[0].im - (gs_re * scale - gs_im * d);
        A[2].re = absxk + b_gs_re;
        A[2].im = ctoc + y_tmp;
        b_gs_re = gs_re * A[1].re - gs_im * A[1].im;
        y_tmp = gs_re * A[1].im + gs_im * A[1].re;
        absxk = c * A[3].re;
        ctoc = c * A[3].im;
        scale = A[3].im;
        d = A[3].re;
        A[1].re = c * A[1].re - (gs_re * A[3].re + gs_im * A[3].im);
        A[1].im = c * A[1].im - (gs_re * scale - gs_im * d);
        A[3].re = absxk + b_gs_re;
        A[3].im = ctoc + y_tmp;
        b_gs_re = gs_re * V[0].re - gs_im * V[0].im;
        y_tmp = gs_re * V[0].im + gs_im * V[0].re;
        V[0].re = c * V[0].re - (gs_re * V[2].re + gs_im * V[2].im);
        V[0].im = c * V[0].im - (gs_re * V[2].im - gs_im * V[2].re);
        V[2].re = c * V[2].re + b_gs_re;
        V[2].im = c * V[2].im + y_tmp;
        b_gs_re = gs_re * V[1].re - gs_im * V[1].im;
        y_tmp = gs_re * V[1].im + gs_im * V[1].re;
        V[1].re = c * V[1].re - (gs_re * V[3].re + gs_im * V[3].im);
        V[1].im = c * V[1].im - (gs_re * V[3].im - gs_im * V[3].re);
        V[3].re = c * V[3].re + b_gs_re;
        V[3].im = c * V[3].im + y_tmp;
      }
    }
  }

  xzhgeqz(A, ilo, ihi, V, info, alpha1, beta1);
  if (*info == 0) {
    xztgevc(A, V);
    if (ilo > 1) {
      for (i = ilo - 2; i + 1 >= 1; i--) {
        rescaledir = rscale[i] - 1;
        if (rscale[i] != i + 1) {
          fs_re = V[i].re;
          fs_im = V[i].im;
          V[i] = V[rescaledir];
          V[rescaledir].re = fs_re;
          V[rescaledir].im = fs_im;
          fs_re = V[i + 2].re;
          fs_im = V[i + 2].im;
          V[i + 2] = V[rescaledir + 2];
          V[rescaledir + 2].re = fs_re;
          V[rescaledir + 2].im = fs_im;
        }
      }
    }

    if (ihi < 2) {
      rescaledir = rscale[1] - 1;
      if (rscale[1] != 2) {
        fs_re = V[1].re;
        fs_im = V[1].im;
        V[1] = V[rescaledir];
        V[rescaledir].re = fs_re;
        V[rescaledir].im = fs_im;
        fs_re = V[3].re;
        fs_im = V[3].im;
        V[3] = V[rescaledir + 2];
        V[rescaledir + 2].re = fs_re;
        V[rescaledir + 2].im = fs_im;
      }
    }

    absxk = std::abs(V[0].re) + std::abs(V[0].im);
    ctoc = std::abs(V[1].re) + std::abs(V[1].im);
    if (ctoc > absxk) {
      absxk = ctoc;
    }

    if (absxk >= 6.7178761075670888E-139) {
      absxk = 1.0 / absxk;
      V[0].re *= absxk;
      V[0].im *= absxk;
      V[1].re *= absxk;
      V[1].im *= absxk;
    }

    absxk = std::abs(V[2].re) + std::abs(V[2].im);
    ctoc = std::abs(V[3].re) + std::abs(V[3].im);
    if (ctoc > absxk) {
      absxk = ctoc;
    }

    if (absxk >= 6.7178761075670888E-139) {
      absxk = 1.0 / absxk;
      V[2].re *= absxk;
      V[2].im *= absxk;
      V[3].re *= absxk;
      V[3].im *= absxk;
    }

    if (ilascl) {
      notdone = true;
      while (notdone) {
        scale = anrmto * 2.0041683600089728E-292;
        d = anrm / 4.9896007738368E+291;
        if ((scale > anrm) && (anrm != 0.0)) {
          y_tmp = 2.0041683600089728E-292;
          anrmto = scale;
        } else if (d > anrmto) {
          y_tmp = 4.9896007738368E+291;
          anrm = d;
        } else {
          y_tmp = anrm / anrmto;
          notdone = false;
        }

        alpha1[0].re *= y_tmp;
        alpha1[0].im *= y_tmp;
        alpha1[1].re *= y_tmp;
        alpha1[1].im *= y_tmp;
      }
    }
  }
}

//
// Arguments    : creal_T A[4]
//                int ilo
//                int ihi
//                creal_T Z[4]
//                int *info
//                creal_T alpha1[2]
//                creal_T beta1[2]
// Return Type  : void
//
static void xzhgeqz(creal_T A[4], int ilo, int ihi, creal_T Z[4], int *info,
                    creal_T alpha1[2], creal_T beta1[2])
{
  double eshift_re;
  double eshift_im;
  creal_T ctemp;
  double anorm;
  double scale;
  double reAij;
  double sumsq;
  double b_atol;
  boolean_T firstNonZero;
  int j;
  int i61;
  double ascale;
  int jp1;
  double imAij;
  boolean_T guard1 = false;
  boolean_T guard2 = false;
  int ifirst;
  int istart;
  double temp2;
  int ilast;
  int ilastm1;
  int iiter;
  boolean_T goto60;
  boolean_T goto70;
  boolean_T goto90;
  int jiter;
  int exitg1;
  boolean_T b_guard1 = false;
  boolean_T guard3 = false;
  boolean_T exitg2;
  creal_T b_ascale;
  double c;
  creal_T shift;
  double ad22_re;
  double t1_im;
  double ad22_im;
  double shift_im;
  *info = 0;
  alpha1[0].re = 0.0;
  alpha1[0].im = 0.0;
  beta1[0].re = 1.0;
  beta1[0].im = 0.0;
  alpha1[1].re = 0.0;
  alpha1[1].im = 0.0;
  beta1[1].re = 1.0;
  beta1[1].im = 0.0;
  eshift_re = 0.0;
  eshift_im = 0.0;
  ctemp.re = 0.0;
  ctemp.im = 0.0;
  anorm = 0.0;
  if (ilo <= ihi) {
    scale = 0.0;
    sumsq = 0.0;
    firstNonZero = true;
    for (j = ilo; j <= ihi; j++) {
      i61 = j + 1;
      if (ihi < j + 1) {
        i61 = ihi;
      }

      for (jp1 = ilo; jp1 <= i61; jp1++) {
        reAij = A[(jp1 + ((j - 1) << 1)) - 1].re;
        imAij = A[(jp1 + ((j - 1) << 1)) - 1].im;
        if (reAij != 0.0) {
          anorm = std::abs(reAij);
          if (firstNonZero) {
            sumsq = 1.0;
            scale = anorm;
            firstNonZero = false;
          } else if (scale < anorm) {
            temp2 = scale / anorm;
            sumsq = 1.0 + sumsq * temp2 * temp2;
            scale = anorm;
          } else {
            temp2 = anorm / scale;
            sumsq += temp2 * temp2;
          }
        }

        if (imAij != 0.0) {
          anorm = std::abs(imAij);
          if (firstNonZero) {
            sumsq = 1.0;
            scale = anorm;
            firstNonZero = false;
          } else if (scale < anorm) {
            temp2 = scale / anorm;
            sumsq = 1.0 + sumsq * temp2 * temp2;
            scale = anorm;
          } else {
            temp2 = anorm / scale;
            sumsq += temp2 * temp2;
          }
        }
      }
    }

    anorm = scale * std::sqrt(sumsq);
  }

  reAij = 2.2204460492503131E-16 * anorm;
  b_atol = 2.2250738585072014E-308;
  if (reAij > 2.2250738585072014E-308) {
    b_atol = reAij;
  }

  reAij = 2.2250738585072014E-308;
  if (anorm > 2.2250738585072014E-308) {
    reAij = anorm;
  }

  ascale = 1.0 / reAij;
  firstNonZero = true;
  if (ihi + 1 <= 2) {
    alpha1[1] = A[3];
  }

  guard1 = false;
  guard2 = false;
  if (ihi >= ilo) {
    ifirst = ilo;
    istart = ilo;
    ilast = ihi - 1;
    ilastm1 = ihi - 2;
    iiter = 0;
    goto60 = false;
    goto70 = false;
    goto90 = false;
    jiter = 0;
    do {
      exitg1 = 0;
      if (jiter <= 30 * ((ihi - ilo) + 1) - 1) {
        b_guard1 = false;
        if (ilast + 1 == ilo) {
          goto60 = true;
          b_guard1 = true;
        } else {
          i61 = ilast + (ilastm1 << 1);
          if (std::abs(A[i61].re) + std::abs(A[ilast + (ilastm1 << 1)].im) <=
              b_atol) {
            A[i61].re = 0.0;
            A[i61].im = 0.0;
            goto60 = true;
            b_guard1 = true;
          } else {
            j = ilastm1;
            guard3 = false;
            exitg2 = false;
            while ((!exitg2) && (j + 1 >= ilo)) {
              if (j + 1 == ilo) {
                guard3 = true;
                exitg2 = true;
              } else if (std::abs(A[j].re) + std::abs(A[j].im) <= b_atol) {
                A[j].re = 0.0;
                A[j].im = 0.0;
                guard3 = true;
                exitg2 = true;
              } else {
                j--;
                guard3 = false;
              }
            }

            if (guard3) {
              ifirst = j + 1;
              goto70 = true;
            }

            if (goto70) {
              b_guard1 = true;
            } else {
              alpha1[0].re = 0.0;
              alpha1[0].im = 0.0;
              beta1[0].re = 0.0;
              beta1[0].im = 0.0;
              alpha1[1].re = 0.0;
              alpha1[1].im = 0.0;
              beta1[1].re = 0.0;
              beta1[1].im = 0.0;
              Z[0].re = 0.0;
              Z[0].im = 0.0;
              Z[1].re = 0.0;
              Z[1].im = 0.0;
              Z[2].re = 0.0;
              Z[2].im = 0.0;
              Z[3].re = 0.0;
              Z[3].im = 0.0;
              *info = 1;
              exitg1 = 1;
            }
          }
        }

        if (b_guard1) {
          if (goto60) {
            goto60 = false;
            alpha1[ilast] = A[ilast + (ilast << 1)];
            ilast = ilastm1;
            ilastm1--;
            if (ilast + 1 < ilo) {
              firstNonZero = false;
              guard2 = true;
              exitg1 = 1;
            } else {
              iiter = 0;
              eshift_re = 0.0;
              eshift_im = 0.0;
              jiter++;
            }
          } else {
            if (goto70) {
              goto70 = false;
              iiter++;
              if (iiter - iiter / 10 * 10 != 0) {
                anorm = ascale * A[ilastm1 + (ilastm1 << 1)].re;
                reAij = ascale * A[ilastm1 + (ilastm1 << 1)].im;
                if (reAij == 0.0) {
                  shift.re = anorm / 0.70710678118654746;
                  shift.im = 0.0;
                } else if (anorm == 0.0) {
                  shift.re = 0.0;
                  shift.im = reAij / 0.70710678118654746;
                } else {
                  shift.re = anorm / 0.70710678118654746;
                  shift.im = reAij / 0.70710678118654746;
                }

                jp1 = ilast << 1;
                anorm = ascale * A[ilast + jp1].re;
                reAij = ascale * A[ilast + (ilast << 1)].im;
                if (reAij == 0.0) {
                  ad22_re = anorm / 0.70710678118654746;
                  ad22_im = 0.0;
                } else if (anorm == 0.0) {
                  ad22_re = 0.0;
                  ad22_im = reAij / 0.70710678118654746;
                } else {
                  ad22_re = anorm / 0.70710678118654746;
                  ad22_im = reAij / 0.70710678118654746;
                }

                temp2 = 0.5 * (shift.re + ad22_re);
                t1_im = 0.5 * (shift.im + ad22_im);
                anorm = ascale * A[ilastm1 + jp1].re;
                reAij = ascale * A[ilastm1 + (ilast << 1)].im;
                if (reAij == 0.0) {
                  imAij = anorm / 0.70710678118654746;
                  sumsq = 0.0;
                } else if (anorm == 0.0) {
                  imAij = 0.0;
                  sumsq = reAij / 0.70710678118654746;
                } else {
                  imAij = anorm / 0.70710678118654746;
                  sumsq = reAij / 0.70710678118654746;
                }

                anorm = ascale * A[ilast + (ilastm1 << 1)].re;
                reAij = ascale * A[ilast + (ilastm1 << 1)].im;
                if (reAij == 0.0) {
                  scale = anorm / 0.70710678118654746;
                  anorm = 0.0;
                } else if (anorm == 0.0) {
                  scale = 0.0;
                  anorm = reAij / 0.70710678118654746;
                } else {
                  scale = anorm / 0.70710678118654746;
                  anorm = reAij / 0.70710678118654746;
                }

                shift_im = shift.re * ad22_im + shift.im * ad22_re;
                shift.re = ((temp2 * temp2 - t1_im * t1_im) + (imAij * scale -
                  sumsq * anorm)) - (shift.re * ad22_re - shift.im * ad22_im);
                shift.im = ((temp2 * t1_im + t1_im * temp2) + (imAij * anorm +
                  sumsq * scale)) - shift_im;
                if (shift.im == 0.0) {
                  if (shift.re < 0.0) {
                    anorm = 0.0;
                    reAij = std::sqrt(-shift.re);
                  } else {
                    anorm = std::sqrt(shift.re);
                    reAij = 0.0;
                  }
                } else if (shift.re == 0.0) {
                  if (shift.im < 0.0) {
                    anorm = std::sqrt(-shift.im / 2.0);
                    reAij = -anorm;
                  } else {
                    anorm = std::sqrt(shift.im / 2.0);
                    reAij = anorm;
                  }
                } else {
                  reAij = std::abs(shift.re);
                  anorm = std::abs(shift.im);
                  if ((reAij > 4.4942328371557893E+307) || (anorm >
                       4.4942328371557893E+307)) {
                    reAij *= 0.5;
                    anorm = rt_hypotd(reAij, anorm * 0.5);
                    if (anorm > reAij) {
                      anorm = std::sqrt(anorm) * std::sqrt(1.0 + reAij / anorm);
                    } else {
                      anorm = std::sqrt(anorm) * 1.4142135623730951;
                    }
                  } else {
                    anorm = std::sqrt((rt_hypotd(reAij, anorm) + reAij) * 0.5);
                  }

                  if (shift.re > 0.0) {
                    reAij = 0.5 * (shift.im / anorm);
                  } else {
                    if (shift.im < 0.0) {
                      reAij = -anorm;
                    } else {
                      reAij = anorm;
                    }

                    anorm = 0.5 * (shift.im / reAij);
                  }
                }

                if ((temp2 - ad22_re) * anorm + (t1_im - ad22_im) * reAij <= 0.0)
                {
                  shift.re = temp2 + anorm;
                  shift.im = t1_im + reAij;
                } else {
                  shift.re = temp2 - anorm;
                  shift.im = t1_im - reAij;
                }
              } else {
                anorm = ascale * A[ilast + (ilastm1 << 1)].re;
                reAij = ascale * A[ilast + (ilastm1 << 1)].im;
                if (reAij == 0.0) {
                  imAij = anorm / 0.70710678118654746;
                  sumsq = 0.0;
                } else if (anorm == 0.0) {
                  imAij = 0.0;
                  sumsq = reAij / 0.70710678118654746;
                } else {
                  imAij = anorm / 0.70710678118654746;
                  sumsq = reAij / 0.70710678118654746;
                }

                eshift_re += imAij;
                eshift_im += sumsq;
                shift.re = eshift_re;
                shift.im = eshift_im;
              }

              j = ilastm1;
              jp1 = ilastm1 + 1;
              exitg2 = false;
              while ((!exitg2) && (j + 1 > ifirst)) {
                istart = 2;
                ctemp.re = ascale * A[3].re - shift.re * 0.70710678118654746;
                ctemp.im = ascale * A[3].im - shift.im * 0.70710678118654746;
                anorm = std::abs(ctemp.re) + std::abs(ctemp.im);
                temp2 = ascale * (std::abs(A[2 + jp1].re) + std::abs(A[2 + jp1].
                  im));
                reAij = anorm;
                if (temp2 > anorm) {
                  reAij = temp2;
                }

                if ((reAij < 1.0) && (reAij != 0.0)) {
                  anorm /= reAij;
                  temp2 /= reAij;
                }

                if ((std::abs(A[1].re) + std::abs(A[1].im)) * temp2 <= anorm *
                    b_atol) {
                  goto90 = true;
                  exitg2 = true;
                } else {
                  jp1 = 1;
                  j = 0;
                }
              }

              if (!goto90) {
                istart = ifirst;
                ctemp.re = ascale * A[(ifirst + ((ifirst - 1) << 1)) - 1].re -
                  shift.re * 0.70710678118654746;
                ctemp.im = ascale * A[(ifirst + ((ifirst - 1) << 1)) - 1].im -
                  shift.im * 0.70710678118654746;
                goto90 = true;
              }
            }

            if (goto90) {
              goto90 = false;
              b_ascale.re = ascale * A[1 + ((istart - 1) << 1)].re;
              b_ascale.im = ascale * A[1 + ((istart - 1) << 1)].im;
              xzlartg(ctemp, b_ascale, &c, &shift);
              j = istart;
              while (j < ilast + 1) {
                anorm = shift.re * A[1].re - shift.im * A[1].im;
                reAij = shift.re * A[1].im + shift.im * A[1].re;
                sumsq = c * A[0].re;
                imAij = c * A[0].im;
                temp2 = A[0].im;
                t1_im = A[0].re;
                A[1].re = c * A[1].re - (shift.re * A[0].re + shift.im * A[0].im);
                A[1].im = c * A[1].im - (shift.re * temp2 - shift.im * t1_im);
                A[0].re = sumsq + anorm;
                A[0].im = imAij + reAij;
                scale = shift.re * A[3].re - shift.im * A[3].im;
                shift_im = shift.re * A[3].im + shift.im * A[3].re;
                sumsq = c * A[2].re;
                imAij = c * A[2].im;
                temp2 = A[2].im;
                t1_im = A[2].re;
                A[3].re = c * A[3].re - (shift.re * A[2].re + shift.im * A[2].im);
                A[3].im = c * A[3].im - (shift.re * temp2 - shift.im * t1_im);
                A[2].re = sumsq + scale;
                A[2].im = imAij + shift_im;
                shift.re = -shift.re;
                shift.im = -shift.im;
                scale = shift.re * A[0].re - shift.im * A[0].im;
                shift_im = shift.re * A[0].im + shift.im * A[0].re;
                anorm = c * A[2].re;
                reAij = c * A[2].im;
                temp2 = A[2].im;
                t1_im = A[2].re;
                A[0].re = c * A[0].re - (shift.re * A[2].re + shift.im * A[2].im);
                A[0].im = c * A[0].im - (shift.re * temp2 - shift.im * t1_im);
                A[2].re = anorm + scale;
                A[2].im = reAij + shift_im;
                scale = shift.re * A[1].re - shift.im * A[1].im;
                shift_im = shift.re * A[1].im + shift.im * A[1].re;
                anorm = c * A[3].re;
                reAij = c * A[3].im;
                temp2 = A[3].im;
                t1_im = A[3].re;
                A[1].re = c * A[1].re - (shift.re * A[3].re + shift.im * A[3].im);
                A[1].im = c * A[1].im - (shift.re * temp2 - shift.im * t1_im);
                A[3].re = anorm + scale;
                A[3].im = reAij + shift_im;
                ad22_re = c * Z[2].re + (shift.re * Z[0].re - shift.im * Z[0].im);
                ad22_im = c * Z[2].im + (shift.re * Z[0].im + shift.im * Z[0].re);
                anorm = Z[2].im;
                reAij = Z[2].re;
                Z[0].re = c * Z[0].re - (shift.re * Z[2].re + shift.im * Z[2].im);
                Z[0].im = c * Z[0].im - (shift.re * anorm - shift.im * reAij);
                Z[2].re = ad22_re;
                Z[2].im = ad22_im;
                ad22_re = c * Z[3].re + (shift.re * Z[1].re - shift.im * Z[1].im);
                ad22_im = c * Z[3].im + (shift.re * Z[1].im + shift.im * Z[1].re);
                anorm = Z[3].im;
                reAij = Z[3].re;
                Z[1].re = c * Z[1].re - (shift.re * Z[3].re + shift.im * Z[3].im);
                Z[1].im = c * Z[1].im - (shift.re * anorm - shift.im * reAij);
                Z[3].re = ad22_re;
                Z[3].im = ad22_im;
                j = 2;
              }
            }

            jiter++;
          }
        }
      } else {
        guard2 = true;
        exitg1 = 1;
      }
    } while (exitg1 == 0);
  } else {
    guard1 = true;
  }

  if (guard2) {
    if (firstNonZero) {
      *info = ilast + 1;
      for (jp1 = 0; jp1 <= ilast; jp1++) {
        alpha1[jp1].re = 0.0;
        alpha1[jp1].im = 0.0;
        beta1[jp1].re = 0.0;
        beta1[jp1].im = 0.0;
      }

      Z[0].re = 0.0;
      Z[0].im = 0.0;
      Z[1].re = 0.0;
      Z[1].im = 0.0;
      Z[2].re = 0.0;
      Z[2].im = 0.0;
      Z[3].re = 0.0;
      Z[3].im = 0.0;
    } else {
      guard1 = true;
    }
  }

  if (guard1) {
    for (j = 0; j <= ilo - 2; j++) {
      alpha1[j] = A[j + (j << 1)];
    }
  }
}

//
// Arguments    : const creal_T f
//                const creal_T g
//                double *cs
//                creal_T *sn
// Return Type  : void
//
static void xzlartg(const creal_T f, const creal_T g, double *cs, creal_T *sn)
{
  double y_tmp;
  double scale;
  double b_y_tmp;
  double f2s;
  double f2;
  double fs_re;
  double fs_im;
  double gs_re;
  double gs_im;
  boolean_T guard1 = false;
  double g2s;
  y_tmp = std::abs(f.re);
  scale = y_tmp;
  b_y_tmp = std::abs(f.im);
  if (b_y_tmp > y_tmp) {
    scale = b_y_tmp;
  }

  f2s = std::abs(g.re);
  f2 = std::abs(g.im);
  if (f2 > f2s) {
    f2s = f2;
  }

  if (f2s > scale) {
    scale = f2s;
  }

  fs_re = f.re;
  fs_im = f.im;
  gs_re = g.re;
  gs_im = g.im;
  guard1 = false;
  if (scale >= 7.4428285367870146E+137) {
    do {
      fs_re *= 1.3435752215134178E-138;
      fs_im *= 1.3435752215134178E-138;
      gs_re *= 1.3435752215134178E-138;
      gs_im *= 1.3435752215134178E-138;
      scale *= 1.3435752215134178E-138;
    } while (!(scale < 7.4428285367870146E+137));

    guard1 = true;
  } else if (scale <= 1.3435752215134178E-138) {
    if ((g.re == 0.0) && (g.im == 0.0)) {
      *cs = 1.0;
      sn->re = 0.0;
      sn->im = 0.0;
    } else {
      do {
        fs_re *= 7.4428285367870146E+137;
        fs_im *= 7.4428285367870146E+137;
        gs_re *= 7.4428285367870146E+137;
        gs_im *= 7.4428285367870146E+137;
        scale *= 7.4428285367870146E+137;
      } while (!(scale > 1.3435752215134178E-138));

      guard1 = true;
    }
  } else {
    guard1 = true;
  }

  if (guard1) {
    f2 = fs_re * fs_re + fs_im * fs_im;
    scale = gs_re * gs_re + gs_im * gs_im;
    f2s = scale;
    if (1.0 > scale) {
      f2s = 1.0;
    }

    if (f2 <= f2s * 2.0041683600089728E-292) {
      if ((f.re == 0.0) && (f.im == 0.0)) {
        *cs = 0.0;
        scale = rt_hypotd(gs_re, gs_im);
        sn->re = gs_re / scale;
        sn->im = -gs_im / scale;
      } else {
        g2s = std::sqrt(scale);
        *cs = rt_hypotd(fs_re, fs_im) / g2s;
        if (b_y_tmp > y_tmp) {
          y_tmp = b_y_tmp;
        }

        if (y_tmp > 1.0) {
          scale = rt_hypotd(f.re, f.im);
          fs_re = f.re / scale;
          fs_im = f.im / scale;
        } else {
          f2 = 7.4428285367870146E+137 * f.re;
          f2s = 7.4428285367870146E+137 * f.im;
          scale = rt_hypotd(f2, f2s);
          fs_re = f2 / scale;
          fs_im = f2s / scale;
        }

        gs_re /= g2s;
        gs_im = -gs_im / g2s;
        sn->re = fs_re * gs_re - fs_im * gs_im;
        sn->im = fs_re * gs_im + fs_im * gs_re;
      }
    } else {
      f2s = std::sqrt(1.0 + scale / f2);
      fs_re *= f2s;
      fs_im *= f2s;
      *cs = 1.0 / f2s;
      scale += f2;
      fs_re /= scale;
      fs_im /= scale;
      sn->re = fs_re * gs_re - fs_im * -gs_im;
      sn->im = fs_re * -gs_im + fs_im * gs_re;
    }
  }
}

//
// Arguments    : const creal_T A[4]
//                creal_T V[4]
// Return Type  : void
//
static void xztgevc(const creal_T A[4], creal_T V[4])
{
  double anorm;
  double xmx;
  double y;
  double ascale;
  double temp;
  double salpha_re;
  double salpha_im;
  double acoeff;
  boolean_T lscalea;
  boolean_T lscaleb;
  double scale;
  double dmin;
  creal_T work1[2];
  int j;
  creal_T work2[2];
  double d_im;
  int jr;
  anorm = std::abs(A[0].re) + std::abs(A[0].im);
  xmx = std::abs(A[3].re) + std::abs(A[3].im);
  y = (std::abs(A[2].re) + std::abs(A[2].im)) + xmx;
  if (y > anorm) {
    anorm = y;
  }

  y = anorm;
  if (2.2250738585072014E-308 > anorm) {
    y = 2.2250738585072014E-308;
  }

  ascale = 1.0 / y;
  y = xmx * ascale;
  if (1.0 > y) {
    y = 1.0;
  }

  temp = 1.0 / y;
  salpha_re = ascale * (temp * A[3].re);
  salpha_im = ascale * (temp * A[3].im);
  acoeff = temp * ascale;
  if ((temp >= 2.2250738585072014E-308) && (std::abs(acoeff) <
       2.0041683600089728E-292)) {
    lscalea = true;
  } else {
    lscalea = false;
  }

  xmx = std::abs(salpha_re) + std::abs(salpha_im);
  if ((xmx >= 2.2250738585072014E-308) && (xmx < 2.0041683600089728E-292)) {
    lscaleb = true;
  } else {
    lscaleb = false;
  }

  scale = 1.0;
  if (lscalea) {
    y = anorm;
    if (4.9896007738368E+291 < anorm) {
      y = 4.9896007738368E+291;
    }

    scale = 2.0041683600089728E-292 / temp * y;
  }

  if (lscaleb) {
    y = 2.0041683600089728E-292 / xmx;
    if (y > scale) {
      scale = y;
    }
  }

  if (lscalea || lscaleb) {
    y = std::abs(acoeff);
    if (1.0 > y) {
      y = 1.0;
    }

    if (xmx > y) {
      y = xmx;
    }

    y = 1.0 / (2.2250738585072014E-308 * y);
    if (y < scale) {
      scale = y;
    }

    if (lscalea) {
      acoeff = ascale * (scale * temp);
    } else {
      acoeff *= scale;
    }

    salpha_re *= scale;
    salpha_im *= scale;
  }

  dmin = 2.2204460492503131E-16 * std::abs(acoeff) * anorm;
  y = 2.2204460492503131E-16 * (std::abs(salpha_re) + std::abs(salpha_im));
  if (y > dmin) {
    dmin = y;
  }

  if (2.2250738585072014E-308 > dmin) {
    dmin = 2.2250738585072014E-308;
  }

  work1[0].re = acoeff * A[2].re;
  work1[0].im = acoeff * A[2].im;
  work1[1].re = 1.0;
  work1[1].im = 0.0;
  for (j = 0; j < 1; j++) {
    anorm = acoeff * A[0].re - salpha_re;
    d_im = acoeff * A[0].im - salpha_im;
    if (std::abs(anorm) + std::abs(d_im) <= dmin) {
      anorm = dmin;
      d_im = 0.0;
    }

    xmx = std::abs(anorm) + std::abs(d_im);
    if (xmx < 1.0) {
      y = std::abs(work1[0].re) + std::abs(work1[0].im);
      if (y >= 2.2471164185778949E+307 * xmx) {
        temp = 1.0 / y;
        for (jr = 0; jr < 2; jr++) {
          work1[jr].re *= temp;
          work1[jr].im *= temp;
        }
      }
    }

    ascale = -work1[0].re;
    if (d_im == 0.0) {
      if (-work1[0].im == 0.0) {
        work1[0].re = -work1[0].re / anorm;
        work1[0].im = 0.0;
      } else if (-work1[0].re == 0.0) {
        work1[0].re = 0.0;
        work1[0].im = -work1[0].im / anorm;
      } else {
        work1[0].re = -work1[0].re / anorm;
        work1[0].im = -work1[0].im / anorm;
      }
    } else if (anorm == 0.0) {
      if (-work1[0].re == 0.0) {
        work1[0].re = -work1[0].im / d_im;
        work1[0].im = 0.0;
      } else if (-work1[0].im == 0.0) {
        work1[0].re = 0.0;
        work1[0].im = -(ascale / d_im);
      } else {
        work1[0].re = -work1[0].im / d_im;
        work1[0].im = -(ascale / d_im);
      }
    } else {
      scale = std::abs(anorm);
      xmx = std::abs(d_im);
      if (scale > xmx) {
        y = d_im / anorm;
        xmx = anorm + y * d_im;
        work1[0].re = (-work1[0].re + y * -work1[0].im) / xmx;
        work1[0].im = (-work1[0].im - y * ascale) / xmx;
      } else if (xmx == scale) {
        if (anorm > 0.0) {
          y = 0.5;
        } else {
          y = -0.5;
        }

        if (d_im > 0.0) {
          xmx = 0.5;
        } else {
          xmx = -0.5;
        }

        work1[0].re = (-work1[0].re * y + -work1[0].im * xmx) / scale;
        work1[0].im = (-work1[0].im * y - ascale * xmx) / scale;
      } else {
        y = anorm / d_im;
        xmx = d_im + y * anorm;
        work1[0].re = (y * -work1[0].re + -work1[0].im) / xmx;
        work1[0].im = (y * -work1[0].im - ascale) / xmx;
      }
    }
  }

  work2[0].re = 0.0;
  work2[0].im = 0.0;
  work2[1].re = 0.0;
  work2[1].im = 0.0;
  for (j = 0; j < 2; j++) {
    jr = j << 1;
    work2[0].re += V[jr].re * work1[j].re - V[j << 1].im * work1[j].im;
    work2[0].im += V[j << 1].re * work1[j].im + V[j << 1].im * work1[j].re;
    work2[1].re += V[1 + jr].re * work1[j].re - V[1 + (j << 1)].im * work1[j].im;
    work2[1].im += V[1 + (j << 1)].re * work1[j].im + V[1 + (j << 1)].im *
      work1[j].re;
  }

  xmx = std::abs(work2[0].re) + std::abs(work2[0].im);
  y = std::abs(work2[1].re) + std::abs(work2[1].im);
  if (y > xmx) {
    xmx = y;
  }

  if (xmx > 2.2250738585072014E-308) {
    temp = 1.0 / xmx;
    V[2].re = temp * work2[0].re;
    V[2].im = temp * work2[0].im;
    V[3].re = temp * work2[1].re;
    V[3].im = temp * work2[1].im;
  } else {
    V[2].re = 0.0;
    V[2].im = 0.0;
    V[3].re = 0.0;
    V[3].im = 0.0;
  }

  work2[0].re = 0.0;
  work2[0].im = 0.0;
  work2[1].re = 0.0;
  work2[1].im = 0.0;
  for (j = 0; j < 1; j++) {
    work2[0].re += V[0].re;
    work2[0].im += V[0].im;
    work2[1].re += V[1].re;
    work2[1].im += V[1].im;
  }

  xmx = std::abs(work2[0].re) + std::abs(work2[0].im);
  y = std::abs(work2[1].re) + std::abs(work2[1].im);
  if (y > xmx) {
    xmx = y;
  }

  if (xmx > 2.2250738585072014E-308) {
    temp = 1.0 / xmx;
    V[0].re = temp * work2[0].re;
    V[0].im = temp * work2[0].im;
    V[1].re = temp * work2[1].re;
    V[1].im = temp * work2[1].im;
  } else {
    V[0].re = 0.0;
    V[0].im = 0.0;
    V[1].re = 0.0;
    V[1].im = 0.0;
  }
}

//
// function [smartLoaderStruct, heightMap_res, debugPtCloudSenceXyz, debugPtCloudSenceIntensity] = PerceptionSmartLoader(configParams, xyz, intensity)
// Arguments    : PerceptionSmartLoaderStackData *SD
//                const PerceptionSmartLoaderConfigParam *configParams
//                const double xyz_data[]
//                const int xyz_size[2]
//                const double intensity_data[]
//                const int intensity_size[1]
//                PerceptionSmartLoaderStruct *smartLoaderStruct
//                float heightMap_res_data[]
//                int heightMap_res_size[2]
// Return Type  : void
//
void PerceptionSmartLoader(PerceptionSmartLoaderStackData *SD, const
  PerceptionSmartLoaderConfigParam *configParams, const double xyz_data[], const
  int xyz_size[2], const double intensity_data[], const int [1],
  PerceptionSmartLoaderStruct *smartLoaderStruct, float heightMap_res_data[],
  int heightMap_res_size[2])
{
  int ptCloudSenceXyz_size[2];
  int ptCloudSenceIntensity_size[1];
  b_struct_T r0;
  int tmp_size_idx_0;
  b_struct_T tmp_data[33];
  unsigned long b_tmp_data[33];
  int i0;
  int i1;
  int tmp_size_idx_0_tmp;
  signed char c_tmp_data[32];
  unsigned long d_tmp_data[32];
  b_struct_T e_tmp_data[32];

  // 'PerceptionSmartLoader:4' coder.cstructname(configParams, 'PerceptionSmartLoaderConfigParam'); 
  // 'PerceptionSmartLoader:5' assert(size(xyz,2) == 3);
  // 'PerceptionSmartLoader:6' assert(size(intensity,2) == 1);
  // 'PerceptionSmartLoader:7' if coder.target('Matlab')
  //  Parameters
  //  BinaryWriteImage(xyz, 'D:\git\cpp\SmartLoader\SmartLoaderDataset\Test3\xyz') 
  //  BinaryWriteImage(intensity, 'D:\git\cpp\SmartLoader\SmartLoaderDataset\Test3\intensity') 
  // percisionMode = 'double';
  //
  // 'PerceptionSmartLoader:19' if ~SmartLoaderGlobal.isInitialized
  if (!SD->pd->SmartLoaderGlobal.isInitialized) {
    // 'PerceptionSmartLoader:20' SmartLoaderGlobalInit();
    SmartLoaderGlobalInit(SD);
  }

  // 'PerceptionSmartLoader:23' debugPtCloudSenceXyz = zeros(0,0,'uint8');
  // 'PerceptionSmartLoader:24' debugPtCloudSenceIntensity = zeros(0,0,'uint8'); 
  // 'PerceptionSmartLoader:25' heightMap_res = zeros(0,0,'single');
  heightMap_res_size[1] = 0;
  heightMap_res_size[0] = 0;

  //  figure, PlotPointCloud([xyz double(intensity)])
  // 'PerceptionSmartLoader:29' smartLoaderStruct = GetSmartLoaderStruct();
  // 'PerceptionSmartLoader:30' smartLoaderStruct.status = PerceptionSmartLoaderReturnValue.eSuccess; 
  //  Align the point cloud to the sensor
  // 'PerceptionSmartLoader:33' [smartLoaderStruct, ptCloudSenceXyz, ptCloudSenceIntensity] = SmartLoaderAlignPointCloud(smartLoaderStruct, configParams, xyz, intensity); 
  smartLoaderStruct->heightMapStatus = false;
  smartLoaderStruct->loaderLocStatus = false;
  smartLoaderStruct->loaderLoc[0] = 0.0;
  smartLoaderStruct->shovelLoc[0] = 0.0;
  smartLoaderStruct->loaderLoc[1] = 0.0;
  smartLoaderStruct->shovelLoc[1] = 0.0;
  smartLoaderStruct->loaderLoc[2] = 0.0;
  smartLoaderStruct->shovelLoc[2] = 0.0;
  smartLoaderStruct->shovelLocStatus = false;
  smartLoaderStruct->loaderYawAngleDeg = 0.0;
  smartLoaderStruct->loaderYawAngleDegSmooth = 0.0;
  smartLoaderStruct->loaderYawAngleStatus = false;
  smartLoaderStruct->loaderToShovelYawAngleDeg = 0.0;
  smartLoaderStruct->loaderToShovelYawAngleDegSmooth = 0.0;
  smartLoaderStruct->loaderToShovelYawAngleDegStatus = false;
  smartLoaderStruct->status = PerceptionSmartLoaderReturnValue_eSuccess;
  SmartLoaderAlignPointCloud(SD, smartLoaderStruct,
    configParams->pcAlignmentProjMat, configParams->xyzLimits,
    configParams->minNumPointsInPc, xyz_data, xyz_size, intensity_data,
    SD->f23.ptCloudSenceXyz_data, ptCloudSenceXyz_size,
    SD->f23.ptCloudSenceIntensity_data, ptCloudSenceIntensity_size);

  // 'PerceptionSmartLoader:34' if smartLoaderStruct.status ~= PerceptionSmartLoaderReturnValue.eSuccess 
  if (!(smartLoaderStruct->status != PerceptionSmartLoaderReturnValue_eSuccess))
  {
    //  figure, PlotPointCloud([ptCloudSenceXyz, ptCloudSenceIntensity])
    //  Create height map
    // 'PerceptionSmartLoader:38' [smartLoaderStruct, heightMap_res] = SmartLoaderCreateHeightMap(smartLoaderStruct, configParams, ptCloudSenceXyz); 
    SmartLoaderCreateHeightMap(SD, smartLoaderStruct, configParams->xyzLimits,
      configParams->heightMapResolutionMeterToPixel,
      SD->f23.ptCloudSenceXyz_data, ptCloudSenceXyz_size, heightMap_res_data,
      heightMap_res_size);

    //  figure, imagesc(heightMap_res);
    //  figure, imshow(heightMap_res, []);
    //  Estiamte the loader, shovel locations and angles
    // 'PerceptionSmartLoader:43' [smartLoaderStruct] = SmartLoaderEstiamteLocations(smartLoaderStruct, configParams, ptCloudSenceXyz, ptCloudSenceIntensity); 
    SmartLoaderEstiamteLocations(SD, smartLoaderStruct, configParams,
      SD->f23.ptCloudSenceXyz_data, ptCloudSenceXyz_size,
      SD->f23.ptCloudSenceIntensity_data, ptCloudSenceIntensity_size);

    // 'PerceptionSmartLoader:44' if smartLoaderStruct.status ~= PerceptionSmartLoaderReturnValue.eSuccess 
    if (!(smartLoaderStruct->status != PerceptionSmartLoaderReturnValue_eSuccess))
    {
      //
      // 'PerceptionSmartLoader:47' smartLoaderStruct = SmartLoaderSmoothAngles(smartLoaderStruct, configParams); 
      SmartLoaderSmoothAngles(SD, smartLoaderStruct,
        configParams->loaderYawAngleSmoothWeight,
        configParams->loaderToShovelYawAngleSmoothWeight);

      //  Plot the point cloud with an image at the side
      // 'PerceptionSmartLoader:50' if coder.target('Matlab') && configParams.debugMode 
      //  Save the location history
      // 'PerceptionSmartLoader:55' if smartLoaderStruct.loaderLocStatus
      if (smartLoaderStruct->loaderLocStatus) {
        // 'PerceptionSmartLoader:56' SmartLoaderGlobal.smartLoaderStructHistory = [SmartLoaderGlobal.smartLoaderStructHistory; smartLoaderStruct]; 
        r0.heightMapStatus = smartLoaderStruct->heightMapStatus;
        r0.loaderLocStatus = smartLoaderStruct->loaderLocStatus;
        r0.loaderLoc[0] = smartLoaderStruct->loaderLoc[0];
        r0.shovelLoc[0] = smartLoaderStruct->shovelLoc[0];
        r0.loaderLoc[1] = smartLoaderStruct->loaderLoc[1];
        r0.shovelLoc[1] = smartLoaderStruct->shovelLoc[1];
        r0.loaderLoc[2] = smartLoaderStruct->loaderLoc[2];
        r0.shovelLoc[2] = smartLoaderStruct->shovelLoc[2];
        r0.shovelLocStatus = smartLoaderStruct->shovelLocStatus;
        r0.loaderYawAngleDeg = smartLoaderStruct->loaderYawAngleDeg;
        r0.loaderYawAngleDegSmooth = smartLoaderStruct->loaderYawAngleDegSmooth;
        r0.loaderYawAngleStatus = smartLoaderStruct->loaderYawAngleStatus;
        r0.loaderToShovelYawAngleDeg =
          smartLoaderStruct->loaderToShovelYawAngleDeg;
        r0.loaderToShovelYawAngleDegSmooth =
          smartLoaderStruct->loaderToShovelYawAngleDegSmooth;
        r0.loaderToShovelYawAngleDegStatus =
          smartLoaderStruct->loaderToShovelYawAngleDegStatus;
        r0.status = smartLoaderStruct->status;
        tmp_size_idx_0 = SD->pd->
          SmartLoaderGlobal.smartLoaderStructHistory.size[0] + 1;
        if (0 <= SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] - 1)
        {
          memcpy(&tmp_data[0], &SD->
                 pd->SmartLoaderGlobal.smartLoaderStructHistory.data[0],
                 (unsigned int)(SD->
                                pd->SmartLoaderGlobal.smartLoaderStructHistory.size
                                [0] * (int)sizeof(b_struct_T)));
        }

        tmp_data[SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0]] =
          r0;
        SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0]++;
        if (0 <= tmp_size_idx_0 - 1) {
          memcpy(&SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.data[0],
                 &tmp_data[0], (unsigned int)(tmp_size_idx_0 * (int)sizeof
                  (b_struct_T)));
        }

        // 'PerceptionSmartLoader:57' SmartLoaderGlobal.loaderTimeTatHistoryMs = [SmartLoaderGlobal.loaderTimeTatHistoryMs; configParams.timeTagMs]; 
        tmp_size_idx_0 = SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0]
          + 1;
        if (0 <= SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] - 1) {
          memcpy(&b_tmp_data[0], &SD->
                 pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[0], (unsigned
                  int)(SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] *
                       (int)sizeof(unsigned long)));
        }

        b_tmp_data[SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0]] =
          configParams->timeTagMs;
        SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0]++;
        if (0 <= tmp_size_idx_0 - 1) {
          memcpy(&SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[0],
                 &b_tmp_data[0], (unsigned int)(tmp_size_idx_0 * (int)sizeof
                  (unsigned long)));
        }

        // 'PerceptionSmartLoader:59' if size(SmartLoaderGlobal.loaderTimeTatHistoryMs,1) >= SmartLoaderCompilationConstants.MaxHistorySize - 1 
        if (SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] >= 31) {
          // 'PerceptionSmartLoader:60' shiftArrayBy = 10;
          // 'PerceptionSmartLoader:62' SmartLoaderGlobal.loaderTimeTatHistoryMs = ... 
          // 'PerceptionSmartLoader:63'             SmartLoaderGlobal.loaderTimeTatHistoryMs((SmartLoaderCompilationConstants.MaxHistorySize - shiftArrayBy):end); 
          if (22 > SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0]) {
            i0 = 1;
            i1 = 0;
          } else {
            i0 = 22;
            i1 = SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0];
          }

          tmp_size_idx_0_tmp = (signed char)i1 - i0;
          tmp_size_idx_0 = tmp_size_idx_0_tmp + 1;
          for (i1 = 0; i1 <= tmp_size_idx_0_tmp; i1++) {
            c_tmp_data[i1] = (signed char)((signed char)(i0 + i1) - 1);
          }

          for (i0 = 0; i0 < tmp_size_idx_0; i0++) {
            d_tmp_data[i0] = SD->
              pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[c_tmp_data[i0]];
          }

          SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] =
            tmp_size_idx_0;
          if (0 <= tmp_size_idx_0 - 1) {
            memcpy(&SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[0],
                   &d_tmp_data[0], (unsigned int)(tmp_size_idx_0 * (int)sizeof
                    (unsigned long)));
          }

          // 'PerceptionSmartLoader:65' SmartLoaderGlobal.smartLoaderStructHistory = ... 
          // 'PerceptionSmartLoader:66'             SmartLoaderGlobal.smartLoaderStructHistory((SmartLoaderCompilationConstants.MaxHistorySize - shiftArrayBy):end,:); 
          if (22 > SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0]) {
            i0 = 0;
            i1 = 0;
          } else {
            i0 = 21;
            i1 = SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0];
          }

          tmp_size_idx_0_tmp = i1 - i0;
          for (i1 = 0; i1 < tmp_size_idx_0_tmp; i1++) {
            e_tmp_data[i1] = SD->
              pd->SmartLoaderGlobal.smartLoaderStructHistory.data[i0 + i1];
          }

          SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] =
            tmp_size_idx_0_tmp;
          if (0 <= tmp_size_idx_0_tmp - 1) {
            memcpy(&SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.data[0],
                   &e_tmp_data[0], (unsigned int)(tmp_size_idx_0_tmp * (int)
                    sizeof(b_struct_T)));
          }
        }
      }

      // 'PerceptionSmartLoader:70' if coder.target('Matlab')
    }
  }
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
// Return Type  : void
//
void PerceptionSmartLoader_initialize(PerceptionSmartLoaderStackData *SD)
{
  int j;
  b_struct_T y_smartLoaderStructHistory_data[32];
  unsigned long y_loaderTimeTatHistoryMs_data[32];
  omp_init_nest_lock(&emlrtNestLockGlobal);
  emxInitStruct_struct_T(&SD->pd->SmartLoaderGlobal);
  for (j = 0; j < 32; j++) {
    y_smartLoaderStructHistory_data[j].heightMapStatus = false;
    y_smartLoaderStructHistory_data[j].loaderLocStatus = false;
    y_smartLoaderStructHistory_data[j].loaderLoc[0] = 0.0;
    y_smartLoaderStructHistory_data[j].shovelLoc[0] = 0.0;
    y_smartLoaderStructHistory_data[j].loaderLoc[1] = 0.0;
    y_smartLoaderStructHistory_data[j].shovelLoc[1] = 0.0;
    y_smartLoaderStructHistory_data[j].loaderLoc[2] = 0.0;
    y_smartLoaderStructHistory_data[j].shovelLoc[2] = 0.0;
    y_smartLoaderStructHistory_data[j].shovelLocStatus = false;
    y_smartLoaderStructHistory_data[j].loaderYawAngleDeg = 0.0;
    y_smartLoaderStructHistory_data[j].loaderYawAngleDegSmooth = 0.0;
    y_smartLoaderStructHistory_data[j].loaderYawAngleStatus = false;
    y_smartLoaderStructHistory_data[j].loaderToShovelYawAngleDeg = 0.0;
    y_smartLoaderStructHistory_data[j].loaderToShovelYawAngleDegSmooth = 0.0;
    y_smartLoaderStructHistory_data[j].loaderToShovelYawAngleDegStatus = false;
    y_smartLoaderStructHistory_data[j].status =
      PerceptionSmartLoaderReturnValue_eFailed;
    y_loaderTimeTatHistoryMs_data[j] = 0UL;
  }

  SD->pd->SmartLoaderGlobal.isInitialized = false;
  SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.size[0] = 32;
  memcpy(&SD->pd->SmartLoaderGlobal.smartLoaderStructHistory.data[0],
         &y_smartLoaderStructHistory_data[0], (unsigned int)(32 * (int)sizeof
          (b_struct_T)));
  SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.size[0] = 32;
  memcpy(&SD->pd->SmartLoaderGlobal.loaderTimeTatHistoryMs.data[0],
         &y_loaderTimeTatHistoryMs_data[0], (unsigned int)(32 * (int)sizeof
          (unsigned long)));
  SmartLoaderCreateHeightMap_init(SD);
  eml_rand_mt19937ar_stateful_init(SD);
  emlrtInitThreadStackData();
}

//
// Arguments    : PerceptionSmartLoaderStackData *SD
// Return Type  : void
//
void PerceptionSmartLoader_terminate(PerceptionSmartLoaderStackData *SD)
{
  emlrtFreeThreadStackData();
  SmartLoaderCreateHeightMap_free(SD);
  omp_destroy_nest_lock(&emlrtNestLockGlobal);
}

//
// File trailer for PerceptionSmartLoader.cpp
//
// [EOF]
//
