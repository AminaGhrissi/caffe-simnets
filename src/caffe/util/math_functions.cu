#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_transpose<float>(const int M, const int N, const float * A, float * B, const cudaStream_t stream) {
  const float alpha = 1;
  const float beta = 0;
  cublasSetStream(Caffe::cublas_handle(), stream);
  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, M, N,
    &alpha, A, N, &beta, B, M, B, M));
  if (stream != 0) {
    cublasSetStream(Caffe::cublas_handle(), 0);
  }
}

template <>
void caffe_gpu_transpose<double>(const int M, const int N, const double * A, double * B, const cudaStream_t stream) {
  const double alpha = 1;
  const double beta = 0;
  cublasSetStream(Caffe::cublas_handle(), stream);
  CUBLAS_CHECK(cublasDgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, M, N,
    &alpha, A, N, &beta, B, M, B, M));
  if (stream != 0) {
    cublasSetStream(Caffe::cublas_handle(), 0);
  }
}

template <>
void caffe_gpu_nrm2(const int N, const float* X, float* result) {
  CUBLAS_CHECK(cublasSnrm2(Caffe::cublas_handle(), N, X, 1, result));
}

template <>
void caffe_gpu_nrm2(const int N, const double* X, double* result) {
  CUBLAS_CHECK(cublasDnrm2(Caffe::cublas_handle(), N, X, 1, result));
}

template <typename Dtype>
Dtype caffe_gpu_nrm2(const int N, const Dtype* X) {
  Dtype result;
  caffe_gpu_nrm2(N, X, &result);
  return result;
}
template float caffe_gpu_nrm2(const int N, const float* X);
template double caffe_gpu_nrm2(const int N, const double* X);

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mask_kernel(const int n, const Dtype* x,
    const Dtype* mask, const Dtype threshold, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = x[index] * (mask[index] > threshold);
  }
}

template <>
void caffe_gpu_mask<float>(const int N, const float* a,
    const float* b, const float threshold, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mask_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, threshold, y);
}

template <>
void caffe_gpu_mask<double>(const int N, const double* a,
    const double* b, const double threshold, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mask_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, threshold, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype, bool ADD_FUDGE = false>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y, const Dtype fudge_factor = 0) {
  CUDA_KERNEL_LOOP(index, n) {
    if (ADD_FUDGE) {
      y[index] = log(a[index] + fudge_factor);
    } else {
      y[index] = log(a[index]);
    }
  }
}

template <typename Dtype>
void caffe_gpu_log(const int N, const Dtype* a, Dtype* y, const Dtype fudge_factor) {
  if (fudge_factor == 0) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    log_kernel<Dtype, false><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, y);
  } else {
    CHECK_GT(fudge_factor, 0) << "The fudge factor must be positive!";
    log_kernel<Dtype, true><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, y, fudge_factor);
  }
}

template void caffe_gpu_log<float>(const int N, const float* a, float* y, const float fudge_factor);
template void caffe_gpu_log<double>(const int N, const double* a, double* y, const double fudge_factor);

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sqr, y[index] = x[index] * x[index]);
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(inv, y[index] = Dtype(1.0) / x[index]);
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(invsqrt, y[index] = Dtype(1.0) / sqrt(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

template <typename Dtype>
__global__ void threshold_kernel(const int n, const Dtype threshold, Dtype* inout) {
  CUDA_KERNEL_LOOP(index, n) {
    inout[index] = Dtype(inout[index] > threshold);
  }
}

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, Dtype* r) {
  caffe_gpu_rng_uniform<Dtype>(n, 0, 1, r);
  threshold_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, Dtype(1)-p, r);
}

template void caffe_gpu_rng_bernoulli<float>(const int n, const float p, float* r);
template void caffe_gpu_rng_bernoulli<double>(const int n, const double p, double* r);

template <typename Dtype>
__global__ void clip_min_kernel(const int n, const Dtype* x,
  Dtype* y, const Dtype min_value) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = max(x[index], min_value);
  }
}

template <>
void caffe_gpu_clip_min<float>(const int n, const float* x, float* y, const float min_value) {
  if (isnan(min_value)) {
    caffe_copy(n, x, y);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    clip_min_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, x, y, min_value);
  }
}

template <>
void caffe_gpu_clip_min<double>(const int n, const double* x, double* y, const double min_value) {
  if (isnan(min_value)) {
    caffe_copy(n, x, y);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    clip_min_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, x, y, min_value);
  }
}

template <>
void caffe_gpu_clip_min<int>(const int n, const int* x, int* y, const int min_value) {
  if (isnan(min_value)) {
    caffe_copy(n, x, y);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    clip_min_kernel<int><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, x, y, min_value);
  }
}

template <>
void caffe_gpu_clip_min<unsigned int>(const int n, const unsigned int* x,
  unsigned int* y, const unsigned int min_value) {
  if (isnan(min_value)) {
    caffe_copy(n, x, y);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    clip_min_kernel<unsigned int><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, x, y, min_value);
  }
}


template <typename Dtype>
__global__ void clip_max_kernel(const int n, const Dtype* a,
    Dtype* y, const Dtype max_value) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = min(a[index], max_value);
  }
}

template <>
void caffe_gpu_clip_max<float>(const int n, const float* x, float* y, const float max_value) {
  if (isnan(max_value)) {
    caffe_copy(n, x, y);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    clip_max_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, x, y, max_value);
  }
}

template <>
void caffe_gpu_clip_max<double>(const int n, const double* x, double* y, const double max_value) {
  if (isnan(max_value)) {
    caffe_copy(n, x, y);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    clip_max_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, x, y, max_value);
  }
}

template <>
void caffe_gpu_clip_max<int>(const int n, const int* x, int* y, const int max_value) {
  if (isnan(max_value)) {
    caffe_copy(n, x, y);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    clip_max_kernel<int><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, x, y, max_value);
  }
}

template <>
void caffe_gpu_clip_max<unsigned int>(const int n, const unsigned int* x, 
  unsigned int* y, const unsigned int max_value) {
  if (isnan(max_value)) {
    caffe_copy(n, x, y);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    clip_max_kernel<unsigned int><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
        n, x, y, max_value);
  }
}

template <>
void caffe_gpu_dgmm<float>(const CBLAS_SIDE mode, const int M, const int N,
    const float* A, const float* x, float* C) {
  cublasSideMode_t cuMode =
      (mode == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  CUBLAS_CHECK(cublasSdgmm(Caffe::cublas_handle(), cuMode, N, M,
      A, N, x, 1, C, N));
}
template <>
void caffe_gpu_dgmm<double>(const CBLAS_SIDE mode, const int M, const int N,
    const double* A, const double* x, double* C) {
  cublasSideMode_t cuMode =
      (mode == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  CUBLAS_CHECK(cublasDdgmm(Caffe::cublas_handle(), cuMode, N, M,
      A, N, x, 1, C, N));
}

template <typename Dtype, int BLOCK_SIZE>
__global__ void kernel_norm_forward(const int patches, const int dim, const Dtype fudge, Dtype* patches_rows,
    const bool normalize_variance) {
  __shared__ Dtype sdata[BLOCK_SIZE];
  const int bx = blockIdx.x;
  const int dx = BLOCK_SIZE;
  const int tid = threadIdx.x;
  // Strided sum of the elements of the patch loaded to shared memory
  sdata[tid] = 0;
  for (int i = tid, pos = bx * dim + tid; i < dim; i += dx, pos += dx) {
    sdata[tid] += patches_rows[pos];
  }
  __syncthreads();
  // Reduction sum
#pragma unroll
  for (unsigned int s=BLOCK_SIZE/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // Total sum saved in first element of array
  const Dtype mean = sdata[0] / dim;

  Dtype std = Dtype(1);
  if (normalize_variance) {
    // Stided sum of the squared elements of the patch after subtracting mean
    sdata[tid] = 0;
    for (int i = tid, pos = bx * dim + tid; i < dim; i += dx, pos += dx) {
      const Dtype x1 = patches_rows[pos] - mean;
      sdata[tid] += x1 * x1;
    }
    __syncthreads();
    // Reduction sum
  #pragma unroll
    for (unsigned int s=BLOCK_SIZE/2; s>0; s>>=1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }
    // Total sum saved in first element of array
    const Dtype variance = sdata[0] / max(dim - 1, 1);
    std = sqrt(variance + fudge);
  }

  // Normalize patch with the calculated mean and variance
  for (int i = tid; i < dim; i += dx) {
    const Dtype x = patches_rows[bx * dim + i];
    patches_rows[bx * dim + i] = (x - mean) / std;
  }
}

template <typename Dtype>
void caffe_gpu_normalize_patches_rows_forward(const int K, const int N, const Dtype fudge_factor,
  Dtype* row_buffer, const bool normalize_variance) {
  int num_threads = min(CAFFE_CUDA_NUM_THREADS, 256);
  //cudaStream_t cuStream = Caffe::cuda_stream(stream);
  while (2*num_threads > K) {
    num_threads >>= 1;
  }
  switch (num_threads) {
    case 1024:
      kernel_norm_forward<Dtype, 1024><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 512:
      kernel_norm_forward<Dtype, 512><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 256:
      kernel_norm_forward<Dtype, 256><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 128:
      kernel_norm_forward<Dtype, 128><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 64:
      kernel_norm_forward<Dtype, 64><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 32:
      kernel_norm_forward<Dtype, 32><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 16:
      kernel_norm_forward<Dtype, 16><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 8:
      kernel_norm_forward<Dtype, 8><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 4:
      kernel_norm_forward<Dtype, 4><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 2:
      kernel_norm_forward<Dtype, 2><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
    case 1:
      kernel_norm_forward<Dtype, 1><<<N, num_threads, 0>>>(
        N, K, fudge_factor, row_buffer, normalize_variance);
      break;
  }
  CUDA_POST_KERNEL_CHECK;
}

template void caffe_gpu_normalize_patches_rows_forward(const int K, const int N, const float fudge_factor,
  float* row_buffer, const bool normalize_variance);
template void caffe_gpu_normalize_patches_rows_forward(const int K, const int N, const double fudge_factor,
  double* row_buffer, const bool normalize_variance);


template <typename Dtype, int BLOCK_SIZE>
__global__ void kernel_norm_backward(const int patches, const int dim, const Dtype fudge,
    const Dtype* patches_rows, const Dtype* patches_normalized_rows, Dtype* patches_rows_diff) {
  __shared__ Dtype sdata[2*BLOCK_SIZE];
  const int bx = blockIdx.x;
  const int dx = BLOCK_SIZE;
  const int tid = threadIdx.x;

  // Calculate the mean and variance in parallel to the gradient
  // This works because we only need the variance at the end of the calculation.

  // Gradient w.r.t. norm by variance
  sdata[tid] = 0;
  sdata[BLOCK_SIZE + tid] = 0;
  for (int i = tid; i < dim; i += dx) {
    sdata[tid] += patches_normalized_rows[bx * dim + i] * patches_rows_diff[bx * dim + i];
    sdata[BLOCK_SIZE + tid] += patches_rows[bx * dim + i];
  }
  __syncthreads();
#pragma unroll
  for (unsigned int s=BLOCK_SIZE/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
      sdata[BLOCK_SIZE + tid] += sdata[BLOCK_SIZE + tid + s];
    }
    __syncthreads();
  }
  const Dtype x_times_diff_mean = sdata[0] / max(dim - 1, 1);
  const Dtype mean = sdata[BLOCK_SIZE] / dim;

  for (int i = tid; i < dim; i += dx) {
    const Dtype x = patches_normalized_rows[bx * dim + i];
    const Dtype x_diff = patches_rows_diff[bx * dim + i];
    patches_rows_diff[bx * dim + i] = x_diff - x * x_times_diff_mean;
  }

  // Gradient w.r.t. subtracting mean
  sdata[tid] = 0;
  sdata[BLOCK_SIZE + tid] = 0;
  for (int i = tid; i < dim; i += dx) {
    sdata[tid] += patches_rows_diff[bx * dim + i];
    const Dtype x = patches_rows[bx * dim + i] - mean;
    sdata[BLOCK_SIZE + tid] += x * x;
  }
  __syncthreads();
#pragma unroll
  for (unsigned int s=BLOCK_SIZE/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
      sdata[BLOCK_SIZE + tid] += sdata[BLOCK_SIZE + tid + s];
    }
    __syncthreads();
  }
  const Dtype x_diff_mean = sdata[0] / dim;
  const Dtype variance = sdata[BLOCK_SIZE] / max(dim - 1, 1);

  for (int i = tid; i < dim; i += dx) {
    const Dtype x_diff = patches_rows_diff[bx * dim + i];
    patches_rows_diff[bx * dim + i] = (x_diff - x_diff_mean) / sqrt(variance + fudge);
  }
}

template <typename Dtype, int BLOCK_SIZE>
__global__ void kernel_norm_backward_no_variance(const int patches, const int dim, const Dtype fudge,
    const Dtype* patches_rows, const Dtype* patches_normalized_rows, Dtype* patches_rows_diff) {
  __shared__ Dtype sdata[BLOCK_SIZE];
  const int bx = blockIdx.x;
  const int dx = BLOCK_SIZE;
  const int tid = threadIdx.x;

  // Gradient w.r.t. subtracting mean
  sdata[tid] = 0;
  for (int i = tid; i < dim; i += dx) {
    sdata[tid] += patches_rows_diff[bx * dim + i];
  }
  __syncthreads();
#pragma unroll
  for (unsigned int s=BLOCK_SIZE/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  const Dtype x_diff_mean = sdata[0] / dim;

  for (int i = tid; i < dim; i += dx) {
    const Dtype x_diff = patches_rows_diff[bx * dim + i];
    patches_rows_diff[bx * dim + i] = x_diff - x_diff_mean;
  }
}

template <typename Dtype>
void caffe_gpu_normalize_patches_rows_backward(const int K, const int N, const Dtype fudge_factor,
    const Dtype* row_buffer, const Dtype* normalized_row_buffer,
    Dtype* row_buffer_diff, const bool normalize_variance) {

//cudaStream_t cuStream = Caffe::cuda_stream(stream);
  int num_threads = min(CAFFE_CUDA_NUM_THREADS, 256);
  while (2*num_threads > K) {
    num_threads >>= 1;
  }
  if (normalize_variance) {
    switch (num_threads) {
      case 1024:
        kernel_norm_backward<Dtype, 1024><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 512:
        kernel_norm_backward<Dtype, 512><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 256:
        kernel_norm_backward<Dtype, 256><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 128:
        kernel_norm_backward<Dtype, 128><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 64:
        kernel_norm_backward<Dtype, 64><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 32:
        kernel_norm_backward<Dtype, 32><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 16:
        kernel_norm_backward<Dtype, 16><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 8:
        kernel_norm_backward<Dtype, 8><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 4:
        kernel_norm_backward<Dtype, 4><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 2:
        kernel_norm_backward<Dtype, 2><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 1:
        kernel_norm_backward<Dtype, 1><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
    }
  } else {
    switch (num_threads) {
      case 1024:
        kernel_norm_backward_no_variance<Dtype, 1024><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 512:
        kernel_norm_backward_no_variance<Dtype, 512><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 256:
        kernel_norm_backward_no_variance<Dtype, 256><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 128:
        kernel_norm_backward_no_variance<Dtype, 128><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 64:
        kernel_norm_backward_no_variance<Dtype, 64><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 32:
        kernel_norm_backward_no_variance<Dtype, 32><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 16:
        kernel_norm_backward_no_variance<Dtype, 16><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 8:
        kernel_norm_backward_no_variance<Dtype, 8><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 4:
        kernel_norm_backward_no_variance<Dtype, 4><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 2:
        kernel_norm_backward_no_variance<Dtype, 2><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
      case 1:
        kernel_norm_backward_no_variance<Dtype, 1><<<N, num_threads, 0>>>(
          N, K, fudge_factor, row_buffer, normalized_row_buffer, row_buffer_diff);
        break;
    }
  }

  CUDA_POST_KERNEL_CHECK;
}

template void caffe_gpu_normalize_patches_rows_backward(const int K, const int N, const float fudge_factor,
    const float* row_buffer, const float* normalized_row_buffer,
    float* row_buffer_diff, const bool normalize_variance);
template void caffe_gpu_normalize_patches_rows_backward(const int K, const int N, const double fudge_factor,
    const double* row_buffer, const double* normalized_row_buffer,
    double* row_buffer_diff, const bool normalize_variance);

template<typename Dtype>
__global__ void logspace_l2_kernel(const int N, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, N) {
    out[index] += weight_decay * Dtype(2) * exp(Dtype(2) * in[index]);
  }
}

template <typename Dtype>
void caffe_gpu_logspace_l2(const int N, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  logspace_l2_kernel<Dtype>
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, weight_decay, in, out);
}
template void caffe_gpu_logspace_l2(const int N, const float weight_decay, const float* in, float* out);
template void caffe_gpu_logspace_l2(const int N, const double weight_decay, const double* in, double* out);


template<typename Dtype>
__global__ void logspace_l2_smoothing_kernel(const int N, const int dim, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, N) {
    const int i = index % dim;
    const int i_minus = (index - 1) * (i > 0) + index * (i == 0);
    const int i_plus = (index + 1) * (i < dim - 1) + index * (i == dim - 1);
    out[index] += weight_decay * Dtype(2) * (
                  Dtype(2) * __expf(Dtype(2) * in[index])
                  - __expf(in[i_minus] + in[index])
                  - __expf(in[i_plus] + in[index]));
  }
}

template <typename Dtype>
void caffe_gpu_logspace_l2_smoothing(const int N, const int dim, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  logspace_l2_smoothing_kernel<Dtype>
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dim, weight_decay, in, out);
}
template void caffe_gpu_logspace_l2_smoothing(const int N, const int dim, const float weight_decay, const float* in, float* out);
template void caffe_gpu_logspace_l2_smoothing(const int N, const int dim, const double weight_decay, const double* in, double* out);

template<typename Dtype>
__global__ void l2_smoothing_kernel(const int N, const int dim, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, N) {
    const int i = index % dim;
    const int i_minus = (index - 1) * (i > 0) + index * (i == 0);
    const int i_plus = (index + 1) * (i < dim - 1) + index * (i == dim - 1);
    out[index] += weight_decay * Dtype(2) * (Dtype(2) * in[index] - in[i_minus] - in[i_plus]);
  }
}

template <typename Dtype>
void caffe_gpu_l2_smoothing(const int N, const int dim, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  l2_smoothing_kernel<Dtype>
    <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
    (N, dim, weight_decay, in, out);
}
template void caffe_gpu_l2_smoothing(const int N, const int dim, const float weight_decay, const float* in, float* out);
template void caffe_gpu_l2_smoothing(const int N, const int dim, const double weight_decay, const double* in, double* out);

template <typename Dtype, int BLOCK_SIZE>
__global__ void kernel_maximum_entropy_regularization(const int patches, const int dim, const Dtype* in, Dtype* out) {
  __shared__ Dtype sdata[BLOCK_SIZE];
  const int bx = blockIdx.x;
  const int dx = BLOCK_SIZE;
  const int tid = threadIdx.x;
  // Strided sum of the elements of the patch loaded to shared memory
  sdata[tid] = 0;
  for (int i = tid, pos = bx * dim + tid; i < dim; i += dx, pos += dx) {
    const Dtype x = in[pos];
    sdata[tid] += __expf(x) * (Dtype(1) + x);
  }
  __syncthreads();
  // Reduction sum
#pragma unroll
  for (unsigned int s=BLOCK_SIZE/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  const Dtype sum = sdata[0];

  // Normalize patch with the calculated mean and variance
  for (int i = tid; i < dim; i += dx) {
    const Dtype x = in[bx * dim + i];
    out[bx * dim + i] = __expf(x) * ((Dtype(1) + x) - sum);
  }
}

template <typename Dtype>
void caffe_gpu_maximum_entropy_regularization(const int N, const int K, const Dtype* in, Dtype* out) {
  int num_threads = min(CAFFE_CUDA_NUM_THREADS, 256);
  //cudaStream_t cuStream = Caffe::cuda_stream(stream);
  while (2*num_threads > K) {
    num_threads >>= 1;
  }
  switch (num_threads) {
    case 1024:
      kernel_maximum_entropy_regularization<Dtype, 1024><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 512:
      kernel_maximum_entropy_regularization<Dtype, 512><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 256:
      kernel_maximum_entropy_regularization<Dtype, 256><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 128:
      kernel_maximum_entropy_regularization<Dtype, 128><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 64:
      kernel_maximum_entropy_regularization<Dtype, 64><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 32:
      kernel_maximum_entropy_regularization<Dtype, 32><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 16:
      kernel_maximum_entropy_regularization<Dtype, 16><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 8:
      kernel_maximum_entropy_regularization<Dtype, 8><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 4:
      kernel_maximum_entropy_regularization<Dtype, 4><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 2:
      kernel_maximum_entropy_regularization<Dtype, 2><<<N, num_threads, 0>>>(N, K, in, out);
      break;
    case 1:
      kernel_maximum_entropy_regularization<Dtype, 1><<<N, num_threads, 0>>>(N, K, in, out);
      break;
  }
  CUDA_POST_KERNEL_CHECK;
}

template void caffe_gpu_maximum_entropy_regularization(const int N, const int K, const float* in, float* out);
template void caffe_gpu_maximum_entropy_regularization(const int N, const int K, const double* in, double* out);

}  // namespace caffe
