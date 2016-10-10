#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
float caffe_cpu_nrm2(const int N, const float* X) {
  return cblas_snrm2(N, X, 1);
}

template <>
double caffe_cpu_nrm2(const int N, const double* X) {
  return cblas_dnrm2(N, X, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y, const float fudge_factor) {
  if (fudge_factor == 0) {
    vsLn(n, a, y);
  } else {
    CHECK_GT(fudge_factor, 0) << "The fudge factor must be positive!";
    for (int i = 0; i < n; ++i) {
      y[i] = std::log(a[i] + fudge_factor);
    }
  }
}

template <>
void caffe_log<double>(const int n, const double* a, double* y, const double fudge_factor) {
  if (fudge_factor == 0) {
    vdLn(n, a, y);
  } else {
    CHECK_GT(fudge_factor, 0) << "The fudge factor must be positive!";
    for (int i = 0; i < n; ++i) {
      y[i] = std::log(a[i] + fudge_factor);
    }
  }
  
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform_int(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_int<Dtype> random_distribution(a, b);
  boost::variate_generator<caffe::rng_t*, boost::uniform_int<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform_int<int>(const int n, const int a, const int b,
                              int* r);
template
void caffe_rng_uniform_int<unsigned int>(const int n, const unsigned int a, const unsigned int b,
                               unsigned int* r);


template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_gamma(const int n, const Dtype alpha, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  boost::gamma_distribution<Dtype> gamma_distribution(alpha);
  boost::variate_generator<caffe::rng_t*, boost::gamma_distribution<Dtype> >
      variate_generator(caffe_rng(), gamma_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gamma<float>(const int n, const float alpha, float* r);

template
void caffe_rng_gamma<double>(const int n, const double alpha, double* r);

template <typename Dtype, typename Otype>
void caffe_rng_bernoulli(const int n, const Dtype p, Otype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<Otype>(variate_generator());
  }
}

template void caffe_rng_bernoulli<float, int>(const int n, const float p, int* r);
template void caffe_rng_bernoulli<float, unsigned int>(const int n, const float p, unsigned int* r);
template void caffe_rng_bernoulli<float, float>(const int n, const float p, float* r);
template void caffe_rng_bernoulli<float, double>(const int n, const float p, double* r);
template void caffe_rng_bernoulli<double, int>(const int n, const double p, int* r);
template void caffe_rng_bernoulli<double, unsigned int>(const int n, const double p, unsigned int* r);
template void caffe_rng_bernoulli<double, float>(const int n, const double p, float* r);
template void caffe_rng_bernoulli<double, double>(const int n, const double p, double* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <typename Dtype>
void caffe_cpu_clip_min(const int n, const Dtype* x, Dtype* y, const Dtype min_value) {
  if (isnan(min_value)) {
    caffe_copy(n, x, y);
  } else {
    for (int i = 0; i < n; ++i) {
      y[i] = std::max(x[i], min_value);
    }
  }
}

template void caffe_cpu_clip_min<double>(const int n, const double* x, double* y, 
  const double min_value);
template void caffe_cpu_clip_min<float>(const int n, const float* x, float* y,
  const float min_value);
template void caffe_cpu_clip_min<int>(const int n, const int* x, int* y, 
  const int min_value);
template void caffe_cpu_clip_min<unsigned int>(const int n, const unsigned int* x,
  unsigned int* y, const unsigned int min_value);

template <typename Dtype>
void caffe_cpu_clip_max(const int n, const Dtype* x, Dtype* y, const Dtype max_value) {
  if (isnan(max_value)) {
    caffe_copy(n, x, y);
  } else {
    for (int i = 0; i < n; ++i) {
      y[i] = std::min(x[i], max_value);
    }
  }
}

template void caffe_cpu_clip_max<double>(const int n, const double* x, double* y,
  const double max_value);
template void caffe_cpu_clip_max<float>(const int n, const float* x, float* y,
  const float max_value);
template void caffe_cpu_clip_max<int>(const int n, const int* x, int* y,
  const int max_value);
template void caffe_cpu_clip_max<unsigned int>(const int n, const unsigned int* x,
  unsigned int* y, const unsigned int max_value);

template <typename Dtype>
void caffe_cpu_dgmm(const CBLAS_SIDE mode, const int M, const int N,
    const Dtype* A, const Dtype* x, Dtype* C) {
  if (mode == CblasRight) {
    for (int i = 0; i < M; ++i) {
      caffe_mul(N, A + N*i, x, C + N*i);
    }
  } else {
    if (A == C) {
      for (int i = 0; i < M; ++i) {
        caffe_scal(N, x[i], C + N*i);
      }
    } else {
      for (int i = 0; i < M; ++i) {
        caffe_cpu_axpby(N, x[i], A + N*i, Dtype(0), C + N*i);
      }
    }
  }
}
template
void caffe_cpu_dgmm<float>(const CBLAS_SIDE mode, const int M, const int N,
    const float* A, const float* x, float* C);
template
void caffe_cpu_dgmm<double>(const CBLAS_SIDE mode, const int M, const int N,
    const double* A, const double* x, double* C);

template <typename Dtype>
Dtype caffe_ceiled_div(const Dtype a, const Dtype b) {
  CHECK_GE(a, 0); CHECK_GT(b, 0);
  return (a / b) + ((a % b) > 0);
}

template
int caffe_ceiled_div(const int a, const int b);
template
long caffe_ceiled_div(const long a, const long b);

template <typename Dtype>
void caffe_cpu_transpose(const int N, const int M, const Dtype * A, Dtype * B) {
  CHECK_NE(A, B); // TODO: add support for in-place transpose;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      B[j * N + i] = A[i * M + j];
    }
  }
}
template void caffe_cpu_transpose(const int N, const int M, const float * A, float * B);
template void caffe_cpu_transpose(const int N, const int M, const double * A, double * B);

template <class Dtype>
void caffe_rng_shuffle(vector<Dtype>& vec) {
  shuffle(vec.begin(), vec.end());
}
template <class Dtype>
void caffe_rng_shuffle(Dtype* begin, Dtype* end) {
  shuffle(begin, end);
}

template void caffe_rng_shuffle(vector<int>& vec);
template void caffe_rng_shuffle(vector<long>& vec);
template void caffe_rng_shuffle(vector<float>& vec);
template void caffe_rng_shuffle(vector<double>& vec);
template void caffe_rng_shuffle(int* begin, int* end);
template void caffe_rng_shuffle(long* begin, long* end);
template void caffe_rng_shuffle(float* begin, float* end);
template void caffe_rng_shuffle(double* begin, double* end);

template <typename Dtype>
void caffe_cpu_normalize_patches_rows_forward(const int K, const int N, const Dtype fudge_factor,
  Dtype* row_buffer, const bool normalize_variance) {
  for (int n = 0; n < N; ++n) {
    Dtype sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += row_buffer[n * K + k];
    }
    const Dtype mean = sum / K;
    for (int k = 0; k < K; ++k) {
      const Dtype x = row_buffer[n * K + k];
      row_buffer[n * K + k] = x - mean;
    }
    if (normalize_variance) {
      sum = 0;
      for (int k = 0; k < K; ++k) {
        const Dtype x_minus_mean = row_buffer[n * K + k];
        sum += x_minus_mean * x_minus_mean;
      }
      const Dtype variance = sum / std::max(K - 1, 1);
      for (int k = 0; k < K; ++k) {
        const Dtype x = row_buffer[n * K + k];
        row_buffer[n * K + k] = x / std::sqrt(variance + fudge_factor);
      }
    }
  }
}

template void caffe_cpu_normalize_patches_rows_forward(const int K, const int N, const float fudge_factor,
  float* row_buffer, const bool normalize_variance);
template void caffe_cpu_normalize_patches_rows_forward(const int K, const int N, const double fudge_factor,
  double* row_buffer, const bool normalize_variance);

template <typename Dtype>
void caffe_cpu_normalize_patches_rows_backward(const int K, const int N, const Dtype fudge_factor,
  const Dtype* row_buffer, const Dtype* normalized_row_buffer,
  Dtype* row_buffer_diff, const bool normalize_variance) {
  for (int n = 0; n < N; ++n) {
    // Calculate gradient w.r.t. variance.
    // Along the way calculate the mean of the original patches.
    if (normalize_variance) {
      Dtype sum_row = 0, x_times_diff_sum = 0;
      for (int k = 0; k < K; ++k) {
        sum_row += row_buffer[n * K + k];
        x_times_diff_sum += normalized_row_buffer[n * K + k] * row_buffer_diff[n * K + k];
      }
      const Dtype mean = sum_row / K;
      const Dtype x_times_diff_mean = x_times_diff_sum / std::max(K - 1, 1);

      sum_row = 0;
      for (int k = 0; k < K; ++k) {
        const Dtype x_minus_mean = row_buffer[n * K + k] - mean;
        sum_row += x_minus_mean * x_minus_mean;
      }
      const Dtype variance = sum_row / std::max(K - 1, 1);

      for (int k = 0; k < K; ++k) {
        const Dtype x = normalized_row_buffer[n * K + k];
        const Dtype x_diff = row_buffer_diff[n * K + k];
        row_buffer_diff[n * K + k] = (x_diff - x * x_times_diff_mean) / std::sqrt(variance + fudge_factor);
      }
    }

    Dtype x_diff_sum = 0;
    for (int k = 0; k < K; ++k) {
      x_diff_sum += row_buffer_diff[n * K + k];
    }
    const Dtype x_diff_mean = x_diff_sum / K;

    for (int k = 0; k < K; ++k) {
      const Dtype x = row_buffer_diff[n * K + k];
      row_buffer_diff[n * K + k] = (x - x_diff_mean);
    }
  }
}

template void caffe_cpu_normalize_patches_rows_backward(const int K, const int N, const float fudge_factor,
  const float* row_buffer, const float* normalized_row_buffer,
  float* row_buffer_diff, const bool normalize_variance);
template void caffe_cpu_normalize_patches_rows_backward(const int K, const int N, const double fudge_factor,
  const double* row_buffer, const double* normalized_row_buffer,
  double* row_buffer_diff, const bool normalize_variance);

template <typename Dtype>
void caffe_cpu_logspace_l2(const int N, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  for (int i = 0; i < N; ++i) {
    out[i] += weight_decay * Dtype(2) * std::exp(Dtype(2) * in[i]);
  }
}
template void caffe_cpu_logspace_l2(const int N, const float weight_decay, const float* in, float* out);
template void caffe_cpu_logspace_l2(const int N, const double weight_decay, const double* in, double* out);

template <typename Dtype>
void caffe_cpu_logspace_l2_smoothing(const int N, const int dim, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  for (int index = 0; index < N; ++index) {
    const int i = index % dim;
    const int i_minus = i > 0 ? (index - 1) : index;
    const int i_plus = (i < dim - 1) ? (index + 1) : index;
    out[index] += weight_decay * Dtype(2) * (
                  Dtype(2) * std::exp(Dtype(2) * in[index])
                  - std::exp(in[i_minus] + in[index])
                  - std::exp(in[i_plus] + in[index]));
  }
}
template void caffe_cpu_logspace_l2_smoothing(const int N, const int dim, const float weight_decay, const float* in, float* out);
template void caffe_cpu_logspace_l2_smoothing(const int N, const int dim, const double weight_decay, const double* in, double* out);

template <typename Dtype>
void caffe_cpu_l2_smoothing(const int N, const int dim, const Dtype weight_decay, const Dtype* in, Dtype* out) {
  for (int index = 0; index < N; ++index) {
    const int i = index % dim;
    const int i_minus = i > 0 ? (index - 1) : index;
    const int i_plus = (i < dim - 1) ? (index + 1) : index;
    out[index] += weight_decay * Dtype(2) * (Dtype(2) * in[index] - in[i_minus] - in[i_plus]);
  }
}
template void caffe_cpu_l2_smoothing(const int N, const int dim, const float weight_decay, const float* in, float* out);
template void caffe_cpu_l2_smoothing(const int N, const int dim, const double weight_decay, const double* in, double* out);

template <typename Dtype>
void caffe_cpu_maximum_entropy_regularization(const int N, const int K, const Dtype* in, Dtype* out) {
  for (int n = 0; n < N; ++n) {
    Dtype sum = 0;
    for (int k = 0; k < K; ++k) {
      const Dtype x = in[n * K + k];
      sum += (Dtype(1) + x) * exp(x);
    }
    for (int k = 0; k < K; ++k) {
      const Dtype x = in[n * K + k];
      out[n * K + k] = exp(x) * ((Dtype(1) + x) - sum);
    }
  }
}

template void caffe_cpu_maximum_entropy_regularization(const int N, const int K, const float* in, float* out);
template void caffe_cpu_maximum_entropy_regularization(const int N, const int K, const double* in, double* out);

}  // namespace caffe
