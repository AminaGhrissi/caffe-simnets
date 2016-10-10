#include <cstring>
#include <vector>
#include <random>
//#include <cmath>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ggemm.cuh"
#include "caffe/test/test_caffe_main.hpp"

#define CUASSERT(expr) ASSERT_EQ((expr), cudaSuccess)

namespace caffe {

static constexpr int kRandomSeed = 1234;

template<typename Dtype> __device__ __host__ __forceinline__
Dtype softmax(Dtype offset, Dtype data, Dtype max, uint8_t nothing) {
#ifdef __CUDA_ARCH__
  return exp(data + offset - max);
#else
  return std::exp(data + offset - max);
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype softmax_activation(Dtype max, Dtype input, uint8_t nothing) {
#ifdef __CUDA_ARCH__
  return log(input) + max;
#else
  return std::log(input) + max;
#endif
}

TEST(GGEMMTest, Test2Ops) {
  typedef float Dtype;
  size_t M = 4, N = 8, K = 3;
  int num_regions = 9;

  std::vector<Dtype> host_a(M*K*num_regions), host_b(K*N*num_regions), host_c (M*N, 0), host_c_res(M*N, 0);
  Dtype *d_a, *d_b, *d_c;

  CUASSERT(cudaMalloc(&d_a, M*K*num_regions*sizeof(Dtype)));
  CUASSERT(cudaMalloc(&d_b, K*N*num_regions*sizeof(Dtype)));
  CUASSERT(cudaMalloc(&d_c, M*N*sizeof(Dtype)));

  CUASSERT(cudaMemset(d_c, 0, M*N*sizeof(Dtype)));

  std::mt19937 gen (kRandomSeed);
  std::uniform_real_distribution<Dtype> rd(-1, 1);
  for (int i = 0; i < M * K*num_regions; ++i) {
    host_a[i] = rd(gen);
  }
  for (int i = 0; i < N * K*num_regions; ++i) {
    host_b[i] = rd(gen);
  }

  CUASSERT(cudaMemcpy(d_a, &host_a[0], M * K * num_regions * sizeof(Dtype), cudaMemcpyHostToDevice));
  CUASSERT(cudaMemcpy(d_b, &host_b[0], N * K * num_regions * sizeof(Dtype), cudaMemcpyHostToDevice));

  ggemm_2ops_gpu
    <Dtype, Dtype, Dtype, uint8_t,
     ggemm_add<Dtype, uint8_t>, ggemm_max<Dtype>,
     softmax<Dtype>, ggemm_add<Dtype>, true,
     softmax_activation<Dtype>, true,
     true, true, false>
    (M, N, K, d_a, d_b, d_c,
     -INFINITY, -INFINITY, -INFINITY, 0, 0, num_regions);
  CUASSERT(cudaMemcpy(&host_c_res[0], d_c, M * N * sizeof(Dtype), cudaMemcpyDeviceToHost));
  ggemm_2ops_cpu
    <Dtype, Dtype, Dtype, uint8_t,
     ggemm_add<Dtype, uint8_t>, ggemm_max<Dtype>,
     softmax<Dtype>, ggemm_add<Dtype>, true,
     softmax_activation<Dtype>, true,
     true, true, false>
    (M, N, K, &host_a[0], &host_b[0], &host_c[0],
    -INFINITY, 0, 0, num_regions);

  for (int i = 0; i < M*N; ++i) {
    EXPECT_NEAR(host_c_res[i], host_c[i], 1e-4);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

}  // namespace caffe
