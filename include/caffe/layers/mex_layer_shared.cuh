/** 
 *  Generalization of Matrix Multiplication (C = A * B).
 *  The original operation is defined by $ C_{ij} = \sum_{k = 1}^K A_{ik} * B_{kj} $.
 *  The generalized operation supports swapping the multiplication and addition opertaions above
 *  with a generation combine_function(a, b) and an accumilation_function(acc_c, c). For example,
 *  one can define the operation by $ C_{ij} = \max_{k = 1}^K (A_{ik} +  B_{kj}) $.
 *
 *  A farther generalization is allowing the combine_function to access the matching element from C,
 *  i.e., $ C_{ij} = \textrm{accumlate}_{k = 1}^K \textrm{combine}(A_{ik}, B_{kj}, C_{ij})$. This 
 *  allows one to implement parametrized softmax (i.e., log(sum(exp(a_i + b_i)))), or the backward
 *  steps of many convolutional-like network layers.
 */

#ifndef _MEX_LAYER_SHARED_H_
#define _MEX_LAYER_SHARED_H_
#include "caffe/util/ggemm_cpu.hpp"
#include "caffe/util/math_functions.hpp"
#include <thrust/reduce.h>
#define EXP_CUDA(x) (__expf(x))
#define LOG_CUDA(x) (__logf(x))
namespace caffe {

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_forward_exp(Dtype offset, Dtype data, Dtype max, typename vec<Dtype>::vec2 extra) {
#ifdef __CUDA_ARCH__
  return EXP_CUDA(extra.x * (data + offset - max) + extra.y);
  //return __fdividef(exp(data + offset - max), Dtype(K));
#else
  return std::exp(extra.x * (data + offset - max) + extra.y);
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_forward_out(Dtype in, typename vec<Dtype>::vec2 extra) {
#ifdef __CUDA_ARCH__
  return LOG_CUDA(in) / extra.x;
#else
  return std::log(in) / extra.x;
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_bottom_finite(Dtype offset, typename vec<Dtype>::vec2 top_data, Dtype data,
  typename vec<Dtype>::vec2 extra) {
#ifdef __CUDA_ARCH__
  return top_data.y * EXP_CUDA(extra.x * (data + offset - top_data.x) + extra.y);
  //return __fdividef(exp(data + offset - max), Dtype(K));
#else
  return top_data.y * std::exp(extra.x * (data + offset - top_data.x) + extra.y);
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_bottom_infinite(Dtype offset, typename vec<Dtype>::vec2 top_data, Dtype data, uint8_t nothing) {
  return top_data.y * ((data + offset) == top_data.x);
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_offsets_finite(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset,
  typename vec<Dtype>::vec2 extra) {
  return mex_backward_bottom_finite(offset, top_data, data, extra);
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_offsets_infinite(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset, uint8_t nothing) {
  return mex_backward_bottom_infinite(offset, top_data, data, nothing);
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_normalized_offsets_finite(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset,
  typename vec<Dtype>::vec2 extra) {
  const Dtype res = mex_backward_offsets_finite(top_data, data, offset, extra);
#ifdef __CUDA_ARCH__
  return res - top_data.y * EXP_CUDA(extra.x * offset + extra.y);
#else
  return res - top_data.y * std::exp(extra.x * offset + extra.y);
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_normalized_offsets_infinite(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset,
  uint8_t nothing) {
  const Dtype res = mex_backward_offsets_infinite(top_data, data, offset, nothing);
  return res - top_data.y * (offset == Dtype(0));
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_epsilon(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset,
  Dtype epsilon) {
  const Dtype x = data + offset;
#ifdef __CUDA_ARCH__
  return top_data.y * (x * EXP_CUDA(epsilon * (x - top_data.x)) - top_data.x);
#else
  return top_data.y * (x * std::exp(epsilon * (x - top_data.x)) - top_data.x);
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_epsilon_with_normalized_offsets(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset,
  Dtype epsilon) {
  const Dtype res = mex_backward_epsilon(top_data, data, offset, epsilon);
#ifdef __CUDA_ARCH__
  return res - top_data.y * offset * EXP_CUDA(epsilon * offset);
#else
  return res - top_data.y * offset * std::exp(epsilon * offset);
#endif
}

}
#endif
