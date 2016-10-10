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

#ifndef _SIM_LAYER_SHARED_H_
#define _SIM_LAYER_SHARED_H_
#include "caffe/util/ggemm_cpu.hpp"
#include "caffe/util/math_functions.hpp"
#define ISNAN(x) (x != x)
namespace caffe {
// Forward
template<typename Dtype> __forceinline__ __device__ __host__
Dtype sim_linear_forward(typename vec<Dtype>::vec2 a, Dtype b, uint8_t nothing) {
  return a.y * a.x * b;
}
template<typename Dtype> __forceinline__ __device__ __host__
Dtype sim_l1_forward(typename vec<Dtype>::vec2 a, Dtype b, uint8_t nothing) {
  return - a.y * std::abs(a.x - b);
}
template<typename Dtype, bool LOGSPACE_WEIGHT> __forceinline__ __device__ __host__
Dtype sim_l2_forward(typename vec<Dtype>::vec2 a, Dtype b, uint8_t nothing) {
  if (!LOGSPACE_WEIGHT) {
    return - a.y * (a.x - b) * (a.x - b);
  } else {
  #ifdef __CUDA_ARCH__
    return - exp(a.y) * (a.x - b) * (a.x - b);
  #else
    return - std::exp(a.y) * (a.x - b) * (a.x - b);
  #endif
  }
}
template<typename Dtype, bool LOGSPACE_WEIGHT, bool IGNORE_NAN> __forceinline__ __device__ __host__
Dtype sim_l2_normalized_forward(typename vec<Dtype>::vec2 a, Dtype b, Dtype fudge_factor) {
  if (IGNORE_NAN) {
    if (ISNAN(b)) {
      return Dtype(0);
    }
  }
  if (!LOGSPACE_WEIGHT) {
    if (!IGNORE_NAN) {
      return Dtype(0.5) * (  sim_l2_forward<Dtype, LOGSPACE_WEIGHT>(a, b, 0) 
                    // - std::log(2.0 * M_PI) // Moved to the main function by explicitly adding it
                    // Leaving it here causes a problem on the boundary cases
    #ifdef __CUDA_ARCH__
                    + __logf(a.y + fudge_factor));
    #else
                    + std::log(a.y + fudge_factor));
    #endif
    } else {
      return Dtype(0.5) * (  sim_l2_forward<Dtype, LOGSPACE_WEIGHT>(a, b, 0) 
    #ifdef __CUDA_ARCH__
                    - __logf(2.0 * M_PI) // when ignoring NANs we must added it here
                    + __logf(a.y + fudge_factor));
    #else
                    - std::log(2.0 * M_PI) // when ignoring NANs we must added it here
                    + std::log(a.y + fudge_factor));
    #endif
    }
  } else {
    if (!IGNORE_NAN) {
      return Dtype(0.5) * (sim_l2_forward<Dtype, LOGSPACE_WEIGHT>(a, b, 0) + a.y);
    } else {
      return Dtype(0.5) * (sim_l2_forward<Dtype, LOGSPACE_WEIGHT>(a, b, 0) + a.y
    #ifdef __CUDA_ARCH__
        - __logf(2.0 * M_PI));
    #else
        - std::log(2.0 * M_PI));
    #endif
    }
  }
}

// Backward weights
template<typename Dtype> __forceinline__ __device__ __host__
typename vec<Dtype>::vec2 sim_linear_backward_weights(Dtype err, Dtype x, typename vec<Dtype>::vec2 p, uint8_t nothing) {
  return make_vec2<Dtype>(err * x * p.y, err * x * p.x);
}
template<typename Dtype> __forceinline__ __device__ __host__
typename vec<Dtype>::vec2 sim_l1_backward_weights(Dtype err, Dtype x, typename vec<Dtype>::vec2 p, uint8_t nothing) {
  return make_vec2<Dtype>(err * p.y * caffe_sign<Dtype>(x - p.x), - err * std::abs(x - p.x));
}
template<typename Dtype, bool LOGSPACE_WEIGHT> __forceinline__ __device__ __host__
typename vec<Dtype>::vec2 sim_l2_backward_weights(Dtype err, Dtype x, typename vec<Dtype>::vec2 p, uint8_t nothing) {
  if (!LOGSPACE_WEIGHT) {
    return make_vec2<Dtype>(Dtype(2) * err * p.y * (x - p.x), - err * (x - p.x) * (x - p.x));
  } else {
  #ifdef __CUDA_ARCH__
    return make_vec2<Dtype>(Dtype(2) * err * exp(p.y) * (x - p.x), - err * exp(p.y) * (x - p.x) * (x - p.x));
  #else
    return make_vec2<Dtype>(Dtype(2) * err * std::exp(p.y) * (x - p.x), - err * std::exp(p.y) * (x - p.x) * (x - p.x));
  #endif
  }
  
}
template<typename Dtype, bool LOGSPACE_WEIGHT, bool IGNORE_NAN> __forceinline__ __device__ __host__
typename vec<Dtype>::vec2 sim_l2_normalized_backward_weights(Dtype err, Dtype x, typename vec<Dtype>::vec2 p, Dtype fudge_factor) {
  if (IGNORE_NAN) {
    if (ISNAN(x)) {
      return make_vec2<Dtype>(Dtype(0), Dtype(0));
    }
  }
  typename vec<Dtype>::vec2 v = sim_l2_backward_weights<Dtype, LOGSPACE_WEIGHT>(err, x, p, 0);
  v.x *= Dtype(0.5);
  if (!LOGSPACE_WEIGHT) {
  #ifdef __CUDA_ARCH__
    v.y = Dtype(0.5) * (v.y + __fdividef(err, p.y + fudge_factor));
  #else
    v.y = Dtype(0.5) * (v.y + err / (p.y + fudge_factor));
  #endif
  } else {
    v.y = Dtype(0.5) * (v.y + err);
  }
  return v;
}

// Backwaro bottom
template<typename Dtype> __forceinline__ __device__ __host__
Dtype sim_linear_backward_bottom(typename vec<Dtype>::vec2 p, Dtype err,
                                 Dtype data, uint8_t nothing) {
  return err * p.x * p.y;
}
template<typename Dtype> __forceinline__ __device__ __host__
Dtype sim_l1_backward_bottom(typename vec<Dtype>::vec2 p, Dtype err,
                             Dtype data, uint8_t nothing) {
  return err * p.y * caffe_sign<Dtype>(p.x - data);
}
template<typename Dtype, bool LOGSPACE_WEIGHT> __forceinline__ __device__ __host__
Dtype sim_l2_backward_bottom(typename vec<Dtype>::vec2 p, Dtype err,
                             Dtype data, uint8_t nothing) {
  if (!LOGSPACE_WEIGHT) {
    return Dtype(2) * err * p.y * (p.x - data);
  } else {
    #ifdef __CUDA_ARCH__
      return Dtype(2) * err * exp(p.y) * (p.x - data);
    #else
      return Dtype(2) * err * std::exp(p.y) * (p.x - data);
    #endif
  }
  
}
template<typename Dtype, bool LOGSPACE_WEIGHT, bool IGNORE_NAN> __forceinline__ __device__ __host__
Dtype sim_l2_normalized_backward_bottom(typename vec<Dtype>::vec2 p, Dtype err,
                                        Dtype data, Dtype fudge_factor) {
  if (IGNORE_NAN) {
    if (ISNAN(data)) {
      return Dtype(0);
    }
  }
  return Dtype(0.5) * sim_l2_backward_bottom<Dtype, LOGSPACE_WEIGHT>(p, err, data, 0);
}
}
#endif
