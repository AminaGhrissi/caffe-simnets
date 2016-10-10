#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/noise_layer.hpp"

namespace caffe {

template <typename Dtype, bool OFFSET_MEAN>
__global__ void NoiseForward(const int n, const Dtype* in,
    const Dtype* mul, const Dtype* add,
    Dtype* out, const Dtype scale = 1, const Dtype shift = 0) {
  CUDA_KERNEL_LOOP(index, n) {
    if (OFFSET_MEAN) {
      out[index] = scale * mul[index] * in[index] + add[index] + shift;
    } else {
      out[index] = mul[index] * in[index] + add[index];
    }
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (this->phase_ == TRAIN || force_random_) {
    add_noise_filler_->Fill_gpu(&add_noise_vec_);
    mul_noise_filler_->Fill_gpu(&mul_noise_vec_);
    if (!offset_mul_mean_ && !offset_add_mean_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      NoiseForward<Dtype, false><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, mul_noise_vec_.gpu_data(), add_noise_vec_.gpu_data(), top_data);
    } else {
      const Dtype scale = offset_mul_mean_ ? Dtype(1.0 / mul_noise_mean_) : Dtype(1);
      const Dtype shift = offset_add_mean_ ? -add_noise_mean_ : Dtype(0);
      // NOLINT_NEXT_LINE(whitespace/operators)
      NoiseForward<Dtype, true><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, mul_noise_vec_.gpu_data(), add_noise_vec_.gpu_data(), top_data, scale, shift);
    }
    CUDA_POST_KERNEL_CHECK;
  } else if (!offset_mul_mean_ && mul_noise_mean_ != Dtype(1) && (add_noise_mean_ == Dtype(0) || offset_add_mean_)) {
    caffe_gpu_axpby<Dtype>(count, mul_noise_mean_, bottom_data, Dtype(0), top_data);
  } else if (add_noise_mean_ != Dtype(0) && !offset_add_mean_) {
    caffe_gpu_set<Dtype>(count, add_noise_mean_, top_data);
    const Dtype scale = offset_mul_mean_ ? Dtype(1) : mul_noise_mean_;
    caffe_gpu_axpy<Dtype>(count, scale, bottom_data, top_data);
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype scale = offset_mul_mean_ ? Dtype(1.0 / mul_noise_mean_) : Dtype(1);
    if (this->phase_ == TRAIN || force_random_) {
      caffe_gpu_mul(count, top_diff, mul_noise_vec_.gpu_data(), bottom_diff);
      caffe_gpu_scal<Dtype>(count, scale, bottom_diff);
    } else if (mul_noise_mean_ != Dtype(1) && !offset_mul_mean_) {
      caffe_gpu_axpby(count, mul_noise_mean_, top_diff, Dtype(0), bottom_diff);
    } else {
      caffe_copy(count, top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);


}  // namespace caffe
