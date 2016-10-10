#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/translation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TranslationForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int shift_x,
    const int shift_y, const Dtype out_of_bounds_value, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int src_h = h - shift_y;
    const int src_w = w - shift_x;
    if (src_h >= 0 && src_h < height && src_w >= 0 && src_w < width) {
      top_data[index] = bottom_data[((n * channels + c) * height + src_h) * width + src_w];
    } else {
      top_data[index] = out_of_bounds_value;
    }
  }
}

template <typename Dtype>
void TranslationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (random_) {
    if (this->phase_ == TRAIN) {
      caffe_rng_uniform_int(1, -shift_x_, shift_x_, &last_shift_x_);
      caffe_rng_uniform_int(1, -shift_y_, shift_y_, &last_shift_y_);
    } else {
      last_shift_x_ = 0;
      last_shift_y_ = 0;
    }
  } else {
    last_shift_x_ = shift_x_;
    last_shift_y_ = shift_y_;
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int height = bottom[i]->height();
    const int width = bottom[i]->width();
    const int count = bottom[i]->count();
    TranslationForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[i]->num(), bottom[i]->channels(),
        height, width, last_shift_x_, last_shift_y_, out_of_bounds_value_, top_data);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
__global__ void TranslationBackward(const int nthreads,
    const Dtype* const top_diff, const int num, const int channels,
    const int height, const int width, const int shift_x,
    const int shift_y, const Dtype out_of_bounds_value, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int src_h = h - shift_y;
    const int src_w = w - shift_x;
    if (src_h >= 0 && src_h < height && src_w >= 0 && src_w < width) {
       bottom_diff[((n * channels + c) * height + src_h) * width + src_w] = top_diff[index];
    }
  }
}

template <typename Dtype>
void TranslationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (!propagate_down[i]) {
      continue;
    }
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom_diff);
    const Dtype* top_diff = top[i]->gpu_diff();
    const int height = bottom[i]->height();
    const int width = bottom[i]->width();
    const int count = bottom[i]->count();
    TranslationBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom[i]->num(), bottom[i]->channels(),
        height, width, last_shift_x_, last_shift_y_, out_of_bounds_value_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(TranslationLayer);


}  // namespace caffe
