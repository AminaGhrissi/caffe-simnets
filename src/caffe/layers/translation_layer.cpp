#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/translation_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void TranslationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TranslationParameter translation_param = this->layer_param_.translation_param();
  shift_x_ = translation_param.shift_x();
  shift_y_ = translation_param.shift_y();
  out_of_bounds_value_ = translation_param.out_of_bounds_value();
  random_ = translation_param.random();
  if (random_) {
    CHECK_GE(shift_x_, 0) << "shift_x must be non-negative when random = true";
    CHECK_GE(shift_y_, 0) << "shift_y must be non-negative when random = true";
  }
}

template <typename Dtype>
void TranslationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(4, bottom[i]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
    top[i]->ReshapeLike(*bottom[i]);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void TranslationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    const int height = bottom[i]->height();
    const int width = bottom[i]->width();
    for (int n = 0; n < bottom[i]->num(); ++n) {
      for (int c = 0; c < bottom[i]->channels(); ++c) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            const int src_h = h - last_shift_y_;
            const int src_w = w - last_shift_x_;
            if (src_h >= 0 && src_h < height && src_w >= 0 && src_w < width) {
              top_data[h * width + w] = bottom_data[src_h * width + src_w];
            } else {
              top_data[h * width + w] = out_of_bounds_value_;
            }
          }
        }
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
  }
}

template <typename Dtype>
void TranslationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (!propagate_down[i]) {
      continue;
    }
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    caffe_set(bottom[i]->count(), Dtype(0), bottom_diff);
    const Dtype* top_diff = top[i]->cpu_diff();
    const int height = bottom[i]->height();
    const int width = bottom[i]->width();
    for (int n = 0; n < bottom[i]->num(); ++n) {
      for (int c = 0; c < bottom[i]->channels(); ++c) {
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            const int src_h = h - last_shift_y_;
            const int src_w = w - last_shift_x_;
            if (src_h >= 0 && src_h < height && src_w >= 0 && src_w < width) {
              bottom_diff[src_h * width + src_w] = top_diff[h * width + w];
            }
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(TranslationLayer);
#endif

INSTANTIATE_CLASS(TranslationLayer);
REGISTER_LAYER_CLASS(Translation);

}  // namespace caffe
