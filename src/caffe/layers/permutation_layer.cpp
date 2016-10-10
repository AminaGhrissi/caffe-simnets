#include <algorithm>
#include <cfloat>
#include <vector>
#include <sstream>
#include <iostream>

#include "caffe/layers/permutation_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;
using std::stringstream;

template <typename Dtype>
void PermutationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  type_ = this->layer_param_.permutation_param().type();

  if (type_ == PermutationParameter_PermutationType_GEN) {
    stringstream ss(this->layer_param_.permutation_param().permute_string());
    int i;
    while (ss >> i) {
          inv_permute_ltable_.push_back(i);
          permute_ltable_.push_back(-1); // to make table of same size

          if (ss.peek() == ',')
              ss.ignore();
    }
    // Fill lookup table
    for (int j = 0; j < inv_permute_ltable_.size(); ++j) {
      permute_ltable_[inv_permute_ltable_[j]] = j;
    }

    for (int j = 0; j < permute_ltable_.size(); ++j) {
      CHECK_GE(permute_ltable_[j],0) << "Illegal input string defining the permutation.";
    }


    for (int i = 0; i < bottom.size(); ++i) {
      CHECK_EQ(permute_ltable_.size(), (bottom[i]->width() * bottom[i]->height()) ) << "Permutation "
        << "must be defined for each spatial location (H*W total entries)";
    }

    for (int i = 0; i < bottom.size(); ++i) {
      CHECK_EQ(inv_permute_ltable_.size(), (bottom[i]->width() * bottom[i]->height()) ) << "Inverse permutation "
        << "must be defined for each spatial location (H*W total entries)";
    }


  }
  
}

template <typename Dtype>
void PermutationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(4, bottom[i]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
    top[i]->ReshapeLike(*bottom[i]);
  }
}

template <typename Dtype>
void PermutationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  switch (type_) {
  case PermutationParameter_PermutationType_QUAD2:
  {
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      const int height = bottom[i]->height();
      const int width = bottom[i]->width();
      for (int n = 0; n < bottom[i]->num(); ++n) {
        for (int c = 0; c < bottom[i]->channels(); ++c) {
          for (int h = 0; h < height; ++h) {
            const int src_h = ((h % 2) == 0) ? (h / 2) : ((height - h / 2) - 1);
            for (int w = 0; w < width; ++w) {
              const int src_w = ((w % 2) == 0) ? (w / 2) : ((width - w / 2) - 1);
              top_data[h * width + w] = bottom_data[src_h * width + src_w];
            }
          }
          bottom_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
        }
      }
    }
  }
    break;

  case PermutationParameter_PermutationType_GEN:
  {
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      const int height = bottom[i]->height();
      const int width = bottom[i]->width();
      for (int n = 0; n < bottom[i]->num(); ++n) {
        for (int c = 0; c < bottom[i]->channels(); ++c) {
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
              const int dst_h = permute_ltable_[h * width + w] / width;
              const int dst_w = permute_ltable_[h * width + w] % width;
              top_data[dst_h * width + dst_w] = bottom_data[h * width + w];
            }
          }
          bottom_data += bottom[0]->offset(0, 1);
          top_data += top[0]->offset(0, 1);
        }
      }
    }
  }
    break;

  default:
      LOG(FATAL) << "Unknown permutation type.";

  }

}

template <typename Dtype>
void PermutationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (!propagate_down[i]) {
      continue;
    }
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    caffe_set(bottom[i]->count(), Dtype(0), bottom_diff);
    const Dtype* top_diff = top[i]->cpu_diff();

    switch (type_) {
    case PermutationParameter_PermutationType_QUAD2:
    {
      
      const int height = bottom[i]->height();
      const int width = bottom[i]->width();
      for (int n = 0; n < bottom[i]->num(); ++n) {
        for (int c = 0; c < bottom[i]->channels(); ++c) {
          // Inverse of the permutation
          for (int h = 0; h < height; ++h) {
            const int src_h = (h * 2 < height) ? (h * 2) : ((height - 1) - (2 * h) % height);
            for (int w = 0; w < width; ++w) {
              const int src_w = (w * 2 < width) ? (w * 2) : ((width - 1) - (2 * w) % width);
              bottom_diff[h * width + w] = top_diff[src_h * width + src_w];
            }
          }
          bottom_diff += bottom[0]->offset(0, 1);
          top_diff += top[0]->offset(0, 1);
        }
      }
      break;
    }

    case PermutationParameter_PermutationType_GEN:
    {
      
      const int height = bottom[i]->height();
      const int width = bottom[i]->width();
      for (int n = 0; n < bottom[i]->num(); ++n) {
        for (int c = 0; c < bottom[i]->channels(); ++c) {
          // Inverse of the permutation
          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
              const int dst_h = inv_permute_ltable_[h * width + w] / width;
              const int dst_w = inv_permute_ltable_[h * width + w] % width;
              bottom_diff[dst_h * width + dst_w] = top_diff[h * width + w];
            }
          }
          bottom_diff += bottom[0]->offset(0, 1);
          top_diff += top[0]->offset(0, 1);
        }
      }
      break;
    }


    default:
      LOG(FATAL) << "Unknown permutation type.";
    }

  }
}


#ifdef CPU_ONLY
STUB_GPU(PermutationLayer);
#endif

INSTANTIATE_CLASS(PermutationLayer);
REGISTER_LAYER_CLASS(Permutation);

}  // namespace caffe
