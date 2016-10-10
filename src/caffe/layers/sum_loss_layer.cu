#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sum_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SumLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();  
  Dtype loss;
  if (use_labeled_data_) {
    caffe_gpu_dot<Dtype>(count, bottom[0]->gpu_data(), sum_multiplier_.gpu_data(), &loss);
    caffe_gpu_set<Dtype>(1, -loss / count, top[0]->mutable_gpu_data());    
  } else {
    const Dtype* label = bottom[1]->cpu_data();
    Dtype loss = 0;
    Dtype unlabeled_count = 0;
    for (int i = 0; i < count; ++i) {
      const int current_label = static_cast<int>(label[i]);
      if (current_label >= 0) continue;
      unlabeled_count++;
      loss -= bottom[0]->cpu_data()[i];
    }
    top[0]->mutable_cpu_data()[0] = unlabeled_count > 0 ? loss / unlabeled_count : 0;
   }

}

template <typename Dtype>
void SumLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(count, Dtype(-1), bottom_diff);
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / count, bottom_diff);
    if (!use_labeled_data_) {
      bottom_diff = bottom[0]->mutable_cpu_diff();
      const Dtype* label = bottom[1]->cpu_data();
      for (int i = 0; i < count; ++i) {
        const int current_label = static_cast<int>(label[i]);
        if (current_label < 0) continue;
        bottom_diff[i] = 0;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SumLossLayer);


}  // namespace caffe
