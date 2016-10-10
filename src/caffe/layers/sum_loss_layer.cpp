#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sum_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SumLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  use_labeled_data_ = this->layer_param_.sum_loss_param().use_labeled_data();
}

template <typename Dtype>
void SumLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[0]->count());
  sum_multiplier_.ReshapeLike(*bottom[0]);
  caffe_set<Dtype>(bottom[0]->count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void SumLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  if (use_labeled_data_) {
    Dtype loss = -caffe_cpu_dot<Dtype>(count, bottom[0]->cpu_data(), sum_multiplier_.cpu_data());
    top[0]->mutable_cpu_data()[0] = loss / count;
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
void SumLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set<Dtype>(count, Dtype(-1), bottom_diff);
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / count, bottom_diff);
    if (!use_labeled_data_) {
      const Dtype* label = bottom[1]->cpu_data();
      for (int i = 0; i < count; ++i) {
        const int current_label = static_cast<int>(label[i]);
        if (current_label < 0) continue;
        bottom_diff[i] = 0;
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SumLossLayer);
#endif

INSTANTIATE_CLASS(SumLossLayer);
REGISTER_LAYER_CLASS(SumLoss);
}  // namespace caffe
