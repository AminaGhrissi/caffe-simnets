#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/generalized_hinge_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void GeneralizedHingeLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  epsilon_ = this->layer_param_.generalized_hinge_loss_param().epsilon();
  margin_ = this->layer_param_.generalized_hinge_loss_param().margin();
  ignore_labeled_data_ = this->layer_param_.generalized_hinge_loss_param().ignore_labeled_data();
  if (ignore_labeled_data_) {
    CHECK_EQ(bottom.size(), 3);
  }
}

template <typename Dtype>
void GeneralizedHingeLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void GeneralizedHingeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* true_label = NULL;
  if (ignore_labeled_data_) {
    true_label = bottom[2]->cpu_data();
  }
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  int labeled_count = 0;
  for (int i = 0; i < num; ++i) {
    int current_label = static_cast<int>(label[i]);
    if (ignore_labeled_data_) {
      const int current_true_label = static_cast<int>(true_label[i]);
      if (current_true_label >= 0) {
        current_label = -1;
      }
    }
    if (current_label < 0) {
      caffe_set(dim, Dtype(0), bottom_diff + i * dim);
    } else {
      labeled_count++;
      caffe_add_scalar(dim, margin_ - bottom_data[i * dim + current_label], bottom_diff + i * dim);
      bottom_diff[i * dim + static_cast<int>(label[i])] -= margin_;        
    }
  }
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int current_label = static_cast<int>(label[i]);
    if (ignore_labeled_data_) {
      const int current_true_label = static_cast<int>(true_label[i]);
      if (current_true_label >= 0) {
        current_label = -1;
      }
    }
    if (current_label < 0) continue;
    Dtype m = bottom_diff[i * dim];
    for (int j = 0; j < dim; ++j) {
      m = (epsilon_ > 0) * std::max(m, bottom_diff[i * dim + j])
        + (epsilon_ < 0) * std::min(m, bottom_diff[i * dim + j]);
    }
    Dtype sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += std::exp(epsilon_ * (bottom_diff[i * dim + j] - m));
    }
    loss += (std::log(sum) / epsilon_) + m;
  }
  if (labeled_count > 0) {
    top[0]->mutable_cpu_data()[0] = loss / labeled_count;  
  } else {
    top[0]->mutable_cpu_data()[0] = 0;
  }
  
}

template <typename Dtype>
void GeneralizedHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    if (!this->layer_param_.generalized_hinge_loss_param().silently_ignore_label_backprop()) {
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to label inputs.";
    }
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    int labeled_count = 0;
    caffe_copy(count, bottom_data, bottom_diff);
    for (int i = 0; i < num; ++i) {
      const int current_label = static_cast<int>(label[i]);
      if (current_label < 0) {
        caffe_set(dim, Dtype(0), bottom_diff + i * dim);
      } else {
        labeled_count++;
        caffe_add_scalar(dim, margin_ - bottom_data[i * dim + current_label], bottom_diff + i * dim);
        bottom_diff[i * dim + static_cast<int>(label[i])] -= margin_;
      }
    }

    for (int i = 0; i < num; ++i) {
      const int current_label = static_cast<int>(label[i]);
      if (current_label < 0) continue;
      Dtype m = bottom_diff[i * dim];
      for (int j = 0; j < dim; ++j) {
        m = (epsilon_ > 0) * std::max(m, bottom_diff[i * dim + j])
          + (epsilon_ < 0) * std::min(m, bottom_diff[i * dim + j]);
      }
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += std::exp(epsilon_ * (bottom_diff[i * dim + j] - m));
      }
      for (int j = 0; j < dim; ++j) {
        bottom_diff[i * dim + j] = std::exp(epsilon_ * (bottom_diff[i * dim + j] - m)) / sum;
      }
      bottom_diff[i * dim + current_label] = (std::exp(-epsilon_ * m) / sum) - Dtype(1);
    }

    if (labeled_count > 0) {
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(count, loss_weight / labeled_count, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GeneralizedHingeLossLayer);
#endif

INSTANTIATE_CLASS(GeneralizedHingeLossLayer);
REGISTER_LAYER_CLASS(GeneralizedHingeLoss);
}  // namespace caffe
