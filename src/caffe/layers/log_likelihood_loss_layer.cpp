#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/log_likelihood_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void LogLikelihoodLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  use_labeled_data_ = this->layer_param_.log_likelihood_loss_param().use_labeled_data();
  maximize_likelihood_ = this->layer_param_.log_likelihood_loss_param().maximize_likelihood();
  filter_selected_labels_ = this->layer_param_.log_likelihood_loss_param().filter_selected_labels();
  if (filter_selected_labels_) {
    CHECK(this->layer_param_.log_likelihood_loss_param().has_selected_label());
    selected_label_ = this->layer_param_.log_likelihood_loss_param().selected_label();
    CHECK(use_labeled_data_ || selected_label_ < 0);
  }
}

template <typename Dtype>
void LogLikelihoodLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void LogLikelihoodLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = NULL;
  if (bottom.size() > 1) {
    label = bottom[1]->cpu_data();  
  } else {
    CHECK(!filter_selected_labels_);
  }
  
  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  const int dim = count / num;

  int examples_count = 0;
  Dtype log_likelihood = 0;
  for (int i = 0; i < num; ++i) {
    if (label) {
      int current_label = static_cast<int>(label[i]);
      if (!use_labeled_data_ && current_label >= 0) {
        continue;
      }
      if (filter_selected_labels_ && current_label != selected_label_) {
        continue;
      }
    }
    examples_count++;
    Dtype m = bottom_data[i * dim];
    for (int j = 1; j < dim; ++j) {
      m = std::max(m, bottom_data[i * dim + j]);
    }
    Dtype sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += std::exp(bottom_data[i * dim + j] - m);
    }
    log_likelihood += std::log(sum / dim) + m;
  }
  if (examples_count > 0) {
    const Dtype modifier = maximize_likelihood_ ? Dtype(-1) : Dtype(1); 
    top[0]->mutable_cpu_data()[0] = modifier * log_likelihood / examples_count;  
  } else {
    top[0]->mutable_cpu_data()[0] = 0;
  }
  
}

template <typename Dtype>
void LogLikelihoodLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype modifier = maximize_likelihood_ ? Dtype(-1) : Dtype(1); 
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set<Dtype>(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype* label = NULL;
    if (bottom.size() > 1) {
      label = bottom[1]->cpu_data();  
    } else {
      CHECK(!filter_selected_labels_);
    }

    const int num = bottom[0]->num();
    const int count = bottom[0]->count();
    const int dim = count / num;
    int examples_count = 0;
    for (int i = 0; i < num; ++i) {
      if (label) {
        int current_label = static_cast<int>(label[i]);
        if (!use_labeled_data_ && current_label >= 0) {
          continue;
        }
        if (filter_selected_labels_ && current_label != selected_label_) {
          continue;
        }
      }
      examples_count++;

      Dtype m = bottom_data[i * dim];
      for (int j = 0; j < dim; ++j) {
        m = std::max(m, bottom_data[i * dim + j]);
      }
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += std::exp(bottom_data[i * dim + j] - m);
      }
      for (int j = 0; j < dim; ++j) {
        bottom_diff[i * dim + j] = modifier * std::exp(bottom_data[i * dim + j] - m) / sum;
      }
    }

    if (examples_count > 0) {
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(count, loss_weight / examples_count, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(LogLikelihoodLossLayer);
#endif

INSTANTIATE_CLASS(LogLikelihoodLossLayer);
REGISTER_LAYER_CLASS(LogLikelihoodLoss);
}  // namespace caffe
