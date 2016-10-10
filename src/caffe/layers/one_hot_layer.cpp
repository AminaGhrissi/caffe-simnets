#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/one_hot_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OneHotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  M_ = this->layer_param_.one_hot_param().input_dim();
  CHECK_GT(M_, 0) << "OneHotLayer input_dim must be positive.";
  to_log_ = this->layer_param_.one_hot_param().to_log();
}

template <typename Dtype>
void OneHotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int num = bottom[0]->shape(0);
  const int channels = bottom[0]->shape(1);
  N_ = bottom[0]->count() / num;
  CHECK_EQ(channels, 1) << " assumes the number of channels is 1";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = M_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->shape(0);
  const Dtype zero_value = to_log_ ? Dtype(-INFINITY) : Dtype(0);
  const Dtype one_value = to_log_ ? Dtype(0) : Dtype(1);
  caffe_set<Dtype>(top[0]->count(), zero_value, top_data);
  for (int n = 0; n < num; ++n) {
    for (int i = 0; i < N_; ++i) {
      const int index = static_cast<int>(std::round(bottom_data[n * N_ + i]));
      top_data[n * N_ * M_ + index * N_ + i] = one_value;
    }
  }
}

template <typename Dtype>
void OneHotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to OneHotLayer input.";
}

#ifdef CPU_ONLY
STUB_GPU(OneHotLayer);
#endif

INSTANTIATE_CLASS(OneHotLayer);
REGISTER_LAYER_CLASS(OneHot);

}  // namespace caffe
