#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/noise_layer.hpp"

namespace caffe {
  
template <typename Dtype>
void NoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(this->layer_param_.noise_param_size(), 1);
  force_random_ = this->layer_param_.noise_param(param_index_).force_random();
  add_noise_filler_.reset(GetFiller<Dtype>(this->layer_param_.noise_param(param_index_).add_noise_filler()));
  if (this->layer_param_.noise_param(param_index_).add_noise_filler().type() == "constant") {
    add_noise_mean_ = this->layer_param_.noise_param(param_index_).add_noise_filler().value();
  } else if (this->layer_param_.noise_param(param_index_).add_noise_filler().type() == "gaussian") {
    add_noise_mean_ = this->layer_param_.noise_param(param_index_).add_noise_filler().mean();
  } else if (this->layer_param_.noise_param(param_index_).add_noise_filler().type() == "uniform") {
    add_noise_mean_ = (this->layer_param_.noise_param(param_index_).add_noise_filler().min()
                        + this->layer_param_.noise_param(param_index_).add_noise_filler().max()) / Dtype(2);
  } else if (this->layer_param_.noise_param(param_index_).add_noise_filler().type() == "bernoulli") {
    add_noise_mean_ = this->layer_param_.noise_param(param_index_).add_noise_filler().non_zero_probability();
  } else {
    LOG(FATAL) << "Only constant, gaussian, and uniform fillers are supported by the noise layer.";
  }
  if (this->layer_param_.noise_param(param_index_).add_noise_filler().to_log()) {
    if (this->layer_param_.noise_param(param_index_).add_noise_filler().type() == "bernoulli") {
      add_noise_mean_ = (1 - add_noise_mean_) 
                      * this->layer_param_.noise_param(param_index_).add_noise_filler().fudge_factor();
    } else {
      add_noise_mean_ = std::log(add_noise_mean_
                      + this->layer_param_.noise_param(param_index_).add_noise_filler().fudge_factor());
    }
  }

  mul_noise_filler_.reset(GetFiller<Dtype>(this->layer_param_.noise_param(param_index_).mul_noise_filler()));
  if (this->layer_param_.noise_param(param_index_).mul_noise_filler().type() == "constant") {
    mul_noise_mean_ = this->layer_param_.noise_param(param_index_).mul_noise_filler().value();
  } else if (this->layer_param_.noise_param(param_index_).mul_noise_filler().type() == "gaussian") {
    mul_noise_mean_ = this->layer_param_.noise_param(param_index_).mul_noise_filler().mean();
  } else if (this->layer_param_.noise_param(param_index_).mul_noise_filler().type() == "uniform") {
    mul_noise_mean_ = (this->layer_param_.noise_param(param_index_).mul_noise_filler().min()
                        + this->layer_param_.noise_param(param_index_).mul_noise_filler().max()) / Dtype(2);
  } else if (this->layer_param_.noise_param(param_index_).mul_noise_filler().type() == "bernoulli") {
    mul_noise_mean_ = this->layer_param_.noise_param(param_index_).mul_noise_filler().non_zero_probability();
  } else {
    LOG(FATAL) << "Only constant, gaussian, and uniform fillers are supported by the noise layer.";
  }
  if (this->layer_param_.noise_param(param_index_).mul_noise_filler().to_log()) {
    if (this->layer_param_.noise_param(param_index_).mul_noise_filler().type() == "bernoulli") {
      mul_noise_mean_ = (1 - mul_noise_mean_) 
                      * this->layer_param_.noise_param(param_index_).mul_noise_filler().fudge_factor();
    } else {
      mul_noise_mean_ = std::log(mul_noise_mean_
                      + this->layer_param_.noise_param(param_index_).mul_noise_filler().fudge_factor());
    }
  }

  offset_add_mean_ = this->layer_param_.noise_param(param_index_).offset_add_mean();
  offset_mul_mean_ = this->layer_param_.noise_param(param_index_).offset_mul_mean();
}

template <typename Dtype>
void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  add_noise_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mul_noise_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  if (this->phase_ == TRAIN || force_random_) {
    add_noise_filler_->Fill(&add_noise_vec_);
    mul_noise_filler_->Fill(&mul_noise_vec_);
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
    const Dtype scale = offset_mul_mean_ ? Dtype(1.0 / mul_noise_mean_) : Dtype(1);
    for (int i = 0; i < mul_noise_vec_.count(); ++i) {
      top_data[i] *= mul_noise_vec_.cpu_data()[i] * scale;
    }
    caffe_axpy<Dtype>(add_noise_vec_.count(), Dtype(1.), add_noise_vec_.cpu_data(), top_data);
    if (offset_add_mean_) {
      caffe_add_scalar<Dtype>(add_noise_vec_.count(), -add_noise_mean_, top_data);
    }
  } else if (!offset_mul_mean_ && mul_noise_mean_ != Dtype(1) && (add_noise_mean_ == Dtype(0) || offset_add_mean_)) {
    caffe_cpu_axpby(bottom[0]->count(), mul_noise_mean_, bottom_data, Dtype(0.0), top_data);
  } else if (add_noise_mean_ != Dtype(0) && !offset_add_mean_) {
    caffe_set(bottom[0]->count(), add_noise_mean_, top_data);
    const Dtype scale = offset_mul_mean_ ? Dtype(1) : mul_noise_mean_;
    caffe_axpy(bottom[0]->count(), scale, bottom_data, top_data);
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
    if (this->phase_ == TRAIN || force_random_) {
      const Dtype scale = offset_mul_mean_ ? Dtype(1.0 / mul_noise_mean_) : Dtype(1);
      for (int i = 0; i < mul_noise_vec_.count(); ++i) {
        bottom_diff[i] *= mul_noise_vec_.cpu_data()[i] * scale;
      }
    } else if (mul_noise_mean_ != Dtype(1) && !offset_mul_mean_) {
      caffe_scal(mul_noise_vec_.count(), mul_noise_mean_, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);
}  // namespace caffe
