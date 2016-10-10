#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void DirectionalAdamSolver<Dtype>::DirectionalAdamPreSolve() {
  // Add the extra history entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    this->temp_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    Blob<Dtype>* t2 = this->temp_[this->temp_.size() - 1].get();
    caffe_set<Dtype>(t2->count(), Dtype(1), t2->mutable_cpu_data());
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void directional_adam_update_moments_gpu(int N, const Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta2);
template <typename Dtype>
void directional_adam_update_gradients_gpu(int N, Dtype* g, const Dtype* m, const Dtype* v,
    Dtype eps_hat, Dtype corrected_local_rate);
#endif

template <typename Dtype>
void DirectionalAdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum();
  const Dtype beta2 = this->param_.momentum2();

  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_t = this->temp_[param_id].get();
  Blob<Dtype>* val_t2 = this->temp_[param_id + update_history_offset].get();

  const int t = this->iter_ + 1;
  const int N = net_params[param_id]->count();
  const int K = net_params[param_id]->shape(-1);
  const int M = N / K;
  const Dtype correction = std::sqrt(K) * std::sqrt(Dtype(1) - pow(beta2, t)) /
      (Dtype(1.) - pow(beta1, t));
  const Dtype eps_hat = this->param_.delta();

  switch (Caffe::mode()) {
    case Caffe::CPU: {
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_cpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->cpu_diff(), beta1,
        val_m->mutable_cpu_data());

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_mul(N,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->cpu_diff(),
    val_t->mutable_cpu_data());
    caffe_cpu_axpby(N, Dtype(1)-beta2,
        val_t->cpu_data(), beta2,
        val_v->mutable_cpu_data());

    // Calculate norm
    caffe_cpu_gemv(CblasNoTrans, M, K,
        Dtype(1), val_v->cpu_data(), val_t2->cpu_data(),
        Dtype(0), val_t2->mutable_cpu_diff());
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M, K, 1,
      Dtype(1), val_t2->cpu_diff(), val_t2->cpu_data(), Dtype(0),
      val_t->mutable_cpu_data());

    // update step
    caffe_powx(N,
        val_t->cpu_data(), Dtype(0.5),
        val_t->mutable_cpu_data());
    caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
    caffe_div(N,
        val_m->cpu_data(),
        val_t->cpu_data(),
        val_t->mutable_cpu_data());

    caffe_cpu_scale(N, local_rate*correction,
        val_t->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    directional_adam_update_moments_gpu(N, net_params[param_id]->gpu_diff(),
        val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), beta1, beta2);
    // Calculate norm
    caffe_cpu_gemv(CblasNoTrans, M, K,
        Dtype(1), val_v->cpu_data(), val_t2->cpu_data(),
        Dtype(0), val_t2->mutable_cpu_diff());
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M, K, 1,
      Dtype(1), val_t2->cpu_diff(), val_t2->cpu_data(), Dtype(0),
      val_t->mutable_cpu_data());
    directional_adam_update_gradients_gpu(N, net_params[param_id]->mutable_gpu_diff(),
        val_m->gpu_data(), val_t->gpu_data(), eps_hat, local_rate*correction);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(DirectionalAdamSolver);
REGISTER_SOLVER_CLASS(DirectionalAdam);

}  // namespace caffe
