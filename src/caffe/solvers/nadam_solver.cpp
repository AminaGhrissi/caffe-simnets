#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void NadamSolver<Dtype>::NadamPreSolve() {
  // Add the extra history entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void nadam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1, Dtype beta1pow,
    Dtype beta2, Dtype beta2pow, Dtype eps_hat, Dtype local_rate);
#endif

template <typename Dtype>
void NadamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
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

  const int t = this->iter_ + 1;
  const Dtype beta1pow = pow(beta1, t);
  const Dtype beta2pow = pow(beta2, t);
  const int N = net_params[param_id]->count();
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

    // update gradient
    const Dtype* m = val_m->cpu_data();
    const Dtype* v = val_v->cpu_data();
    Dtype* grad = net_params[param_id]->mutable_cpu_diff();
    for (int i = 0; i < N; ++i) {
      const Dtype m_hat = (beta1 / (Dtype(1) - beta1pow*beta1)) * m[i]
          + ((1 - beta1) / (Dtype(1) - beta1pow)) * grad[i];
      const Dtype v_hat = (beta2 / (Dtype(1) - beta2pow)) * v[i];
      grad[i] = local_rate * m_hat / (sqrt(v_hat) + eps_hat);
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    nadam_update_gpu(N, net_params[param_id]->mutable_gpu_diff(),
        val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), beta1, beta1pow,
        beta2, beta2pow, eps_hat, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(NadamSolver);
REGISTER_SOLVER_CLASS(Nadam);

}  // namespace caffe
