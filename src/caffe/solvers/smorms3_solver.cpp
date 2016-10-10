#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void SMORMS3Solver<Dtype>::SMORMS3PreSolve() {
  // Add the extra history entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    Blob<Dtype>* mem = this->history_[this->history_.size()-1].get();
    caffe_set<Dtype>(mem->count(), Dtype(1), mem->mutable_cpu_data());
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void smorms3_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype* mem,
    Dtype eps_hat, Dtype local_rate);
#endif

template <typename Dtype>
void SMORMS3Solver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];

  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_mem = this->history_[param_id + 2*update_history_offset].get();

  const int N = net_params[param_id]->count();
  const Dtype eps_hat = this->param_.delta();

  switch (Caffe::mode()) {
    case Caffe::CPU: {
    Dtype* grad = net_params[param_id]->mutable_cpu_diff();
    Dtype* m = val_m->mutable_cpu_data();
    Dtype* v = val_v->mutable_cpu_data();
    Dtype* mem = val_mem->mutable_cpu_data();
    for (int i = 0; i < N; ++i) {
      const Dtype r = Dtype(1) / (mem[i] + Dtype(1));
      m[i] = (Dtype(1) - r) * m[i] + r * grad[i];
      v[i] = (Dtype(1) - r) * v[i] + r * grad[i] * grad[i];
      grad[i] *= std::min(local_rate, m[i] * m[i] / (v[i] + eps_hat)) / (std::sqrt(v[i] + eps_hat));
      mem[i] = Dtype(1) + mem[i] * (Dtype(1) - m[i] * m[i] / (v[i] + eps_hat));
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    smorms3_update_gpu(N, net_params[param_id]->mutable_gpu_diff(),
        val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), val_mem->mutable_gpu_data(),
        eps_hat, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(SMORMS3Solver);
REGISTER_SOLVER_CLASS(SMORMS3);

}  // namespace caffe
