#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SMORMS3Update(int N, Dtype* g, Dtype* m, Dtype* v, Dtype* mem,
    Dtype eps_hat, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    const Dtype r = Dtype(1) / (mem[i] + Dtype(1));
    const Dtype grad = g[i];
    Dtype mi = m[i] = m[i]*(1-r) + grad*r;
    Dtype vi = v[i] = v[i]*(1-r) + grad*grad*r;
    g[i] *= min(local_rate, mi * mi / (vi + eps_hat)) / (sqrt(vi) + eps_hat);
    mem[i] = Dtype(1) + mem[i] * (Dtype(1) - mi * mi / (vi + eps_hat));
  }
}
template <typename Dtype>
void smorms3_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype* mem,
    Dtype eps_hat, Dtype local_rate) {
  SMORMS3Update<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, m, v, mem, eps_hat, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void smorms3_update_gpu<float>(int N, float* g, float* m, float* v, float* mem,
    float eps_hat, float local_rate);
template void smorms3_update_gpu<double>(int N, double* g, double* m, double* v, double* mem,
    double eps_hat, double local_rate);

}  // namespace caffe
