#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void NadamUpdate(int N, Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta1pow, Dtype beta2, Dtype beta2pow, Dtype eps_hat, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    Dtype gi = g[i];
    Dtype mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    Dtype vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    Dtype mi_hat = (beta1 / (Dtype(1) - beta1pow * beta1)) * mi
         + ((Dtype(1) - beta1) / (Dtype(1) - beta1pow)) * gi;
    Dtype vi_hat = (beta2 / (Dtype(1) - beta2pow)) * vi;
    g[i] = local_rate * mi_hat / (sqrt(vi_hat) + eps_hat);
  }
}
template <typename Dtype>
void nadam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1, Dtype beta1pow,
    Dtype beta2, Dtype beta2pow, Dtype eps_hat, Dtype local_rate) {
  NadamUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, m, v, beta1, beta1pow, beta2, beta2pow, eps_hat, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void nadam_update_gpu<float>(int, float*, float*, float*,
    float, float, float, float, float, float);
template void nadam_update_gpu<double>(int, double*, double*, double*,
    double, double, double, double, double, double);

}  // namespace caffe
