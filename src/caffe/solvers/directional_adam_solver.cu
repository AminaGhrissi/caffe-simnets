#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void DirectionalAdamUpdateMoments(int N, const Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta2) {
  CUDA_KERNEL_LOOP(i, N) {
    const Dtype gi = g[i];
    m[i] = m[i]*beta1 + gi*(1-beta1);
    v[i] = v[i]*beta2 + gi*gi*(1-beta2);
  }
}

template <typename Dtype>
__global__ void DirectionalAdamUpdateGradients(int N, Dtype* g, const Dtype* m,
    const Dtype* v, Dtype eps_hat, Dtype corrected_local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = corrected_local_rate * m[i] / (sqrt(v[i]) + eps_hat);
  }
}

template <typename Dtype>
void directional_adam_update_moments_gpu(int N, const Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta2) {
  DirectionalAdamUpdateMoments<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, m, v, beta1, beta2);
  CUDA_POST_KERNEL_CHECK;
}
template void directional_adam_update_moments_gpu<float>(int N, const float* g,
    float* m, float* v, float beta1, float beta2);
template void directional_adam_update_moments_gpu<double>(int N, const double* g,
    double* m, double* v, double beta1, double beta2);

template <typename Dtype>
void directional_adam_update_gradients_gpu(int N, Dtype* g, const Dtype* m, const Dtype* v,
    Dtype eps_hat, Dtype corrected_local_rate) {
  DirectionalAdamUpdateGradients<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, m, v, eps_hat, corrected_local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void directional_adam_update_gradients_gpu<float>(int N, float* g,
    const float* m, const float* v, float eps_hat, float corrected_local_rate);
template void directional_adam_update_gradients_gpu<double>(int N, double* g,
    const double* m, const double* v, double eps_hat, double corrected_local_rate);

}  // namespace caffe
