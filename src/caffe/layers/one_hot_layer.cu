#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/one_hot_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void OneHotForward(const int nthreads, const Dtype* bottom_data,
    const Dtype one_value, const int M, const int N, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / N;
    const int i = index % N;
    const int top_index = static_cast<int>(round(bottom_data[index]));
    top_data[n * N * M + top_index * N + i] = one_value;
  }
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype zero_value = to_log_ ? Dtype(-INFINITY) : Dtype(0);
  const Dtype one_value = to_log_ ? Dtype(0) : Dtype(1);
  caffe_gpu_set<Dtype>(top[0]->count(), zero_value, top_data);
  OneHotForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, one_value, M_, N_, top_data);
}

template <typename Dtype>
void OneHotLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to OneHotLayer input.";
}

INSTANTIATE_LAYER_GPU_FUNCS(OneHotLayer);

}  // namespace caffe
