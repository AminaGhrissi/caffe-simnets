#include "caffe/util/unsupervised_learner.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
bool LabelingDensityLearner<Dtype>::step_gpu(const vector<shared_ptr<Blob<Dtype> > >&  input, Dtype* objective) {
  CHECK_GE(input.size(), 2);
  CHECK(input[0] && input[1]) << "step_gpu input was null";
  CHECK_GT(input[0]->count(), 0) << "step_gpu patch input has data";
  CHECK_GT(input[1]->count(), 0) << "step_gpu label input has data";
  if (objective) {
    *objective = INFINITY;
  }
  if (!called_setup_) {
    this->setup(input);
    called_setup_ = true;
  }

  update_densities_gpu(input);
  iter_++;

  if (iter_ % num_batches_ == 0) {
    caffe_copy(densities_.count(), densities_.gpu_data(), densities_.mutable_gpu_diff());
    caffe_gpu_set(densities_.count(), Dtype(0), densities_.mutable_gpu_data());
    caffe_gpu_set(per_label_count_.count(), Dtype(0), per_label_count_.mutable_gpu_data());
  }

  return iter_ / num_batches_ < max_iterations_;
}

template <typename Dtype>
void LabelingDensityLearner<Dtype>::fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  CHECK_EQ(blobs.size(), 1);
  CHECK_EQ(blobs[0]->count(), num_labels_ * dim_);
  caffe_copy(num_labels_ * dim_, densities_.gpu_diff(), blobs[0]->mutable_gpu_data());
}

template <typename Dtype>
Dtype LabelingDensityLearner<Dtype>::objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  return INFINITY;
}

template <typename Dtype>
__global__ void kernel_normalize_by_softmax(const int N, const int dim,
  const Dtype* labels, const Dtype* priors, Dtype* data) {
  CUDA_KERNEL_LOOP(n, N) {
    const int label = static_cast<int>(labels[n]);
    Dtype m = data[n * dim] + priors[label * dim];
    int m_idx = 0;
    for (int k = 1; k < dim; ++k) {
      const Dtype x = data[n * dim + k] + priors[label * dim + k];
      m_idx = (m <= x) * k + (m > x) * m_idx;
      m = max(m, x);
    }
    Dtype sum = 0;
    for (int k = 0; k < dim; ++k) {
      sum += exp(data[n * dim + k] + priors[label * dim + k] - m);
    }
    const Dtype softmax = log(sum) + m;
    for (int k = 0; k < dim; ++k) {
      data[n * dim + k] += priors[label * dim + k] - softmax;
    }
  }
}

template <typename Dtype>
__global__ void kernel_normalize_by_hard_assignment(const int N, const int dim, const Dtype fudge_factor,
  const Dtype* labels, const Dtype* priors, Dtype* data) {
  CUDA_KERNEL_LOOP(n, N) {
    const int label = static_cast<int>(labels[n]);
    Dtype m = data[n * dim] + priors[label * dim];
    int m_idx = 0;
    for (int k = 1; k < dim; ++k) {
      const Dtype x = data[n * dim + k] + priors[label * dim + k];
      m_idx = (m <= x) * k + (m > x) * m_idx;
      m = max(m, x);
    }
    for (int k = 0; k < dim; ++k) {
      data[n * dim + k] = fudge_factor; // TODO: maybe switch to log(fudge_factor) ??
    }
    data[n * dim + m_idx] = 0; // TODO: maybe switch to log(1.0 - fudge_factor) ??
  }
}

template <typename Dtype>
void normalize_data_gpu(const int N, const int dim, const bool soft_assignment, const Dtype fudge_factor,
  const Dtype* labels, const Dtype* priors, Dtype* data) {
  if (soft_assignment) {
    kernel_normalize_by_softmax<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, dim, labels, priors, data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    kernel_normalize_by_hard_assignment<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, dim, fudge_factor, labels, priors, data);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
__global__ void kernel_normalize_per_label_by_softmax(const int num_labels, const int N, const int dim,
  const Dtype* priors, const Dtype* data, Dtype* densities) {
  CUDA_KERNEL_LOOP(index, num_labels * N) {
    const int n = index % N;
    const int label = index / N;
    Dtype m = data[n * dim] + priors[label * dim];
    int m_idx = 0;
    for (int k = 1; k < dim; ++k) {
      const Dtype x = data[n * dim + k] + priors[label * dim + k];
      m_idx = (m <= x) * k + (m > x) * m_idx;
      m = max(m, x);
    }
    Dtype sum = 0;
    for (int k = 0; k < dim; ++k) {
      sum += exp(data[n * dim + k] + priors[label * dim + k] - m);
    }
    const Dtype softmax = log(sum) + m;
    for (int k = 0; k < dim; ++k) {
      densities[(label * N + n) * dim + k] = data[n * dim + k] + priors[label * dim + k] - softmax;
    }
  }
}

template <typename Dtype>
__global__ void kernel_normalize_per_label_by_hard_assignment(const int num_labels, const int N, const int dim,
  const Dtype fudge_factor, const Dtype* priors, const Dtype* data, Dtype* densities) {
  CUDA_KERNEL_LOOP(index, num_labels * N) {
    const int n = index % N;
    const int label = index / N;
    Dtype m = data[n * dim] + priors[label * dim];
    int m_idx = 0;
    for (int k = 1; k < dim; ++k) {
      const Dtype x = data[n * dim + k] + priors[label * dim + k];
      m_idx = (m <= x) * k + (m > x) * m_idx;
      m = max(m, x);
    }
    for (int k = 0; k < dim; ++k) {
      densities[(label * N + n) * dim + k] = fudge_factor; // TODO: maybe switch to log(fudge_factor) ??
    }
    densities[(label * N + n) * dim + m_idx] = 0; // TODO: maybe switch to log(1.0 - fudge_factor) ??
  }
}

template <typename Dtype>
void normalize_data_per_label_gpu(const int num_labels, const int N, const int dim,
  const bool soft_assignment, const Dtype fudge_factor,
  const Dtype* priors, const Dtype* data, Dtype* densities) {
  if (soft_assignment) {
    kernel_normalize_per_label_by_softmax<Dtype><<<CAFFE_GET_BLOCKS(N*num_labels), CAFFE_CUDA_NUM_THREADS>>>(
        num_labels, N, dim, priors, data, densities);
    CUDA_POST_KERNEL_CHECK;
  } else {
    kernel_normalize_per_label_by_hard_assignment<Dtype><<<CAFFE_GET_BLOCKS(N*num_labels), CAFFE_CUDA_NUM_THREADS>>>(
        num_labels, N, dim, fudge_factor, priors, data, densities);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
__global__ void kernel_count_labels(const int N, const int num_labels,
    const Dtype* labels, Dtype* labels_count) {
  CUDA_KERNEL_LOOP(i, num_labels) {;
    int count = 0;
    for (int n = 0; n < N; ++n) {
      const int label = static_cast<int>(labels[n]);
      count += (label == i);
    }
    labels_count[i] = count;
  }
}

template <typename Dtype>
__global__ void kernel_update_densities(const int N, const int dim, const int num_labels,
    const Dtype* labels, const Dtype* labels_count, const Dtype* batch_labels_count,
    const Dtype* patch_densities, Dtype* densities) {
  CUDA_KERNEL_LOOP(index, num_labels * dim) {
    const int i = index % num_labels;
    const int k = index / num_labels;
    const int batch_count = batch_labels_count[i];
    const int old_count = labels_count[i];
    const int new_count = old_count + batch_count;

    // Calculate batch density in log space
    Dtype m = -INFINITY;
    for (int n = 0; n < N; ++n) {
      const int label = static_cast<int>(labels[n]);
      const Dtype x = patch_densities[n * dim + k];
      m = max(m, x) * (i == label) + m * (i != label);
    }
    Dtype sum = 0;
    for (int n = 0; n < N; ++n) {
      const int label = static_cast<int>(labels[n]);
      const Dtype x = patch_densities[n * dim + k] * (label == i);
      sum += exp(x - m) * (label == i);
    }
    const Dtype batch_density = log(sum / batch_count) + m;

    // Calculate update step in log space
    const Dtype old_density = densities[i * dim + k];
    const Dtype x1 = batch_density + log(batch_count / Dtype(new_count));
    const Dtype x2 = old_density + log(old_count / Dtype(new_count));
    const Dtype m2 = max(x1, x2);
    const Dtype new_density = (log(exp(x1 - m2) + exp(x2 - m2)) + m2) * (old_count > 0)
                            + batch_density * (old_count == 0);
    densities[i * dim + k] = new_density * (batch_count > 0) + old_density * (batch_count == 0);
  }
}

template <typename Dtype>
__global__ void kernel_update_densities_per_label(const int N, const int dim, const int num_labels,
    const Dtype lambda, const Dtype fudge_factor, const Dtype* labels, const Dtype* labels_count, const Dtype* batch_labels_count,
    const Dtype* patch_densities, Dtype* densities) {
  CUDA_KERNEL_LOOP(index, num_labels * dim) {
    const int i = index % num_labels;
    const int k = index / num_labels;
    const int batch_count = batch_labels_count[i];
    const int old_count = labels_count[i];
    const int new_count = old_count + batch_count;

    // Calculate batch density in log space
    Dtype m1 = -INFINITY, m2 = -INFINITY;
    for (int n = 0; n < N; ++n) {
      const int label = static_cast<int>(labels[n]);
      const Dtype x = patch_densities[(i * N + n) * dim + k];
      m1 = max(m1, x) * (i == label) + m1 * (i != label);
      m2 = max(m2, x) * (i != label) + m2 * (i == label);
    }
    // printf("i = %d, k = %d, m1 = %f, m2 = %f\n", i, k, m1, m2);
    Dtype sum1 = 0, sum2 = 0;
    for (int n = 0; n < N; ++n) {
      const int label = static_cast<int>(labels[n]);
      const Dtype x = patch_densities[(i * N + n) * dim + k];
      sum1 += exp((x - m1) * (label == i)) * (label == i);
      sum2 += exp((x - m2) * (label != i)) * (label != i);
    }
    // printf("i = %d, k = %d, sum1 = %f, sum2 = %f, batch_count = %d, N = %d, lambda = %f\n", i, k, sum1, sum2, batch_count, N, lambda);
    const Dtype correct_term = log(sum1/batch_count) + m1;
    const Dtype incorrect_term = log(sum2/(N - batch_count)) + m2;
    const Dtype m3 = max(correct_term, incorrect_term + log(lambda));
    const Dtype inner_sum = exp(correct_term - m3) - exp(incorrect_term + log(lambda) - m3);
    const Dtype fixed_inner_sum = inner_sum * (inner_sum > 0) + (inner_sum <= 0);
    const Dtype batch_density = (log((fixed_inner_sum) / (1.0 - lambda)) + m3) * (inner_sum > 0)
                              + fudge_factor * (inner_sum <= 0);
    // printf("i = %d, k = %d, batch_density: %f\n", i, k, batch_density);
    // Calculate update step in log space
    const Dtype old_density = densities[i * dim + k];
    const Dtype x1 = batch_density + log(batch_count / Dtype(new_count));
    const Dtype x2 = old_density + log(old_count / Dtype(new_count));
    const Dtype m4 = max(x1, x2);
    const Dtype new_density = (log(exp(x1 - m4) + exp(x2 - m4)) + m4) * (old_count > 0)
                            + batch_density * (old_count == 0);
    densities[i * dim + k] = new_density * (batch_count > 0) + old_density * (batch_count == 0);
  }
}

template <typename Dtype>
void LabelingDensityLearner<Dtype>::update_densities_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  const Dtype* data = input[0]->gpu_data();
  const Dtype* labels = input[1]->gpu_data();

  // Normalize patches to densities in log space
  const Dtype* priors = densities_.gpu_diff();
  if (lambda_ == 0 || iter_ < num_batches_) {
    Dtype* patch_densities = input[0]->mutable_gpu_diff();
    caffe_copy(input[0]->count(), data, patch_densities);
    normalize_data_gpu(batch_size_, dim_, soft_assignment_, fudge_factor_, labels, priors, patch_densities);
  } else {
    normalize_data_per_label_gpu(num_labels_, batch_size_, dim_, soft_assignment_, fudge_factor_,
      priors, data, per_label_densities_.mutable_gpu_data());
  }

  // Count how many patches per label
  Dtype* batch_labels_count = per_label_count_.mutable_gpu_diff();
  kernel_count_labels<Dtype><<<CAFFE_GET_BLOCKS(num_labels_), CAFFE_CUDA_NUM_THREADS>>>(
      batch_size_, num_labels_, labels, batch_labels_count);
  CUDA_POST_KERNEL_CHECK;

  // Update densities
  Dtype* densities = densities_.mutable_gpu_data();
  Dtype* labels_count = per_label_count_.mutable_gpu_data();
  if (lambda_ == 0 || iter_ < num_batches_) {
    const Dtype* patch_densities = input[0]->gpu_diff();
    kernel_update_densities<Dtype><<<CAFFE_GET_BLOCKS(num_labels_ * dim_), CAFFE_CUDA_NUM_THREADS>>>(
        batch_size_, dim_, num_labels_, labels, labels_count, batch_labels_count,
        patch_densities, densities);
  } else {
    const Dtype* per_label_densities = per_label_densities_.gpu_data();
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024);
    kernel_update_densities_per_label<Dtype><<<CAFFE_GET_BLOCKS(num_labels_ * dim_), CAFFE_CUDA_NUM_THREADS>>>(
        batch_size_, dim_, num_labels_, lambda_, fudge_factor_, labels, labels_count, batch_labels_count,
        per_label_densities, densities);
  }
  CUDA_POST_KERNEL_CHECK;

  // Update label's count
  caffe_gpu_add(num_labels_, labels_count, batch_labels_count, labels_count);
}

template void LabelingDensityLearner<float>::fill_gpu(const vector<shared_ptr<Blob<float> > >& blobs);
template void LabelingDensityLearner<double>::fill_gpu(const vector<shared_ptr<Blob<double> > >& blobs);
template bool LabelingDensityLearner<float>::step_gpu(const vector<shared_ptr<Blob<float> > >& input, float* objective);
template bool LabelingDensityLearner<double>::step_gpu(const vector<shared_ptr<Blob<double> > >& input, double* objective);
template float LabelingDensityLearner<float>::objective_gpu(const vector<shared_ptr<Blob<float> > >& input);
template double LabelingDensityLearner<double>::objective_gpu(const vector<shared_ptr<Blob<double> > >& input);
}  // namespace caffe
