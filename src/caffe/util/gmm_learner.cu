#include "caffe/util/unsupervised_learner.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GMMLearner<Dtype>::fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  CHECK_GE(blobs.size(), 1);
  CHECK_LE(blobs.size(), 3);
  CHECK_EQ(blobs[0]->count(), num_clusters_ * dim_);
  caffe_copy(num_clusters_*dim_, clusters_.gpu_data(), blobs[0]->mutable_gpu_data());
  if (blobs.size() >= 2) {
    CHECK_EQ(blobs[1]->count(), num_clusters_ * dim_);
    caffe_copy(num_clusters_ * dim_, variances_.gpu_data(), blobs[1]->mutable_gpu_data());
  }
  if (blobs.size() == 3) {
    CHECK_EQ(blobs[2]->count(), num_clusters_);
    caffe_copy(num_clusters_, cluster_weights_.gpu_data(), blobs[2]->mutable_gpu_data());
  }
}

template <typename T>
void no_op(T*) {}

template <typename Dtype>
bool GMMLearner<Dtype>::step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective) {
  CHECK_GE(input.size(), 1);
  CHECK(input[0]) << "step_gpu input was null";
  CHECK_GT(input[0]->count(), 0) << "step_gpu input has data";
  if (!called_setup_) {
    this->setup(input);
    called_setup_ = true;
  }

  if (!did_finished_kmeans_init_) {
    did_finished_kmeans_init_ = !kmeans_init_.step_gpu(input, objective);
    if (did_finished_kmeans_init_) {
      shared_ptr<Blob<Dtype> > cluster_ptr(&clusters_, no_op<Blob<Dtype> >);
      const vector<shared_ptr<Blob<Dtype> > > blobs(1, cluster_ptr);
      kmeans_init_.fill_gpu(blobs);
    } else {
      return true;
    }
  }

  estep_gpu(input[0], objective);
  iter_ += batch_size_;
  // if (iter_ - last_mstep_iter_ >= mstep_frequency_ && epoch_iter_ < 2) {
  //   mstep_gpu();
  //   last_mstep_iter_ = iter_;
  //   if (epoch_iter_ == 0) {
  //     split_clusters_gpu();
  //   }
  // }

  if (iter_ >= num_batches_ * batch_size_) {
    current_log_likelihood_ /= iter_;
    current_kmeans_objective_ /= iter_;
    LOG(INFO) << "\tEpoch #" << epoch_iter_ 
      << " log-likelihood: " << current_log_likelihood_
      << " kmeans: " << current_kmeans_objective_;
    mstep_gpu();
    if (epoch_iter_ >= 2 &&
        (abs(current_log_likelihood_ - previous_log_likelihood_) <= convergence_threshold_
         || epoch_iter_ >= max_iterations_)) {
      return false;
    }
    // if (epoch_iter_ < 2) {
    //   tie_variances_gpu();
    // }
    iter_ = 0;
    epoch_iter_++;
    previous_log_likelihood_ = current_log_likelihood_;
    current_log_likelihood_ = 0;
    current_kmeans_objective_ = 0;
    caffe_gpu_set(M_.count(), Dtype(0), M_.mutable_gpu_data());
    caffe_gpu_set(Q_.count(), Dtype(0), Q_.mutable_gpu_data());
    caffe_gpu_set(N_.count(), Dtype(0), N_.mutable_gpu_data());
  }
  return true;
}

template <typename Dtype>
Dtype GMMLearner<Dtype>::objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  return current_log_likelihood_;
}

template <typename Dtype>
__global__ void kernel_weighted_distance(const int num_examples, const int num_clusters, const int dim,
    const Dtype* data, const Dtype* clusters, const Dtype* weights, Dtype* distance, Dtype* unweighted_distance) {
  CUDA_KERNEL_LOOP(index, num_clusters * num_examples) {
    const int c = index % num_clusters; // cluster index
    const int i = index / num_clusters; // input index
    Dtype val = 0;
    Dtype unweighted_val = 0;
    for (int j = 0; j < dim; ++j) {
      const Dtype x = data[i * dim + j];
      const Dtype cluster = clusters[c * dim + j];
      const Dtype weight = weights[c * dim + j];
      const Dtype dist = (x - cluster) * (x - cluster);
      val += dist / weight;
      unweighted_val += dist;
    }
    distance[index] = val;
    unweighted_distance[index] = unweighted_val;
  }
}

template <typename Dtype>
void weighted_distance_gpu(const int num_examples, const int num_clusters, const int dim,
  const Dtype* data, const Dtype* clusters, const Dtype* weights, Dtype* distance, Dtype* unweighted_distance) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  const int num_kernels = num_clusters * num_examples;
  kernel_weighted_distance<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_examples, num_clusters, dim, data, clusters, weights, distance, unweighted_distance);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void kernel_arg_max(const int rows, const int cols,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(i, rows) {
    Dtype max_value = in[i * cols];
    int max_index = 0;
    for (int j = 1; j < cols; ++j) {
      const Dtype dist = in[i * cols + j];
      max_index = (dist >= max_value) * j + (dist < max_value) * max_index;
      max_value = max(dist, max_value);
    }
    out[i * cols + max_index] = Dtype(1);
  }
}

template <typename Dtype>
void arg_max_gpu(const int rows, const int cols, const Dtype* in, Dtype* out) {
  caffe_gpu_set(rows * cols, Dtype(0), out);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_arg_max<Dtype><<<CAFFE_GET_BLOCKS(rows), CAFFE_CUDA_NUM_THREADS>>>(
      rows, cols, in, out);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void kernel_arg_min(const int rows, const int cols,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(i, rows) {
    Dtype min_value = in[i * cols];
    int min_index = 0;
    for (int j = 1; j < cols; ++j) {
      const Dtype dist = in[i * cols + j];
      min_index = (dist <= min_value) * j + (dist > min_value) * min_index;
      min_value = min(dist, min_value);
    }
    out[i * cols + min_index] = Dtype(1);
  }
}

template <typename Dtype>
void arg_min_gpu(const int rows, const int cols, const Dtype* in, Dtype* out) {
  caffe_gpu_set(rows * cols, Dtype(0), out);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_arg_min<Dtype><<<CAFFE_GET_BLOCKS(rows), CAFFE_CUDA_NUM_THREADS>>>(
      rows, cols, in, out);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void kernel_log_sum_exp(const int rows, const int cols,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(i, rows) {
    Dtype max_value = in[i * cols];
    for (int j = 0; j < cols; ++j) {
      max_value = max(max_value, in[i * cols + j]);
    }
    Dtype value = 0;
    for (int j = 0; j < cols; ++j) {
      value += exp(in[i * cols + j] - max_value);
    }
    out[i] = log(value) + max_value;
  }
}

template <typename Dtype>
void log_sum_exp_gpu(const int rows, const int cols, const Dtype* in, Dtype* out) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_log_sum_exp<Dtype><<<CAFFE_GET_BLOCKS(rows), CAFFE_CUDA_NUM_THREADS>>>(
      rows, cols, in, out);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void GMMLearner<Dtype>::estep_gpu(const shared_ptr<Blob<Dtype> >& input, Dtype* objective) {
  weighted_distance_gpu(batch_size_, num_clusters_, dim_,
    input->gpu_data(), clusters_.gpu_data(), variances_.gpu_data(),
    distances_.mutable_gpu_data(), distances_.mutable_gpu_diff());
  // Calculate log probabilities
  // caffe_copy(distances_.count(), distances_.gpu_data(), distances_.mutable_gpu_diff());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, num_clusters_, 1,
    Dtype(1), batch_size_ones_.gpu_data(), log_norm_factor_.gpu_data(),
    Dtype(-0.5), distances_.mutable_gpu_data());
  // Assign examples to clusters
  arg_max_gpu<Dtype>(batch_size_, num_clusters_,
    distances_.gpu_data(), assignments_.mutable_gpu_data());
  //arg_min_gpu<Dtype>(batch_size_, num_clusters_,
  //  distances_.gpu_data(), assignments_.mutable_gpu_data());
  // log_sum_exp_gpu<Dtype>(batch_size_, num_clusters_,
  //   distances_.gpu_diff(), batch_size_ones_.mutable_gpu_diff());
  // thrust::device_ptr<Dtype> temp_ptr = thrust::device_pointer_cast(batch_size_ones_.mutable_gpu_diff());
  // thrust::for_each(thrust::device, temp_ptr, temp_ptr + batch_size_, _1 < -5);
  // arg_min<Dtype>(batch_size_, num_clusters_, distances_.cpu_data(), assignments_.mutable_cpu_data());
  // KMeans Objective...
  caffe_gpu_mul<Dtype>(batch_size_ * num_clusters_,
    distances_.gpu_diff(), assignments_.gpu_data(), assignments_.mutable_gpu_diff());
  const Dtype d = assignments_.asum_diff();
  current_kmeans_objective_ += d;
  //caffe_gpu_dot<Dtype>(batch_size_, distances_.gpu_data(), batch_size_ones_.gpu_data(), &d);
  // LOG(INFO) << "Kmeans Mini-Batch Objective: " << d / batch_size_;
  // Update M
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_clusters_, dim_, batch_size_,
    Dtype(1), assignments_.gpu_data(), input->gpu_data(),
    Dtype(1), M_.mutable_gpu_data());
  // Update Q
  caffe_gpu_sqr<Dtype>(input->count(), input->gpu_data(), input->mutable_gpu_diff());
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_clusters_, dim_, batch_size_,
    Dtype(1), assignments_.gpu_data(), input->gpu_diff(),
    Dtype(1), Q_.mutable_gpu_data());
  // Update N
  caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, num_clusters_,
    Dtype(1), assignments_.gpu_data(), batch_size_ones_.gpu_data(),
    Dtype(1), N_.mutable_gpu_data());
  // Update log-likelihood
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, num_clusters_, 1,
    Dtype(1), batch_size_ones_.gpu_data(), cluster_weights_.gpu_diff(),
    Dtype(1), distances_.mutable_gpu_data());
  log_sum_exp_gpu<Dtype>(batch_size_, num_clusters_,
    distances_.gpu_data(), batch_size_ones_.mutable_gpu_diff());
  Dtype batch_log_likelihood;
  caffe_gpu_dot(batch_size_ones_.count(),
    batch_size_ones_.gpu_data(), batch_size_ones_.gpu_diff(),
    &batch_log_likelihood);
  current_log_likelihood_ += batch_log_likelihood;
  if (objective) {
    *objective = batch_log_likelihood / batch_size_;
  }
}

template <typename Dtype>
void GMMLearner<Dtype>::mstep_gpu() {
  // Backup the old value first
  caffe_copy(cluster_weights_.count(), cluster_weights_.gpu_data(), cluster_weights_.mutable_gpu_diff());
  caffe_copy(clusters_.count(), clusters_.gpu_data(), clusters_.mutable_gpu_diff());
  caffe_copy(variances_.count(), variances_.gpu_data(), variances_.mutable_gpu_diff());
  caffe_copy(log_norm_factor_.count(), log_norm_factor_.gpu_data(), log_norm_factor_.mutable_gpu_diff());

  caffe_gpu_inv<Dtype>(num_clusters_, N_.gpu_data(), N_.mutable_gpu_diff());
  // Update the cluster's weights
  Dtype total_n;
  caffe_gpu_dot(num_clusters_, batch_size_ones_.gpu_data(), N_.gpu_data(), &total_n);
  if (!soft_kmeans_) {
    caffe_gpu_axpby<Dtype>(num_clusters_, Dtype(1) / total_n, N_.gpu_data(),
        Dtype(0), cluster_weights_.mutable_gpu_data());
  }
  // Update the centroids
  caffe_gpu_dgmm<Dtype>(CblasLeft, num_clusters_, dim_, M_.gpu_data(),
            N_.gpu_diff(), clusters_.mutable_gpu_data());
  // Update the variances
  if (!soft_kmeans_) { // When using soft kmeans dont' update the variances to keep them at 1.
    caffe_gpu_dgmm<Dtype>(CblasLeft, num_clusters_, dim_, Q_.gpu_data(),
              N_.gpu_diff(), variances_.mutable_gpu_data());
    caffe_gpu_sqr<Dtype>(num_clusters_ * dim_, clusters_.gpu_data(), M_.mutable_gpu_diff());
    caffe_gpu_axpy<Dtype>(num_clusters_ * dim_, Dtype(-1), M_.gpu_diff(), variances_.mutable_gpu_data());
    caffe_gpu_add_scalar<Dtype>(num_clusters_ * dim_, fudge_factor_, variances_.mutable_gpu_data());
  }
  // Keep the old value if the cluster contains a single data-point
  const Dtype* N = N_.cpu_data();
  for (int i = 0; i < N_.count(); ++i) {
    if (N[i] <= 1) {
      LOG(INFO) << "Found single cluster at: " << i;
      caffe_copy(1, cluster_weights_.gpu_diff() + i, cluster_weights_.mutable_gpu_data() + i);
      caffe_copy(dim_, clusters_.gpu_diff() + i * dim_, clusters_.mutable_gpu_data() + i * dim_);
      caffe_copy(dim_, variances_.gpu_diff() + i * dim_, variances_.mutable_gpu_data() + i * dim_);
    }
  }
  // Save the log cluster's weights for later use
  caffe_gpu_log<Dtype>(num_clusters_, cluster_weights_.gpu_data(), cluster_weights_.mutable_gpu_diff());
  // Update normalization factor
  caffe_gpu_set<Dtype>(log_norm_factor_.count(),
    Dtype(-0.5 * dim_ * log(2.0 * M_PI)), log_norm_factor_.mutable_gpu_data());
  caffe_gpu_log<Dtype>(variances_.count(), variances_.gpu_data(), variances_.mutable_gpu_diff());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_clusters_, dim_,
    Dtype(-0.5), variances_.gpu_diff(), batch_size_ones_.gpu_data(),
    Dtype(1), log_norm_factor_.mutable_gpu_data());
  // Keep old value as above.
  // Done separately so it would be possible to reuse variances_.mutable_gpu_diff()
  for (int i = 0; i < N_.count(); ++i) {
    if (N[i] <= 1) {
      caffe_copy(1, log_norm_factor_.gpu_diff() + i, log_norm_factor_.mutable_gpu_data() + i);
    }
  }
}

template <typename Dtype>
void GMMLearner<Dtype>::split_clusters_gpu() {
  // TODO: finish implementing splitting...
  // From looking at the article splitting isn't critical to convergence
}

template <typename Dtype>
void GMMLearner<Dtype>::tie_variances_gpu() {
  caffe_gpu_gemv<Dtype>(CblasTrans, num_clusters_, dim_,
    Dtype(1), variances_.gpu_data(), cluster_weights_.gpu_data(),
    Dtype(0), variances_.mutable_gpu_diff());
  for (int i = 0; i < num_clusters_; ++i) {
    caffe_copy(dim_, variances_.gpu_diff(), variances_.mutable_gpu_data() + i * dim_);
  }
  caffe_gpu_set<Dtype>(log_norm_factor_.count(),
    Dtype(-0.5 * dim_ * log(2.0 * M_PI)), log_norm_factor_.mutable_gpu_data());
  caffe_gpu_log<Dtype>(variances_.count(), variances_.gpu_data(), variances_.mutable_gpu_diff());
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_clusters_, dim_,
    Dtype(-0.5), variances_.gpu_diff(), batch_size_ones_.gpu_data(),
    Dtype(1), log_norm_factor_.mutable_gpu_data());
}


template void GMMLearner<float>::fill_gpu(const vector<shared_ptr<Blob<float> > >& blobs);
template void GMMLearner<double>::fill_gpu(const vector<shared_ptr<Blob<double> > >& blobs);
template bool GMMLearner<float>::step_gpu(const vector<shared_ptr<Blob<float> > >& input, float* objective);
template bool GMMLearner<double>::step_gpu(const vector<shared_ptr<Blob<double> > >& input, double* objective);
template float GMMLearner<float>::objective_gpu(const vector<shared_ptr<Blob<float> > >& input);
template double GMMLearner<double>::objective_gpu(const vector<shared_ptr<Blob<double> > >& input);
template void GMMLearner<float>::tie_variances_gpu();
template void GMMLearner<double>::tie_variances_gpu();
template void GMMLearner<float>::split_clusters_gpu();
template void GMMLearner<double>::split_clusters_gpu();
template void GMMLearner<float>::mstep_gpu();
template void GMMLearner<double>::mstep_gpu();
template void GMMLearner<float>::estep_gpu(const shared_ptr<Blob<float> >& input, float* objective);
template void GMMLearner<double>::estep_gpu(const shared_ptr<Blob<double> >& input, double* objective);
}  // namespace caffe
