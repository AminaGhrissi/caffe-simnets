#include "caffe/util/unsupervised_learner.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

namespace caffe {

template <typename Dtype>
void GMMLearner<Dtype>::setup(const vector<shared_ptr<Blob<Dtype> > >& input) {
  batch_size_ = input[0]->num();
  dim_ = input[0]->channels();
  CHECK_EQ(input[0]->width(), 1);
  CHECK_EQ(input[0]->height(), 1);
  CHECK_GT(num_clusters_, 0);

  // Reshape matrices
  clusters_.Reshape(num_clusters_, dim_, 1, 1);
  variances_.Reshape(num_clusters_, dim_, 1, 1);
  log_norm_factor_.Reshape(num_clusters_, 1, 1, 1);
  cluster_weights_.Reshape(num_clusters_, 1, 1, 1);
  M_.Reshape(num_clusters_, dim_, 1, 1);
  Q_.Reshape(num_clusters_, dim_, 1, 1);
  N_.Reshape(num_clusters_, 1, 1, 1);
  distances_.Reshape(batch_size_, num_clusters_, 1, 1);
  assignments_.Reshape(batch_size_, num_clusters_, 1, 1);
  batch_size_ones_.Reshape(std::max(std::max(batch_size_, num_clusters_), dim_), 1, 1, 1);
  clusters_helper_.Reshape(num_clusters_, 1, 1, 1);

  current_kmeans_objective_ = 0;
  current_log_likelihood_ = 0;
  previous_log_likelihood_ = INFINITY;

  if (mstep_frequency_ <= 0) {
    mstep_frequency_ = (int) std::sqrt(num_batches_ * batch_size_);
  }

  if (init_cluster_scale_factor_ <= 0) {
    init_cluster_scale_factor_ = Dtype(1.0 / (dim_ * num_clusters_));
  }

  if (Caffe::mode() == Caffe::CPU) {
    caffe_set(M_.count(), Dtype(0), M_.mutable_cpu_data());
    caffe_set(Q_.count(), Dtype(0), Q_.mutable_cpu_data());
    caffe_set(N_.count(), Dtype(0), N_.mutable_cpu_data());
    caffe_set(batch_size_ones_.count(), Dtype(1), batch_size_ones_.mutable_cpu_data());
    caffe_set(variances_.count(), Dtype(1), variances_.mutable_cpu_data());
    caffe_set(log_norm_factor_.count(), Dtype(-0.5* dim_ * log(2 * M_PI)), log_norm_factor_.mutable_cpu_data());
    caffe_set(cluster_weights_.count(), Dtype(1.0 / num_clusters_), cluster_weights_.mutable_cpu_data());
    caffe_log<Dtype>(num_clusters_, cluster_weights_.cpu_data(), cluster_weights_.mutable_cpu_diff());
    // caffe_rng_gaussian<Dtype>(clusters_.count(), 0, 1, clusters_.mutable_cpu_data());
    // caffe_scal<Dtype>(clusters_.count(), init_cluster_scale_factor_, clusters_.mutable_cpu_data());
  } else {
    caffe_gpu_set(M_.count(), Dtype(0), M_.mutable_gpu_data());
    caffe_gpu_set(Q_.count(), Dtype(0), Q_.mutable_gpu_data());
    caffe_gpu_set(N_.count(), Dtype(0), N_.mutable_gpu_data());
    caffe_gpu_set(batch_size_ones_.count(), Dtype(1), batch_size_ones_.mutable_gpu_data());
    caffe_gpu_set(variances_.count(), Dtype(1), variances_.mutable_gpu_data());
    caffe_gpu_set(log_norm_factor_.count(), Dtype(-0.5* dim_ * log(2 * M_PI)), log_norm_factor_.mutable_gpu_data());
    caffe_gpu_set(cluster_weights_.count(), Dtype(1.0 / num_clusters_), cluster_weights_.mutable_gpu_data());
    caffe_gpu_log<Dtype>(num_clusters_, cluster_weights_.gpu_data(), cluster_weights_.mutable_gpu_diff());
    // caffe_gpu_rng_uniform<Dtype>(num_clusters_, -1, 1, clusters_.mutable_gpu_diff());
    // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_clusters_, dim_, 1,
    //   Dtype(1), clusters_.gpu_diff(), batch_size_ones_.gpu_data(),
    //   Dtype(0), clusters_.mutable_gpu_data());
    // caffe_gpu_rng_gaussian<Dtype>(clusters_.count(), 0, 1, clusters_.mutable_gpu_data());
    // caffe_gpu_scal<Dtype>(clusters_.count(), init_cluster_scale_factor_, clusters_.mutable_gpu_data());
  }
}

template <typename Dtype>
void GMMLearner<Dtype>::fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  CHECK_GE(blobs.size(), 1);
  CHECK_LE(blobs.size(), 3);
  CHECK_EQ(blobs[0]->count(), num_clusters_ * dim_);
  caffe_copy(num_clusters_*dim_, clusters_.cpu_data(), blobs[0]->mutable_cpu_data());
  if (blobs.size() >= 2) {
    CHECK_EQ(blobs[1]->count(), num_clusters_ * dim_);
    caffe_copy(num_clusters_ * dim_, variances_.cpu_data(), blobs[1]->mutable_cpu_data());
  }
  if (blobs.size() == 3) {
    CHECK_EQ(blobs[2]->count(), num_clusters_);
    caffe_copy(num_clusters_, cluster_weights_.cpu_data(), blobs[2]->mutable_cpu_data());
  }
}

template <typename T>
void no_op(T*) {}

template <typename Dtype>
bool GMMLearner<Dtype>::step_cpu(const vector<shared_ptr<Blob<Dtype> > >&  input, Dtype* objective) {
  CHECK_GE(input.size(), 1);
  CHECK(input[0]) << "step_cpu input was null";
  CHECK_GT(input[0]->count(), 0) << "step_cpu input has data";
  if (!called_setup_) {
    this->setup(input);
    called_setup_ = true;
  }

  if (!did_finished_kmeans_init_) {
    did_finished_kmeans_init_ = !kmeans_init_.step_cpu(input, objective);
    if (did_finished_kmeans_init_) {
      shared_ptr<Blob<Dtype> > cluster_ptr(&clusters_, no_op<Blob<Dtype> >);
      const vector<shared_ptr<Blob<Dtype> > > blobs(1, cluster_ptr);
      kmeans_init_.fill_cpu(blobs);
    } else {
      return true;
    }
  }

  estep_cpu(input[0], objective);
  iter_ += batch_size_;
  // if (iter_ - last_mstep_iter_ >= mstep_frequency_ && epoch_iter_ < 2) {
  //   mstep_cpu();
  //   last_mstep_iter_ = iter_;
  //   if (epoch_iter_ == 0) {
  //     split_clusters_cpu();
  //   }
  // }

  if (iter_ >= num_batches_ * batch_size_) {
    current_log_likelihood_ /= iter_;
    current_kmeans_objective_ /= iter_;
    LOG(INFO) << "\tEpoch #" << epoch_iter_ 
      << " log-likelihood: " << current_log_likelihood_
      << " kmeans: " << current_kmeans_objective_;
    mstep_cpu();
    if (epoch_iter_ >= 2 &&
        (abs(current_log_likelihood_ - previous_log_likelihood_) <= convergence_threshold_
         || epoch_iter_ >= max_iterations_)) {
      return false;
    }
    // if (epoch_iter_ < 2) {
    //   tie_variances_cpu();
    // }
    iter_ = 0;
    epoch_iter_++;
    previous_log_likelihood_ = current_log_likelihood_;
    current_log_likelihood_ = 0;
    current_kmeans_objective_ = 0;
    caffe_set(M_.count(), Dtype(0), M_.mutable_cpu_data());
    caffe_set(Q_.count(), Dtype(0), Q_.mutable_cpu_data());
    caffe_set(N_.count(), Dtype(0), N_.mutable_cpu_data());
  }
  return true;
}

#ifdef CPU_ONLY
template <typename Dtype>
bool GMMLearner<Dtype>::step_gpu(const vector<shared_ptr<Blob<Dtype> > >&  input, Dtype* objective) {
  NO_GPU;
}
template <typename Dtype>
bool GMMLearner<Dtype>::objective_gpu(const vector<shared_ptr<Blob<Dtype> > >&  input) {
  NO_GPU;
}
template <typename Dtype>
void GMMLearner<Dtype>::fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  NO_GPU;
}
#endif


template <typename Dtype>
Dtype GMMLearner<Dtype>::objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  return current_log_likelihood_;
}

template <typename Dtype>
void weighted_distance_cpu(const int num_examples, const int num_clusters, const int dim,
  const Dtype* data, const Dtype* clusters, const Dtype* weights, Dtype* out, Dtype* unweighted_distance) {
  for (int i = 0; i < num_examples; ++i) {
    for (int j = 0; j < num_clusters; ++j) {
      Dtype dist = 0;
      Dtype unweighted_dist = 0;
      for (int k = 0; k < dim; ++k) {
        const Dtype x = data[i * dim + k];
        const Dtype z = clusters[j * dim + k];
        const Dtype u = weights[j * dim + k];
        const Dtype dist_unit = (x - z) * (x - z);
        dist += dist_unit / u;
        unweighted_dist += dist_unit;
      }
      out[i * num_clusters + j] = dist;
      unweighted_distance[i * num_clusters + j] = unweighted_dist;
    }
  }
}

template <typename Dtype>
void arg_max_cpu(const int rows, const int cols, const Dtype* in, Dtype* out) {
  caffe_set(rows * cols, Dtype(0), out);
  for (int i = 0; i < rows; ++i) {
    Dtype max_value = in[i * cols];
    int max_index = 0;
    for (int j = 0; j < cols; ++j) {
      const Dtype x = in[i * cols + j];
      max_index = max_index * (max_value > x) + j * (max_value <= x);
      max_value = std::max(max_value, x);
    }
    out[i * cols + max_index] = Dtype(1);
  }
}

template <typename Dtype>
void log_sum_exp_cpu(const int rows, const int cols, const Dtype* in, Dtype* out) {
  for (int i = 0; i < rows; ++i) {
    Dtype max_value = in[i * cols];
    for (int j = 0; j < cols; ++j) {
      max_value = std::max(max_value, in[i * cols + j]);
    }
    Dtype value = 0;
    for (int j = 0; j < cols; ++j) {
      value += exp(in[i * cols + j] - max_value);
    }
    out[i] = log(value) + max_value;
  }
}

template <typename Dtype>
void GMMLearner<Dtype>::estep_cpu(const shared_ptr<Blob<Dtype> >& input, Dtype* objective) {
  weighted_distance_cpu(batch_size_, num_clusters_, dim_,
    input->cpu_data(), clusters_.cpu_data(), variances_.cpu_data(),
    distances_.mutable_cpu_data(), distances_.mutable_cpu_diff());
  // Calculate log probabilities
  // caffe_copy(distances_.count(), distances_.cpu_data(), distances_.mutable_cpu_diff());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, num_clusters_, 1,
    Dtype(1), batch_size_ones_.cpu_data(), log_norm_factor_.cpu_data(),
    Dtype(-0.5), distances_.mutable_cpu_data());
  // Assign examples to clusters
  arg_max_cpu<Dtype>(batch_size_, num_clusters_,
    distances_.cpu_data(), assignments_.mutable_cpu_data());
  // Test if some of the probabilities are too low...
  // log_sum_exp_cpu<Dtype>(batch_size_, num_clusters_,
    // distances_.cpu_diff(), batch_size_ones_.mutable_cpu_diff());
  // thrust::for_each(thrust::host,
  //   batch_size_ones_.mutable_cpu_diff(), batch_size_ones_.mutable_cpu_diff() + batch_size_,
  //   _1 < -5);
  // arg_min<Dtype>(batch_size_, num_clusters_, distances_.cpu_data(), assignments_.mutable_cpu_data());
  // KMeans Objective...
  caffe_mul<Dtype>(batch_size_ * num_clusters_,
    distances_.cpu_diff(), assignments_.cpu_data(), assignments_.mutable_cpu_diff());
  const Dtype d = assignments_.asum_diff();
  current_kmeans_objective_ += d;
  // Update M
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_clusters_, dim_, batch_size_,
    Dtype(1), assignments_.cpu_data(), input->cpu_data(),
    Dtype(1), M_.mutable_cpu_data());
  // Update Q
  caffe_sqr<Dtype>(input->count(), input->cpu_data(), input->mutable_cpu_diff());
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_clusters_, dim_, batch_size_,
    Dtype(1), assignments_.cpu_data(), input->cpu_diff(),
    Dtype(1), Q_.mutable_cpu_data());
  // Update N
  caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, num_clusters_,
    Dtype(1), assignments_.cpu_data(), batch_size_ones_.cpu_data(),
    Dtype(1), N_.mutable_cpu_data());
  // Update log-likelihood
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, num_clusters_, 1,
    Dtype(1), batch_size_ones_.cpu_data(), cluster_weights_.cpu_diff(),
    Dtype(1), distances_.mutable_cpu_diff());
  log_sum_exp_cpu<Dtype>(batch_size_, num_clusters_,
    distances_.cpu_diff(), batch_size_ones_.mutable_cpu_diff());
  const Dtype batch_log_likelihood = caffe_cpu_dot(batch_size_ones_.count(),
    batch_size_ones_.cpu_data(), batch_size_ones_.cpu_diff());
  current_log_likelihood_ += batch_log_likelihood;
  if (objective) {
    *objective = batch_log_likelihood / batch_size_;
  }
}

template <typename Dtype>
void GMMLearner<Dtype>::mstep_cpu() {
  // Backup the old value first
  caffe_copy(cluster_weights_.count(), cluster_weights_.cpu_data(), cluster_weights_.mutable_cpu_diff());
  caffe_copy(clusters_.count(), clusters_.cpu_data(), clusters_.mutable_cpu_diff());
  caffe_copy(variances_.count(), variances_.cpu_data(), variances_.mutable_cpu_diff());
  caffe_copy(log_norm_factor_.count(), log_norm_factor_.cpu_data(), log_norm_factor_.mutable_cpu_diff());

  caffe_cpu_inv<Dtype>(num_clusters_, N_.cpu_data(), N_.mutable_cpu_diff());
  // Update the cluster's weights
  if (!soft_kmeans_) { // When using soft kmeans, keep weights uniform!
    Dtype total_n = caffe_cpu_dot(num_clusters_, batch_size_ones_.cpu_data(), N_.cpu_data());
    caffe_cpu_axpby<Dtype>(num_clusters_, Dtype(1) / total_n, N_.cpu_data(),
      Dtype(0), cluster_weights_.mutable_cpu_data());
  }
  // Update the centroids
  caffe_cpu_dgmm<Dtype>(CblasLeft, num_clusters_, dim_, M_.cpu_data(),
            N_.cpu_diff(), clusters_.mutable_cpu_data());
  // Update the variances
  if (!soft_kmeans_) { // When using soft kmeans don't update the variances to keep them at 1.
    caffe_cpu_dgmm<Dtype>(CblasLeft, num_clusters_, dim_, Q_.cpu_data(),
              N_.cpu_diff(), variances_.mutable_cpu_data());
    caffe_sqr<Dtype>(num_clusters_ * dim_, clusters_.cpu_data(), M_.mutable_cpu_diff());
    caffe_axpy<Dtype>(num_clusters_ * dim_, Dtype(-1), M_.cpu_diff(), variances_.mutable_cpu_data());
    caffe_add_scalar<Dtype>(num_clusters_ * dim_, fudge_factor_, variances_.mutable_cpu_data());
  }
  // Keep the old value if the cluster contains a single data-point
  const Dtype* N = N_.cpu_data();
  for (int i = 0; i < N_.count(); ++i) {
    if (N[i] <= 1) {
      LOG(INFO) << "Found single cluster at: " << i;
      caffe_copy(1, cluster_weights_.cpu_diff() + i, cluster_weights_.mutable_cpu_data() + i);
      caffe_copy(dim_, clusters_.cpu_diff() + i * dim_, clusters_.mutable_cpu_data() + i * dim_);
      caffe_copy(dim_, variances_.cpu_diff() + i * dim_, variances_.mutable_cpu_data() + i * dim_);
    }
  }
  // Save the log cluster's weights for later use
  caffe_log<Dtype>(num_clusters_, cluster_weights_.cpu_data(), cluster_weights_.mutable_cpu_diff());
  // Update normalization factor
  if (!soft_kmeans_) { // When using soft kmeans, keep normalization factor uniform!
    caffe_set<Dtype>(log_norm_factor_.count(),
      Dtype(-0.5 * dim_ * log(2.0 * M_PI)), log_norm_factor_.mutable_cpu_data());
    caffe_log<Dtype>(variances_.count(), variances_.cpu_data(), variances_.mutable_cpu_diff());
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_clusters_, dim_,
      Dtype(-0.5), variances_.cpu_diff(), batch_size_ones_.cpu_data(),
      Dtype(1), log_norm_factor_.mutable_cpu_data());
    // Keep old value as above.
    // Done separately so it would be possible to reuse variances_.mutable_cpu_diff()
    for (int i = 0; i < N_.count(); ++i) {
      if (N[i] <= 1) {
        caffe_copy(1, log_norm_factor_.cpu_diff() + i, log_norm_factor_.mutable_cpu_data() + i);
      }
    }
  }
}

template <typename Dtype>
void GMMLearner<Dtype>::split_clusters_cpu() {
  // TODO: finish implementing splitting...
  // From looking at the article splitting isn't critical to convergence
}

template <typename Dtype>
void GMMLearner<Dtype>::tie_variances_cpu() {
  caffe_cpu_gemv<Dtype>(CblasTrans, num_clusters_, dim_,
    Dtype(1), variances_.cpu_data(), cluster_weights_.cpu_data(),
    Dtype(0), variances_.mutable_cpu_diff());
  for (int i = 0; i < num_clusters_; ++i) {
    caffe_copy(dim_, variances_.cpu_diff(), variances_.mutable_cpu_data() + i * dim_);
  }
  caffe_set<Dtype>(log_norm_factor_.count(),
    Dtype(-0.5 * dim_ * log(2.0 * M_PI)), log_norm_factor_.mutable_cpu_data());
  caffe_log<Dtype>(variances_.count(), variances_.cpu_data(), variances_.mutable_cpu_diff());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_clusters_, dim_,
    Dtype(-0.5), variances_.cpu_diff(), batch_size_ones_.cpu_data(),
    Dtype(1), log_norm_factor_.mutable_cpu_data());
}

INSTANTIATE_CLASS(GMMLearner);
}  // namespace caffe
