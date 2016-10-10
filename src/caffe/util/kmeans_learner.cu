#include "caffe/util/unsupervised_learner.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_dist_clusters(const int dim, const int num_clusters, const int batch_size, 
    const Dtype* data, const Dtype* clusters, Dtype* dist, const int max_cluster_to_check) {
  CUDA_KERNEL_LOOP(index, max_cluster_to_check * batch_size) {
    const int c = index % max_cluster_to_check; // cluster index
    const int i = index / max_cluster_to_check; // input index
    Dtype val = 0;
    for (int j = 0; j < dim; ++j) {
      const Dtype x = data[i * dim + j];
      const Dtype cluster = clusters[c * dim + j];
      val += (x - cluster) * (x - cluster);
    }
    dist[i * num_clusters + c] = val;
  }
}

template <typename Dtype>
void dist_clusters_gpu(const int dim, const int num_clusters, const int batch_size, 
    const Dtype* data, const Dtype* clusters, Dtype* dist, const int max_cluster_to_check) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  const int num_kernels = max_cluster_to_check * batch_size;
  kernel_dist_clusters<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      dim, num_clusters, batch_size, data, clusters, dist, max_cluster_to_check);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void kernel_arg_min(const int num_clusters, const int batch_size,
    const Dtype* dist_data, Dtype* assignment_data, const int max_cluster_to_check) {
  CUDA_KERNEL_LOOP(i, batch_size) {
    Dtype min_value = dist_data[i * num_clusters];
    int min_index = 0;
    for (int j = 1; j < max_cluster_to_check; ++j) {
      const Dtype dist = dist_data[i * num_clusters + j];
      min_index = (dist <= min_value) * j + (dist > min_value) * min_index;
      min_value = min(dist, min_value);
    }
    assignment_data[i * num_clusters + min_index] = Dtype(1);
  }
}

template <typename Dtype>
void arg_min_gpu(const int num_clusters, const int batch_size,
    const Dtype* dist_data, Dtype* assignment_data, const int max_cluster_to_check) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_arg_min<Dtype><<<CAFFE_GET_BLOCKS(batch_size), CAFFE_CUDA_NUM_THREADS>>>(
      num_clusters, batch_size, dist_data, assignment_data, max_cluster_to_check);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void KmeansLearner<Dtype>::fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  CHECK_GE(blobs.size(), 1);
  CHECK_LE(blobs.size(), 2);
  CHECK_EQ(blobs[0]->count(), num_clusters_ * dim_);
  caffe_copy(num_clusters_*dim_, clusters_.gpu_data(), blobs[0]->mutable_gpu_data());
  if (blobs.size() == 2) {
    CHECK_EQ(blobs[1]->count(), num_clusters_ * dim_);
    caffe_copy(num_clusters_* dim_, clusters_.gpu_diff(), blobs[1]->mutable_gpu_data());
    caffe_gpu_add_scalar(num_clusters_* dim_, fudge_factor_, blobs[1]->mutable_gpu_data());
  }
}

template <typename Dtype>
bool KmeansLearner<Dtype>::step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective) {
  CHECK_GE(input.size(), 1);
  CHECK(input[0]) << "step_gpu input was null";
  CHECK_GT(input[0]->count(), 0) << "step_gpu input has data";
  if (!called_setup_) {
    this->setup(input);
    called_setup_ = true;
  }
  if (!init_clusters_) {
    if (use_kmeans_plus_plus_) {
      init_clusters_ = sample_kmeans_plus_plus_gpu(input[0]);
    } else {
      init_clusters_ = random_initialization_gpu(input[0]);
    }
    if (!init_clusters_) {
      return true;
    }
  }
  if (iter_ < max_iterations_) {
    assign_to_clusters_gpu(input[0], objective);
    update_clusters_gpu(input[0]);
  } else if (iter_ >= max_iterations_ && iter_ < max_iterations_ + num_batches_) {
    assign_to_clusters_gpu(input[0], NULL);
    update_variances_gpu(input[0]);
  }
  iter_++;
  return iter_ < max_iterations_ + num_batches_;
}

template <typename Dtype>
Dtype KmeansLearner<Dtype>::objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  if (num_init_clusters_ != num_clusters_) {
    return INFINITY;
  }
  Dtype objective;
  assign_to_clusters_gpu(input[0], &objective);
  return objective;
}

template <typename Dtype>
bool KmeansLearner<Dtype>::random_initialization_gpu(const shared_ptr<Blob<Dtype> >& input) {
  if (num_init_clusters_ >= num_clusters_) return true;
  const Dtype* data = input->gpu_data();
  Dtype* clusters = clusters_.mutable_gpu_data();
  std::vector<int> coin_tosses;
  coin_tosses.resize(batch_size_);
  caffe_rng_bernoulli(batch_size_, prob_choose_centroid_, &coin_tosses[0]);
  for (int i = 0; i < batch_size_ && num_init_clusters_ < num_clusters_; ++i) {
    if (coin_tosses[i] == 1) {
      caffe_copy(dim_, data + i * dim_, clusters + num_init_clusters_ * dim_);
      ++num_init_clusters_;
    }
  }
  return num_init_clusters_ >= num_clusters_;
}
template <typename Dtype>
bool KmeansLearner<Dtype>::sample_kmeans_plus_plus_gpu(const shared_ptr<Blob<Dtype> >& input) {
  if (num_init_clusters_ >= num_clusters_) return true;
  const Dtype* data = input->gpu_data();
  Dtype* clusters = clusters_.mutable_gpu_data();
  // Init the first cluster randomly.
  if (num_init_clusters_ == 0) {
    std::vector<int> coin_tosses;
    coin_tosses.resize(batch_size_);
    caffe_rng_bernoulli(batch_size_, prob_choose_centroid_, &coin_tosses[0]);
    for (int i = 0; i < batch_size_; ++i) {
      if (coin_tosses[i] == 1) {
        caffe_copy(dim_, data + i * dim_, clusters + num_init_clusters_ * dim_);
        ++num_init_clusters_;
        break;
      }
    }
  } else { // pick the other k-1 cluster with probability relative to their distance to nearest cluster
    Dtype* dist_data = assignments_.mutable_gpu_diff();
    dist_clusters_gpu(dim_, num_clusters_, batch_size_, data, clusters, dist_data, num_init_clusters_);
    Dtype* assignment_data = assignments_.mutable_gpu_data();
    caffe_gpu_set(batch_size_*num_clusters_, Dtype(0), assignment_data);
    arg_min_gpu(num_clusters_, batch_size_, dist_data, assignment_data, num_init_clusters_);

    Dtype total_distances;
    caffe_gpu_dot(assignments_.count(), dist_data, assignment_data, &total_distances);
    caffe_gpu_mul(batch_size_*num_clusters_, assignment_data, dist_data, dist_data);
    // Reuse assignment data for distance per input.
    caffe_gpu_gemv<Dtype>(CblasNoTrans, batch_size_, num_clusters_,
      1., dist_data,
      batch_size_ones_.gpu_data(), 1,
      assignment_data);
    assignment_data = NULL; // So we won't accidentally use the GPU pointer.
    const Dtype* distances = assignments_.cpu_data(); // move data to host
    for (int i = 0; i < batch_size_; ++i) {
      int coin_toss;
      caffe_rng_bernoulli(1, distances[i] / total_distances, &coin_toss);
      if (coin_toss == 1) {
        caffe_copy(dim_, data + i * dim_, clusters + num_init_clusters_ * dim_);
        ++num_init_clusters_;
        LOG(INFO) << "Picked init cluster number " << num_init_clusters_ << " of " << num_clusters_ << " with relative distance: " << distances[i] * batch_size_ / total_distances;
        break;
      }
    }
  }
  return num_init_clusters_ >= num_clusters_;
}

template <typename Dtype>
void KmeansLearner<Dtype>::assign_to_clusters_gpu(const shared_ptr<Blob<Dtype> >& input, Dtype* objective) {
  const Dtype* data = input->gpu_data();
  const Dtype* clusters_data = clusters_.gpu_data();
  Dtype* dist_data = assignments_.mutable_gpu_diff();
  dist_clusters_gpu(dim_, num_clusters_, batch_size_, data, clusters_data, dist_data, num_clusters_);
  Dtype* assignment_data = assignments_.mutable_gpu_data();
  caffe_gpu_set(batch_size_*num_clusters_, Dtype(0), assignment_data);
  arg_min_gpu(num_clusters_, batch_size_, dist_data, assignment_data, num_clusters_);
  if (objective) {
    caffe_gpu_dot(assignments_.count(), dist_data, assignment_data, objective);
    // const Dtype* d = input->cpu_data();
    // LOG(INFO) << "sample data: " << d[0] << ", " << d[1] << ", " << d[2];
    // const Dtype* c = clusters_.cpu_data();
    // LOG(INFO) << "sample cluster: " << c[0] << ", " << c[1] << ", " << c[2];
    // const Dtype* di = assignments_.cpu_diff();
    // LOG(INFO) << "sample dist: " << di[0] << ", " << di[1] << ", " << di[2];
    *objective = *objective / batch_size_;
  }
}

template <typename Dtype>
void KmeansLearner<Dtype>::update_clusters_gpu(const shared_ptr<Blob<Dtype> >& input) {
  // Scales the old clusters for computing the weighted average
  caffe_gpu_dgmm<Dtype>(CblasLeft, num_clusters_, dim_, clusters_.mutable_gpu_data(),
            per_center_count_.gpu_data(), clusters_.mutable_gpu_data());
  // Compute new weighted sum
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_clusters_, dim_, batch_size_,
              (Dtype)1., assignments_.gpu_data(), input->gpu_data(),
              (Dtype)1, clusters_.mutable_gpu_data());
  // update the count of points for the new clusters
  caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, num_clusters_,
            1., assignments_.gpu_data(),
            batch_size_ones_.gpu_data(), 1,
            per_center_count_.mutable_gpu_data());
  // prepare normalizing by the new point count
  caffe_gpu_set(num_clusters_, Dtype(1), per_center_count_.mutable_gpu_diff());
  caffe_gpu_div(num_clusters_, per_center_count_.gpu_diff(), per_center_count_.gpu_data(),
                           per_center_count_.mutable_gpu_diff());
  // Normalize result by the new number of points
  caffe_gpu_dgmm<Dtype>(CblasLeft, num_clusters_, dim_, clusters_.gpu_data(),
            per_center_count_.gpu_diff(), clusters_.mutable_gpu_data());
  // const Dtype* c = per_center_count_.cpu_data();
  // std::ostringstream oss;
  // Dtype total = 0;
  // for (int i = 0; i < num_clusters_; ++i) {
  //   oss << c[i] << ", ";
  //   total += c[i];
  // }
  // LOG(INFO) << "Counts: " << oss.str();
  // LOG(INFO) << "Total: " << total;
}

template <typename Dtype>
void KmeansLearner<Dtype>::update_variances_gpu(const shared_ptr<Blob<Dtype> >& input) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, dim_, num_clusters_,
    Dtype(-1.0), assignments_.gpu_data(), clusters_.gpu_data(),
    (Dtype)1, input->mutable_gpu_data());
  caffe_gpu_mul(batch_size_ * dim_, input->mutable_gpu_data(), input->mutable_gpu_data(), input->mutable_gpu_data());
  const int var_iter = iter_ - max_iterations_;
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_clusters_, dim_, batch_size_,
    (Dtype)(1.0 / ((var_iter + 1) * batch_size_ - 1.0)), assignments_.gpu_data(), input->gpu_data(),
    (Dtype)((var_iter * batch_size_ - 1.0) / ((var_iter + 1) * batch_size_ - 1.0)), clusters_.mutable_gpu_diff());
}

template void KmeansLearner<float>::fill_gpu(const vector<shared_ptr<Blob<float> > >& blobs);
template void KmeansLearner<double>::fill_gpu(const vector<shared_ptr<Blob<double> > >& blobs);
template bool KmeansLearner<float>::step_gpu(const vector<shared_ptr<Blob<float> > >& input, float* objective);
template bool KmeansLearner<double>::step_gpu(const vector<shared_ptr<Blob<double> > >& input, double* objective);
template float KmeansLearner<float>::objective_gpu(const vector<shared_ptr<Blob<float> > >& input);
template double KmeansLearner<double>::objective_gpu(const vector<shared_ptr<Blob<double> > >& input);
}  // namespace caffe
