#include "caffe/util/unsupervised_learner.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void KmeansLearner<Dtype>::setup(const vector<shared_ptr<Blob<Dtype> > >& input) {
  batch_size_ = input[0]->num();
  dim_ = input[0]->channels();
  CHECK_EQ(input[0]->width(), 1);
  CHECK_EQ(input[0]->height(), 1);
  CHECK_GT(num_clusters_, 0);
  assignments_.Reshape(batch_size_, num_clusters_, 1, 1);
  clusters_.Reshape(num_clusters_, dim_, 1, 1);
  per_center_count_.Reshape(num_clusters_, 1, 1, 1);
  init_clusters_ = false;
  batch_size_ones_.Reshape(batch_size_, 1, 1, 1);
  if (Caffe::mode() == Caffe::CPU) {
    caffe_set(batch_size_, Dtype(1), batch_size_ones_.mutable_cpu_data());
    caffe_set(num_clusters_, Dtype(1), per_center_count_.mutable_cpu_data());
    caffe_set(num_clusters_ * dim_, Dtype(0), clusters_.mutable_cpu_diff());
  } else {
    caffe_gpu_set(batch_size_, Dtype(1), batch_size_ones_.mutable_gpu_data());
    caffe_gpu_set(num_clusters_, Dtype(1), per_center_count_.mutable_gpu_data());
    caffe_gpu_set(num_clusters_ * dim_, Dtype(0), clusters_.mutable_gpu_diff());
  }
}

template <typename Dtype>
bool KmeansLearner<Dtype>::step_cpu(const vector<shared_ptr<Blob<Dtype> > >&  input, Dtype* objective) {
  CHECK_GE(input.size(), 1);
  CHECK(input[0]) << "step_cpu input was null";
  CHECK_GT(input[0]->count(), 0) << "step_cpu input has data";
  if (!called_setup_) {
    this->setup(input);
    called_setup_ = true;
  }
  if (!init_clusters_) {
    if (use_kmeans_plus_plus_) {
      init_clusters_ = sample_kmeans_plus_plus_cpu(input[0]);
    } else {
      init_clusters_ = random_initialization_cpu(input[0]);
    }
    if (!init_clusters_) {
      return true;
    }
  }
  if (iter_ < max_iterations_) {
    assign_to_clusters_cpu(input[0], objective);
    update_clusters_cpu(input[0]);
  } else if (iter_ >= max_iterations_ && iter_ < max_iterations_ + num_batches_) {
    assign_to_clusters_cpu(input[0], NULL);
    update_variances_cpu(input[0]);
  }
  iter_++;
  return iter_ < max_iterations_ + num_batches_;
}

template <typename Dtype>
void KmeansLearner<Dtype>::fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  CHECK_GE(blobs.size(), 1);
  CHECK_LE(blobs.size(), 2);
  CHECK_EQ(blobs[0]->count(), num_clusters_ * dim_);
  caffe_copy(num_clusters_*dim_, clusters_.cpu_data(), blobs[0]->mutable_cpu_data());
  if (blobs.size() == 2) {
    CHECK_EQ(blobs[1]->count(), num_clusters_ * dim_);
    caffe_copy(num_clusters_* dim_, clusters_.cpu_diff(), blobs[1]->mutable_cpu_data());
    caffe_add_scalar(num_clusters_* dim_, fudge_factor_, blobs[1]->mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
template <typename Dtype>
bool KmeansLearner<Dtype>::step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  NO_GPU;
}
template <typename Dtype>
void KmeansLearner<Dtype>::fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  NO_GPU;
}
#endif

template <typename Dtype>
bool KmeansLearner<Dtype>::random_initialization_cpu(const shared_ptr<Blob<Dtype> >& input) {
  const Dtype* data = input->cpu_data();
  Dtype* clusters = clusters_.mutable_cpu_data();
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
bool KmeansLearner<Dtype>::sample_kmeans_plus_plus_cpu(const shared_ptr<Blob<Dtype> >& input) {
  const Dtype* data = input->cpu_data();
  Dtype* clusters = clusters_.mutable_cpu_data();
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
    std::vector<Dtype> distances;
    distances.resize(batch_size_);
    Dtype total_distances = 0;
    for (int i = 0; i <  batch_size_; ++i) {
      Dtype min = INFINITY;
      for (int c = 0; c < num_init_clusters_; ++c) {
        Dtype dist = 0;
        for (int j = 0; j < dim_; ++j) {
          dist += (data[i * dim_ + j] - clusters[c * dim_ + j]) 
                * (data[i * dim_ + j] - clusters[c * dim_ + j]);
        }
        if (dist < min) {
          min = dist;
        }
      }
      distances[i] = min;
      total_distances += min;
    }
    for (int i = 0; i < batch_size_; ++i) {
      int coin_toss;
      caffe_rng_bernoulli(1, distances[i] / total_distances, &coin_toss);
      if (coin_toss == 1) {
        caffe_copy(dim_, data + i * dim_, clusters + num_init_clusters_ * dim_);
        ++num_init_clusters_;
        break;
      }
    }
  }
  return num_init_clusters_ >= num_clusters_;
}
template <typename Dtype>
void KmeansLearner<Dtype>::assign_to_clusters_cpu(const shared_ptr<Blob<Dtype> >& input, Dtype* objective_out) {
  const Dtype* data = input->cpu_data();
  const Dtype* clusters_data = clusters_.cpu_data();
  Dtype* assignment_data = assignments_.mutable_cpu_data();
  caffe_set(batch_size_*num_clusters_, Dtype(0), assignment_data);
  Dtype objective = 0;
  for (int i = 0; i <  batch_size_; ++i) {
    Dtype min = INFINITY;
    int min_idx = -1;
    for (int c = 0; c < num_clusters_; ++c) {
      Dtype dist = 0;
      for (int j = 0; j < dim_; ++j) {
        dist += (data[i * dim_ + j] - clusters_data[c * dim_ + j]) 
              * (data[i * dim_ + j] - clusters_data[c * dim_ + j]);
      }
      if (dist <= min) {
        min = dist;
        min_idx = c;
      }
    }
    objective += min;
    assignment_data[i * num_clusters_ + min_idx] = Dtype(1.0);
  }
  objective = objective / batch_size_;
  if (objective_out) {
    *objective_out = objective;
  }
}

template <typename Dtype>
Dtype KmeansLearner<Dtype>::objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  if (num_init_clusters_ != num_clusters_) {
    return INFINITY;
  }
  Dtype objective;
  assign_to_clusters_cpu(input[0], &objective);
  return objective;
}


template <typename Dtype>
void KmeansLearner<Dtype>::update_clusters_cpu(const shared_ptr<Blob<Dtype> >& input) {
  // Scales the old clusters for computing the weighted average
  caffe_cpu_dgmm<Dtype>(CblasLeft, num_clusters_, dim_, clusters_.mutable_cpu_data(),
            per_center_count_.cpu_data(), clusters_.mutable_cpu_data());
  // Compute new weighted sum
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_clusters_, dim_, batch_size_,
              (Dtype)1., assignments_.cpu_data(), input->cpu_data(),
              (Dtype)1, clusters_.mutable_cpu_data());
  // update the count of points for the new clusters
  caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, num_clusters_,
            1., assignments_.cpu_data(),
            batch_size_ones_.cpu_data(), 1,
            per_center_count_.mutable_cpu_data());
  // prepare normalizing by the new point count
  caffe_set(num_clusters_, Dtype(1), per_center_count_.mutable_cpu_diff());
  caffe_div(num_clusters_, per_center_count_.mutable_cpu_diff(), per_center_count_.mutable_cpu_data(),
                           per_center_count_.mutable_cpu_diff());
  // Normalize result by the new number of points
  caffe_cpu_dgmm<Dtype>(CblasLeft, num_clusters_, dim_, clusters_.mutable_cpu_data(),
            per_center_count_.cpu_diff(), clusters_.mutable_cpu_data());
}

template <typename Dtype>
void KmeansLearner<Dtype>::update_variances_cpu(const shared_ptr<Blob<Dtype> >& input) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, dim_, num_clusters_,
    Dtype(-1.0), assignments_.cpu_data(), clusters_.cpu_data(),
    (Dtype)1, input->mutable_cpu_data());
  caffe_mul(batch_size_ * dim_, input->mutable_cpu_data(), input->mutable_cpu_data(), input->mutable_cpu_data());
  const int var_iter = iter_ - max_iterations_;
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_clusters_, dim_, batch_size_,
    (Dtype)(1.0 / ((var_iter + 1) * batch_size_ - 1.0)), assignments_.cpu_data(), input->cpu_data(),
    (Dtype)((var_iter * batch_size_ - 1.0) / ((var_iter + 1) * batch_size_ - 1.0)), clusters_.mutable_cpu_diff());
}

INSTANTIATE_CLASS(KmeansLearner);
}  // namespace caffe
