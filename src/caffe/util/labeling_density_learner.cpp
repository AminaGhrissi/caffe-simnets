#include "caffe/util/unsupervised_learner.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void LabelingDensityLearner<Dtype>::setup(const vector<shared_ptr<Blob<Dtype> > >& input) {
  batch_size_ = input[0]->num();
  dim_ = input[0]->channels();
  CHECK_EQ(input[0]->width(), 1);
  CHECK_EQ(input[0]->height(), 1);
  CHECK_GT(num_labels_, 0);
  CHECK_GT(num_batches_, 0);
  CHECK_GT(dim_, 0);
  CHECK_EQ(input[1]->count(), batch_size_);
  CHECK_EQ(input[1]->num(), batch_size_);
  CHECK_LT(fudge_factor_, 0);
  CHECK_GE(lambda_, 0);

  densities_.Reshape(num_labels_, dim_, 1, 1);
  if (lambda_ > 0) {
    per_label_densities_.Reshape(num_labels_, batch_size_, dim_, 1);
  }
  per_label_count_.Reshape(num_labels_, 1, 1, 1);
  sum_multiplier_.Reshape(batch_size_, 1, 1, 1); // TODO: do we need it??
  if (Caffe::mode() == Caffe::CPU) {
    caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
    caffe_set(densities_.count(), Dtype(0), densities_.mutable_cpu_data());
    caffe_set(densities_.count(), Dtype(std::log(1.0 / dim_)), densities_.mutable_cpu_diff());
    caffe_set(per_label_count_.count(), Dtype(0), per_label_count_.mutable_cpu_data());
  } else {
    caffe_gpu_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_gpu_data());
    caffe_gpu_set(densities_.count(), Dtype(0), densities_.mutable_gpu_data());
    caffe_gpu_set(densities_.count(), Dtype(std::log(1.0 / dim_)), densities_.mutable_gpu_diff());
    caffe_gpu_set(per_label_count_.count(), Dtype(0), per_label_count_.mutable_gpu_data());
  }
}

template <typename Dtype>
bool LabelingDensityLearner<Dtype>::step_cpu(const vector<shared_ptr<Blob<Dtype> > >&  input, Dtype* objective) {
  CHECK_GE(input.size(), 2);
  CHECK(input[0] && input[1]) << "step_cpu input was null";
  CHECK_GT(input[0]->count(), 0) << "step_cpu patch input has data";
  CHECK_GT(input[1]->count(), 0) << "step_cpu label input has data";
  if (objective) {
    *objective = INFINITY;
  }
  if (!called_setup_) {
    this->setup(input);
    called_setup_ = true;
  }

  update_densities_cpu(input);
  iter_++;

  if (iter_ % num_batches_ == 0) {
    caffe_copy(densities_.count(), densities_.cpu_data(), densities_.mutable_cpu_diff());
    caffe_set(densities_.count(), Dtype(0), densities_.mutable_cpu_data());
    caffe_set(per_label_count_.count(), Dtype(0), per_label_count_.mutable_cpu_data());
  }

  return iter_ / num_batches_ < max_iterations_;
}

template <typename Dtype>
void LabelingDensityLearner<Dtype>::fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  CHECK_EQ(blobs.size(), 1);
  CHECK_EQ(blobs[0]->count(), num_labels_ * dim_);
  caffe_copy(num_labels_ * dim_, densities_.cpu_diff(), blobs[0]->mutable_cpu_data());
}

template <typename Dtype>
Dtype LabelingDensityLearner<Dtype>::objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  return INFINITY;
}

template <typename Dtype>
void normalize_data_cpu(const int N, const int dim, const bool soft_assignment, const Dtype fudge_factor,
  const Dtype* labels, const Dtype* priors, Dtype* data) {
  for (int n = 0; n < N; ++n) {
    const int label = static_cast<int>(labels[n]);
    Dtype m = data[n * dim] + priors[label * dim];
    int m_idx = 0;
    for (int k = 1; k < dim; ++k) {
      const Dtype x = data[n * dim + k] + priors[label * dim + k];
      if (m <= x) {
        m = x;
        m_idx = k;
      }
    }
    if (soft_assignment) {
      Dtype sum = 0;
      for (int k = 0; k < dim; ++k) {
        sum += std::exp(data[n * dim + k] + priors[label * dim + k] - m);
      }
      const Dtype softmax = std::log(sum) + m;
      for (int k = 0; k < dim; ++k) {
        data[n * dim + k] += priors[label * dim + k] - softmax;
      }
    } else {
      for (int k = 0; k < dim; ++k) {
        data[n * dim + k] = fudge_factor; // TODO: maybe switch to log(fudge_factor) ??
      }
      data[n * dim + m_idx] = 0; // TODO: maybe switch to log(1.0 - fudge_factor) ??
    }
  }
}

template <typename Dtype>
void normalize_data_per_label_cpu(const int num_labels, const int N, const int dim,
  const bool soft_assignment, const Dtype fudge_factor,
  const Dtype* priors, const Dtype* data, Dtype* densities) {
  for (int n = 0; n < N; ++n) {
    for (int label = 0; label < num_labels; ++label) {
      Dtype m = data[n * dim] + priors[label * dim];
      int m_idx = 0;
      for (int k = 1; k < dim; ++k) {
        const Dtype x = data[n * dim + k] + priors[label * dim + k];
        if (m <= x) {
          m = x;
          m_idx = k;
        }
      }
      if (soft_assignment) {
        Dtype sum = 0;
        for (int k = 0; k < dim; ++k) {
          sum += std::exp(data[n * dim + k] + priors[label * dim + k] - m);
        }
        const Dtype softmax = std::log(sum) + m;
        for (int k = 0; k < dim; ++k) {
          densities[(label* N + n) * dim + k] = data[n * dim + k] + priors[label * dim + k] - softmax;
        }
      } else {
        for (int k = 0; k < dim; ++k) {
          densities[(label* N + n) * dim + k] = fudge_factor; // TODO: maybe switch to log(fudge_factor) ??
        }
        densities[(label* N + n) * dim + m_idx] = 0; // TODO: maybe switch to log(1.0 - fudge_factor) ??
      }
    }
  }
}

template <typename Dtype>
void LabelingDensityLearner<Dtype>::update_densities_cpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  const Dtype* data = input[0]->cpu_data();
  const Dtype* labels = input[1]->cpu_data();

  const Dtype* priors = densities_.cpu_diff();
  Dtype* patch_densities = input[0]->mutable_cpu_data();
  Dtype* per_label_densities = per_label_densities_.mutable_cpu_data();
  if (lambda_ == 0) {
    caffe_copy(input[0]->count(), data, patch_densities);
    normalize_data_cpu(batch_size_, dim_, soft_assignment_, fudge_factor_, labels, priors, patch_densities);
  } else {
    normalize_data_per_label_cpu(num_labels_, batch_size_, dim_, soft_assignment_, fudge_factor_,
      priors, data, per_label_densities_.mutable_cpu_data());
  }

  // Count how many patches per label
  Dtype* batch_labels_count = per_label_count_.mutable_cpu_diff();
  caffe_set(num_labels_, Dtype(0), batch_labels_count);
  for (int n = 0; n < batch_size_; ++n) {
    const int label = static_cast<int>(labels[n]);
    batch_labels_count[label] += Dtype(1.0);
  }

  Dtype* densities = densities_.mutable_cpu_data();
  Dtype* labels_count = per_label_count_.mutable_cpu_data();
  Dtype* temp_densities = densities_.mutable_cpu_diff();
  caffe_set(densities_.count(), Dtype(0), temp_densities);
  if (lambda_) {
    for (int k = 0; k < dim_; ++k) {
      Dtype m[num_labels_];
      caffe_set<Dtype>(num_labels_, Dtype(-INFINITY), m);
      for (int n = 0; n < batch_size_; ++n) {
        const int label = static_cast<int>(labels[n]);
        m[label] = std::max(m[label], patch_densities[n * dim_ + k]);
      }
      for (int n = 0; n < batch_size_; ++n) {
        const int label = static_cast<int>(labels[n]);
        temp_densities[label * dim_ + k] += std::exp(patch_densities[n * dim_ + k] - m[label]);
      }
      for (int i = 0; i < num_labels_; ++i) {
        if (batch_labels_count[i] > 0) {
          const int batch_count = batch_labels_count[i];
          const int old_count = labels_count[i];
          const Dtype batch_density = std::log(temp_densities[i * dim_ + k] / batch_count) + m[i];
          if (old_count == 0) {
            densities[i * dim_ + k] = batch_density;
          } else {
            const int new_count = old_count + batch_count;
            const Dtype log_ratio_old = std::log(old_count / Dtype(new_count));
            const Dtype log_ratio_batch = std::log(batch_count / Dtype(new_count));
            const Dtype old_density = densities[i * dim_ + k];
            const Dtype x1 = batch_density + log_ratio_batch;
            const Dtype x2 = old_density + log_ratio_old;
            const Dtype m2 = std::max(x1, x2);
            densities[i * dim_ + k] = std::log(std::exp(x1 - m2) + std::exp(x2 - m2)) + m2;
          }
        }
      }
    }
  } else {
    for (int k = 0; k < dim_; ++k) {
      for (int l = 0; l < num_labels_; ++l) {
        const int batch_count = batch_labels_count[l];
        if (batch_count == 0) continue;
        const Dtype correct_bias = std::log(1.0 / batch_count);
        const Dtype incorrect_bias = std::log(lambda_ / (batch_size_ - batch_count));
        Dtype m = -INFINITY;
        for (int n = 0; n < batch_size_; ++n) {
          const int label = static_cast<int>(labels[n]);
          const Dtype x = per_label_densities[(l * batch_size_ + n) * dim_ + k];
          if (label == l) {
            m = std::max(m, x + correct_bias);
          } else {
            m = std::max(m, x + incorrect_bias);
          }
        }

        Dtype sum1 = 0, sum2 = 0;
        for (int n = 0; n < batch_size_; ++n) {
          const int label = static_cast<int>(labels[n]);
          const Dtype x = per_label_densities[(l * batch_size_ + n) * dim_ + k];
          if (label == l) {
            sum1 += std::exp(x + correct_bias - m);
          } else {
            sum2 += std::exp(x + incorrect_bias - m);
          }
        }

        
        const int old_count = labels_count[l];
        const Dtype batch_density = std::log((sum1 - sum2) / (Dtype(1.0) - lambda_)) + m;
        if (old_count == 0) {
          densities[l * dim_ + k] = batch_density;
        } else {
          const int new_count = old_count + batch_count;
          const Dtype log_ratio_old = std::log(old_count / Dtype(new_count));
          const Dtype log_ratio_batch = std::log(batch_count / Dtype(new_count));
          const Dtype old_density = densities[l * dim_ + k];
          const Dtype x1 = batch_density + log_ratio_batch;
          const Dtype x2 = old_density + log_ratio_old;
          const Dtype m2 = std::max(x1, x2);
          densities[l * dim_ + k] = std::log(std::exp(x1 - m2) + std::exp(x2 - m2)) + m2;
        }
      }
    }
  }
  for (int i = 0; i < num_labels_; ++i) {
    labels_count[i] += batch_labels_count[i];
  }
}

#ifdef CPU_ONLY
template <typename Dtype>
bool LabelingDensityLearner<Dtype>::step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  NO_GPU;
}
template <typename Dtype>
void LabelingDensityLearner<Dtype>::fill_gpu(const vector<Blob<Dtype>* >& blobs) {
  NO_GPU;
}

template <typename Dtype>
Dtype LabelingDensityLearner<Dtype>::objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  NO_GPU;
}

#endif

INSTANTIATE_CLASS(LabelingDensityLearner);
}  // namespace caffe
