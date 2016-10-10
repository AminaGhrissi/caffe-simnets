#include "caffe/util/unsupervised_learner.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void PCALearner<Dtype>::setup(const vector<shared_ptr<Blob<Dtype> > >& input) {
  CHECK_GE(input.size(), 1);
  batch_size_ = input[0]->num();
  dim_ = input[0]->channels();
  CHECK_EQ(input[0]->width(), 1);
  CHECK_EQ(input[0]->height(), 1);
  CHECK_GT(out_dim_, 0);
  CHECK_LE(out_dim_, dim_) << "The output dimension of PCA must be at most the input dimension.";
  if (zca_whitening_) {
    CHECK_EQ(out_dim_, dim_) << "ZCA whitening can't be used with dimension reduction.";
  }
  CHECK_GT(num_batches_, 0);
  mean_.Reshape(dim_, 1, 1, 1);
  cov_.Reshape(dim_, dim_, 1, 1);
  P_.Reshape(out_dim_, dim_, 1, 1);
  sum_multiplier_.Reshape(batch_size_, 1, 1, 1);
  if (Caffe::mode() == Caffe::CPU) {
    caffe_set(batch_size_, Dtype(1), sum_multiplier_.mutable_cpu_data());
    caffe_set(dim_ * dim_, Dtype(0), cov_.mutable_cpu_data());
  } else {
    caffe_gpu_set(batch_size_, Dtype(1), sum_multiplier_.mutable_gpu_data());
    caffe_gpu_set(dim_ * dim_, Dtype(0), cov_.mutable_gpu_data());
  }
}

template <typename Dtype>
bool PCALearner<Dtype>::step_cpu(const vector<shared_ptr<Blob<Dtype> > >&  input, Dtype* objective) {
  CHECK_GE(input.size(), 1);
  CHECK(input[0]) << "step_cpu input was null";
  CHECK_GT(input[0]->count(), 0) << "step_cpu input has data";
  if (objective) {
    *objective = INFINITY;
  }
  if (!called_setup_) {
    this->setup(input);
    called_setup_ = true;
  }
  if (!calculated_mean_) {
    update_mean_cpu(input[0]);
    iter_++;
    calculated_mean_ = iter_ >= num_batches_;
    return true;
  }
  if (iter_ < num_batches_ * 2) {
    update_covariance_cpu(input[0]);
    iter_++;
  }
  if (iter_ == num_batches_ * 2) {
    calc_pca_cpu();
  }
  return iter_ < num_batches_ * 2;
}

template <typename Dtype>
void PCALearner<Dtype>::fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) {
  CHECK_EQ(blobs.size(), 2);
  CHECK_EQ(blobs[0]->count(), out_dim_ * dim_);
  CHECK_EQ(blobs[1]->count(), out_dim_);
  caffe_copy(out_dim_ * dim_, P_.cpu_data(), blobs[0]->mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasNoTrans, out_dim_, dim_,
    Dtype(-1.0), P_.cpu_data(), mean_.cpu_data(),
    Dtype(0.0), blobs[1]->mutable_cpu_data());
}

template <typename Dtype>
Dtype PCALearner<Dtype>::objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  return INFINITY;
}

template <typename Dtype>
void PCALearner<Dtype>::update_mean_cpu(const shared_ptr<Blob<Dtype> >& input) {
  const Dtype* data = input->cpu_data();
  caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, dim_,
    Dtype(1.0 / ((iter_ + 1) * batch_size_)), data,
    sum_multiplier_.cpu_data(), Dtype(iter_ / (iter_ + 1.0)),
    mean_.mutable_cpu_data());
}

template <typename Dtype>
void PCALearner<Dtype>::update_covariance_cpu(const shared_ptr<Blob<Dtype> >& input) {
  Dtype* data = input->mutable_cpu_data();
  // Subtract the mean from data
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size_, dim_, 1,
    (Dtype)(-1.0), sum_multiplier_.cpu_data(), mean_.cpu_data(),
    (Dtype)1., data);
  // Construct partial covariance matrix
  const int cov_iter = iter_ - num_batches_;
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim_, dim_, batch_size_,
    (Dtype)(1.0 / ((cov_iter + 1) * batch_size_ - 1.0)), data, data,
    (Dtype)((cov_iter * batch_size_ - 1.0) / ((cov_iter + 1) * batch_size_ - 1.0)), cov_.mutable_cpu_data());
}

template <typename Dtype>
void PCALearner<Dtype>::calc_pca_cpu() {
  const Dtype* X = cov_.cpu_data();
  Dtype* R = cov_.mutable_cpu_diff();
  Dtype* U = P_.mutable_cpu_data();
  Dtype* V = P_.mutable_cpu_diff();
  Blob<Dtype> lambda, temp;
  lambda.Reshape(out_dim_, 1, 1, 1);
  Dtype* L = lambda.mutable_cpu_data();
  temp.Reshape(out_dim_, 1, 1, 1);
  Dtype* A = temp.mutable_gpu_data();
  Dtype* B = temp.mutable_gpu_diff();
  // Using the GS-PCA algorithm as presented in
  // Parallel GPU Implementation of Iterative PCA Algorithms (2008)
  // by M. Andrecut

  // input: X, MxN matrix (data)
  // input: M = number of rows in X
  // input: N = number of columns in X
  // input: K = number of components (K<=N)
  // output: T, MxK scores matrix // output: P, NxK loads matrix // output: R, MxN residual matrix
  // Note: remeber that BLAS expects column-major matrices vs. Caffe's row-major matrices.

  const int J = 10000; // max number of powers... TODO: switch to parameter
  const int min_J = 0; // min number of powers...
  CHECK_LE(min_J, J);
  Dtype er = 1.0e-7; // max error
  caffe_copy(dim_ * dim_, X, R);
  for (int k = 0; k < out_dim_; ++k) {
    Dtype mu = 0.0;
    caffe_copy(dim_, R + k * dim_, V + k * dim_);
    int j;
    for (j = 0; j < J; ++j) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, dim_, dim_,
        Dtype(1.0), R, V + k * dim_,
        Dtype(0.0), U + k * dim_);
      //cublasDgemv (’t’, dim_, dim_, 1.0, dR, dim_, &dT[k*dim_], 1, 0.0, &dP[k*dim_], 1);
      if (k > 0) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, k, dim_,
          Dtype(1.0), U, U + k * dim_,
          Dtype(0.0), A);
        caffe_cpu_gemv<Dtype>(CblasTrans, k, dim_,
          Dtype(-1.0), U, A,
          Dtype(1.0), U + k * dim_);
        // cublasDgemv (’t’, dim_, k, 1.0, dP, dim_, &dP[k*dim_], 1, 0.0, dU, 1);
        // cublasDgemv (’n’, dim_, k, -1.0, dP, dim_, dU, 1, 1.0, &dP[k*dim_], 1);
      }
      caffe_scal(dim_, Dtype(1.0 / caffe_cpu_nrm2(dim_, U + k * dim_)), U + k * dim_);
      // cublasDscal (dim_, 1.0/cublasDnrm2(dim_, &dP[k*dim_], 1), &dP[k*dim_], 1);
      caffe_cpu_gemv<Dtype>(CblasTrans, dim_, dim_,
        Dtype(1.0), R, U + k * dim_,
        Dtype(0.0), V + k * dim_);
      // cublasDgemv (’n’, dim_, dim_, 1.0, dR, dim_, &dP[k*dim_], 1, 0.0, &dT[k*dim_], 1);
      if (k > 0) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, k, dim_,
          Dtype(1.0), V, V + k * dim_,
          Dtype(0.0), B);
        caffe_cpu_gemv<Dtype>(CblasTrans, k, dim_,
          Dtype(-1.0), V, B,
          Dtype(1.0), V + k * dim_);
        // cublasDgemv (’t’, dim_, k, 1.0, dT, dim_, &dT[k*dim_], 1, 0.0, dU, 1);
        // cublasDgemv (’n’, dim_, k, -1.0, dT, dim_, dU, 1, 1.0, &dT[k*dim_], 1);
      }
      L[k] = caffe_cpu_nrm2(dim_, V + k * dim_);
      // L[k] = cublasDnrm2(dim_, &dT[k*dim_], 1);
      caffe_scal(dim_, Dtype(1.0/L[k]), V + k * dim_);
      // cublasDscal(dim_, 1.0/L[k], &dT[k*dim_], 1);
      if (fabs(L[k] - mu) < er * L[k] && j >= min_J) {
        break;
      }
      mu = L[k];
    }
    LOG(INFO) << "Lambda #" << k << ": " << L[k] << " (" << j << " iterations)";
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, dim_, dim_, 1,
      -L[k], V + k * dim_, U + k * dim_,
      (Dtype)1., R);
    // cublasDger (dim_, dim_, - L[k], &dT[k*dim_], 1, &dP[k*dim_], 1, dR, dim_);
  }
  // Calculate P = (D^-0.5) * U^T if whitening is needed
  if (apply_whitening_) {
    if (zca_whitening_) {
      caffe_copy(dim_ * dim_, U, V);
    }
    for (int k = 0; k < out_dim_; k++) {
      caffe_scal(dim_, Dtype(1.0 / sqrt(L[k] + fudge_factor_)), U + k * dim_);
    }
    if (zca_whitening_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim_, dim_, dim_,
        Dtype(1.0), V, U,
        Dtype(0.0), R);
      caffe_copy(dim_ * dim_, R, U);
    }
  }
}

#ifdef CPU_ONLY
template <typename Dtype>
bool PCALearner<Dtype>::step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  NO_GPU;
}
template <typename Dtype>
void PCALearner<Dtype>::fill_gpu(const vector<Blob<Dtype>* >& blobs) {
  NO_GPU;
}

template <typename Dtype>
Dtype PCALearner<Dtype>::objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) {
  NO_GPU;
}

template <typename Dtype>
void PCALearner<Dtype>::update_mean_gpu(const shared_ptr<Blob<Dtype> >& input) {
  NO_GPU;
}

template <typename Dtype>
void PCALearner<Dtype>::update_covariance_gpu(const shared_ptr<Blob<Dtype> >& input) {
  NO_GPU;
}

template <typename Dtype>
void PCALearner<Dtype>::calc_pca_gpu() {
  NO_GPU;
}
#endif

INSTANTIATE_CLASS(PCALearner);
}  // namespace caffe
