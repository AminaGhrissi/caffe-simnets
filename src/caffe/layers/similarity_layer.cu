#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/similarity_layer.hpp"
#include "caffe/util/ggemm.cuh"
#include "caffe/layers/similarity_layer_shared.cuh"

namespace caffe {

template <typename Dtype>
Dtype SimilarityLayer<Dtype>::test_init_step_objective_gpu(const vector<Blob<Dtype>*>& bottom) {
  if (!needs_unsupervised_init()) {
    return INFINITY;
  }
  int batch_size = 0;
  for (int i = 0; i < bottom.size(); ++i) {
    batch_size += N_ * bottom[i]->num();
  }
  input_for_learner_[0]->Reshape(batch_size, K_, 1, 1);
  Dtype* patches_data = input_for_learner_[0]->mutable_gpu_data();
  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    const Dtype* bottom_data = bottom[bottom_idx]->gpu_data();
    Dtype* col_buff = NULL;
    if (!is_1x1_ || normalize_patches_) {
      col_buff = col_buffer_.mutable_gpu_data();
    }
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      if (!is_1x1_) {
        im2col_3d_gpu(
            bottom_data + bottom[bottom_idx]->offset(n),
            channels_, height_, width_,
            block_c_, block_h_, block_w_,
            pad_c_, pad_h_, pad_w_,
            stride_c_, 1, 1, // For init it is best to densly sample patches for translation invariance
            col_buff, true, std::isnan(block_out_of_bounds_value_) ? 0 : block_out_of_bounds_value_);
      } else {  // special case for 1x1 convolution
        if (!normalize_patches_) {
          col_buff = bottom[bottom_idx]->mutable_gpu_data() + bottom[bottom_idx]->offset(n);
        } else {
          caffe_copy(N_ * K_, bottom[bottom_idx]->gpu_data() + bottom[bottom_idx]->offset(n), col_buff);
        }
      }
      if (normalize_patches_) {
        caffe_gpu_transpose(K_, N_,
          col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
        caffe_gpu_normalize_patches_rows_forward(K_, N_,
          normalization_fudge_factor_, patches_data + (bottom_idx * num_ + n) * K_ * N_, normalize_variance_);
      } else {
        caffe_gpu_transpose(K_, N_,
          col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
      }
    }
  }
  return unsupervised_learner_->objective_gpu(input_for_learner_);
}

template <typename Dtype>
bool SimilarityLayer<Dtype>::init_step_gpu(const vector<Blob<Dtype>*>& bottom, Dtype* objective) {
  if (!needs_unsupervised_init()) {
    return false;
  }
  int batch_size = 0;
  for (int i = 0; i < bottom.size(); ++i) {
    batch_size += N_ * bottom[i]->num();
  }
  input_for_learner_[0]->Reshape(batch_size, K_, 1, 1);
  Dtype* patches_data = input_for_learner_[0]->mutable_gpu_data();
  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    const Dtype* bottom_data = bottom[bottom_idx]->gpu_data();
    Dtype* col_buff = NULL;
    if (!is_1x1_ || normalize_patches_) {
      col_buff = col_buffer_.mutable_gpu_data();
    }
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      if (!is_1x1_) {
        im2col_3d_gpu(
            bottom_data + bottom[bottom_idx]->offset(n),
            channels_, height_, width_,
            block_c_, block_h_, block_w_,
            pad_c_, pad_h_, pad_w_,
            stride_c_, 1, 1, // For init it is best to densly sample patches for translation invariance
            col_buff, true, std::isnan(block_out_of_bounds_value_) ? 0 : block_out_of_bounds_value_);
      } else {  // special case for 1x1 convolution
        if (!normalize_patches_) {
          col_buff = bottom[bottom_idx]->mutable_gpu_data() + bottom[bottom_idx]->offset(n);
        } else {
          caffe_copy(N_ * K_, bottom[bottom_idx]->gpu_data() + bottom[bottom_idx]->offset(n), col_buff);
        }
      }
      if (normalize_patches_) {
        caffe_gpu_transpose(K_, N_,
          col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
        caffe_gpu_normalize_patches_rows_forward(K_, N_,
          normalization_fudge_factor_, patches_data + (bottom_idx * num_ + n) * K_ * N_, normalize_variance_);
      } else {
        caffe_gpu_transpose(K_, N_,
          col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
      }
    }
  }
  bool not_finished = unsupervised_learner_->step_gpu(input_for_learner_, objective);
  if (!not_finished) {
    SimilarityParameter sim_param = this->layer_param_.similarity_param();
    UnsupervisedInitialization init_param = sim_param.unsupervised_init();
    if (init_param.use_centroids_variance() || init_param.type() == "gmm") {
      unsupervised_learner_->fill_gpu(this->blobs_);
      if (bias_term_ && init_param.type() == "gmm") {
        caffe_gpu_log<Dtype>(this->blobs_[2]->count(),
          this->blobs_[2]->mutable_gpu_data(), this->blobs_[2]->mutable_gpu_data());
        if (!normalization_term_) {
          caffe_gpu_add_scalar<Dtype>(this->blobs_[2]->count(),
            Dtype(-0.5 * K_ * log(2.0 * M_PI)), this->blobs_[2]->mutable_gpu_data());
          caffe_gpu_log<Dtype>(this->blobs_[1]->count(),
            this->blobs_[1]->gpu_data(), this->blobs_[1]->mutable_gpu_diff());
          caffe_gpu_gemv<Dtype>(CblasNoTrans, num_instances_, K_,
            Dtype(-0.5), this->blobs_[1]->gpu_diff(), bias_multiplier_.gpu_data(),
            Dtype(1), this->blobs_[2]->mutable_gpu_data());
          caffe_gpu_set(this->blobs_[1]->count(),
            Dtype(0), this->blobs_[1]->mutable_gpu_diff());
        }
      }
      if (this->layer_param_.similarity_param().similarity_function() == SimilarityParameter_SimilarityFunction_L2) {
        caffe_gpu_inv(num_instances_* K_, this->blobs_[1]->mutable_gpu_data(), this->blobs_[1]->mutable_gpu_data());
        if (!normalization_term_) {
          caffe_gpu_scal(num_instances_* K_, Dtype(0.5), this->blobs_[1]->mutable_gpu_data());
        }
        if (use_log_space_weight_param_) {
          caffe_gpu_log(num_instances_* K_,
                        this->blobs_[1]->mutable_gpu_data(), this->blobs_[1]->mutable_gpu_data(),
                        normalization_term_fudge_);
        }
      }
    } else {
      const vector<shared_ptr<Blob<Dtype> > > blobs(1, this->blobs_[0]);
      unsupervised_learner_->fill_gpu(blobs);
    }
    for (int i = 0; i < input_for_learner_.size(); ++i) {
      input_for_learner_[i].reset();
    }
    input_for_learner_.clear();
    unsupervised_learner_.reset();
    param_initialized_ = true;
  }
  return not_finished;
}

template <typename Dtype>
void SimilarityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* col_buff = NULL;
  if (!is_1x1_ || normalize_patches_) {
    col_buff = col_buffer_.mutable_gpu_data();
  }
  const Dtype* templates = this->blobs_[0]->gpu_data();
  const Dtype* weights = this->blobs_[1]->gpu_data();

  const int params_size = num_instances_ * block_w_ * block_h_ * block_c_;
  typename vec<Dtype>::vec2 * inter_params = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_->mutable_gpu_data());
  interlace_gpu<Dtype>(params_size, templates, weights, inter_params);

  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    const Dtype* bottom_data = bottom[bottom_idx]->gpu_data();
    Dtype* top_data = top[bottom_idx]->mutable_gpu_data();
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      if (!is_1x1_) {
        im2col_3d_gpu(
            bottom_data + bottom[bottom_idx]->offset(n),
            channels_, height_, width_,
            block_c_, block_h_, block_w_,
            pad_c_, pad_h_, pad_w_,
            stride_c_, stride_h_, stride_w_,
            col_buff, true, block_out_of_bounds_value_);
      } else {
        if (!normalize_patches_) {
          col_buff = bottom[bottom_idx]->mutable_gpu_data() + bottom[bottom_idx]->offset(n);
        } else {
          caffe_copy(K_ * N_, bottom[bottom_idx]->gpu_data() + bottom[bottom_idx]->offset(n), col_buff);
        }
      }
      if (normalize_patches_) {
        caffe_gpu_transpose(K_, N_,
          col_buff,
          row_buffer_.mutable_gpu_data());
        caffe_gpu_normalize_patches_rows_forward(K_, N_, normalization_fudge_factor_,
          row_buffer_.mutable_gpu_data(), normalize_variance_);
        caffe_gpu_transpose(N_, K_,
          row_buffer_.gpu_data(),
          col_buff);
      }
      switch (this->layer_param_.similarity_param().similarity_function()) {
        case SimilarityParameter_SimilarityFunction_CONVOLUTION:
          ggemm_gpu
            <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
            sim_linear_forward<Dtype>, ggemm_add<Dtype>, false>
            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n),
              make_vec2<Dtype>(0, 0), 0, 0, 0);
          break;
        case SimilarityParameter_SimilarityFunction_L1:
          ggemm_gpu
            <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
             sim_l1_forward<Dtype>, ggemm_add<Dtype>, false>
            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n),
              make_vec2<Dtype>(0, 0), 0, 0, 0);
          break;
        case SimilarityParameter_SimilarityFunction_L2:
          if (normalization_term_) {
            if (use_log_space_weight_param_) {
              if (ignore_nan_input_) {
                ggemm_gpu
                  <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                  sim_l2_normalized_forward<Dtype, true, true>, ggemm_add<Dtype>, false>
                  (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n),
                    make_vec2<Dtype>(0, 0), NAN, 0, normalization_term_fudge_);
              } else {
                ggemm_gpu
                  <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                  sim_l2_normalized_forward<Dtype, true, false>, ggemm_add<Dtype>, false>
                  (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n),
                    make_vec2<Dtype>(0, 0), 0, 0, normalization_term_fudge_);
                caffe_gpu_add_scalar<Dtype>(M_ * N_, Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI),
                  top_data + top[bottom_idx]->offset(n));
              }
            } else {
              if (ignore_nan_input_) {
                ggemm_gpu
                  <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                  sim_l2_normalized_forward<Dtype, false, true>, ggemm_add<Dtype>, false>
                  (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n),
                    make_vec2<Dtype>(0, Dtype(1)-normalization_term_fudge_), NAN, 0, normalization_term_fudge_);
              } else {
                ggemm_gpu
                  <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                  sim_l2_normalized_forward<Dtype, false, false>, ggemm_add<Dtype>, false>
                  (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n),
                    make_vec2<Dtype>(0, Dtype(1)-normalization_term_fudge_), 0, 0, normalization_term_fudge_);
                caffe_gpu_add_scalar<Dtype>(M_ * N_, Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI),
                  top_data + top[bottom_idx]->offset(n));
              }
            }
          } else {
            if (use_log_space_weight_param_) {
              ggemm_gpu
                <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                sim_l2_forward<Dtype, true>, ggemm_add<Dtype>, false>
                (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 
                  make_vec2<Dtype>(0, 0), 0, 0, 0);
            } else {
              ggemm_gpu
                <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                sim_l2_forward<Dtype, false>, ggemm_add<Dtype>, false>
                (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 
                  make_vec2<Dtype>(0, 0), 0, 0, 0);
            }
          }
          break;
        default:
          break;
      }
      // Add bias.
      if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_instances_, N_, 1,
          (Dtype)1., this->blobs_[2]->gpu_data(), bias_multiplier_.gpu_data(),
          (Dtype)1., top_data + top[bottom_idx]->offset(n));
      }
    }
  }
}


template <typename Dtype>
void SimilarityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* templates = this->blobs_[0]->gpu_data();
  const Dtype* weights = this->blobs_[1]->gpu_data();

  bool propagate_down_any = false;
  for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
    if (propagate_down[top_idx]) {
      propagate_down_any = true;
      break;
    }
  }

  typename vec<Dtype>::vec2 * inter_params_transposed = NULL;
  if (propagate_down_any) {
    Dtype* templates_transposed = static_cast<Dtype *>(templates_transposed_->mutable_gpu_data());
    Dtype* weights_transposed = static_cast<Dtype *>(weights_transposed_->mutable_gpu_data());
    caffe_gpu_transpose(M_, K_,
            templates,
            templates_transposed);
    caffe_gpu_transpose(M_, K_,
            weights,
            weights_transposed);
    inter_params_transposed = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_transposed_->mutable_gpu_data());
    interlace_gpu<Dtype>(M_ * K_,
      templates_transposed, weights_transposed,
      inter_params_transposed);
  }

  Dtype* templates_diff = NULL;
  Dtype* weights_diff = NULL;
  typename vec<Dtype>::vec2 * interlaced_params_diff = NULL;
  typename vec<Dtype>::vec2 * inter_params = NULL;
  if (this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
    templates_diff = this->blobs_[0]->mutable_gpu_diff();
    weights_diff = this->blobs_[1]->mutable_gpu_diff();
    interlaced_params_diff = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_diff_->mutable_gpu_data());
    const int params_size = M_ * K_;
    interlace_gpu<Dtype>(params_size,
      templates_diff, weights_diff,
      interlaced_params_diff);
    inter_params = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_->mutable_gpu_data());
    interlace_gpu<Dtype>(M_ * K_,
      templates, weights,
      inter_params);
  }

  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[2]) {
    bias_diff = this->blobs_[2]->mutable_gpu_diff();
  }

  for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[2]) {
      top_diff = top[top_idx]->gpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_instances_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_.gpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || this->param_propagate_down_[1] || propagate_down[top_idx]) {
     Dtype* col_buff = NULL;
      if (!is_1x1_ || normalize_patches_) {
        col_buff = col_buffer_.mutable_gpu_data();
      }
      const Dtype* bottom_data = bottom[top_idx]->gpu_data();
      Dtype* bottom_diff = bottom[top_idx]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        if (!is_1x1_) {
          im2col_3d_gpu(
              bottom_data + bottom[top_idx]->offset(n),
              channels_, height_, width_,
              block_c_, block_h_, block_w_,
              pad_c_, pad_h_, pad_w_,
              stride_c_, stride_h_, stride_w_,
              col_buff, true, block_out_of_bounds_value_);
        } else {
          if (!normalize_patches_) {
            col_buff = bottom[top_idx]->mutable_gpu_data() + bottom[top_idx]->offset(n);
          } else {
            caffe_copy(N_ * K_, bottom[top_idx]->mutable_gpu_data() + bottom[top_idx]->offset(n), col_buff);
          }
        }
        Dtype* row_buff = row_buffer_.mutable_gpu_data();
        caffe_gpu_transpose(K_, N_,
            col_buff,
            row_buff);
        if (normalize_patches_) {
          caffe_copy(K_ * N_,
            row_buff,
            row_buffer_.mutable_gpu_diff());
          caffe_gpu_normalize_patches_rows_forward(K_, N_, normalization_fudge_factor_,
            row_buff, normalize_variance_);
          caffe_gpu_transpose(N_, K_,
            row_buff,
            col_buff);
        }
        top_diff = top[top_idx]->gpu_diff() + top[0]->offset(n);
        // gradient w.r.t. weights and templates. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
          switch (this->layer_param_.similarity_param().similarity_function()) {
            case SimilarityParameter_SimilarityFunction_CONVOLUTION:
              ggemm_readc_gpu
                <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                sim_linear_backward_weights<Dtype>, add_vec2<Dtype>, true, no_op<typename vec<Dtype>::vec2>, false>
                (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, 0, 0, make_vec2<Dtype>(0,0), 0);
              break;
            case SimilarityParameter_SimilarityFunction_L1:
              ggemm_readc_gpu
                <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                sim_l1_backward_weights<Dtype>, add_vec2<Dtype>, true, no_op<typename vec<Dtype>::vec2>, false>
                (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, 0, 0, make_vec2<Dtype>(0,0), 0);
              break;
            case SimilarityParameter_SimilarityFunction_L2:
              if (normalization_term_) {
                if (use_log_space_weight_param_) {
                  if (ignore_nan_input_) {
                    ggemm_readc_gpu
                      <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                      sim_l2_normalized_backward_weights<Dtype, true, true>, add_vec2<Dtype>, true, no_op<typename vec<Dtype>::vec2>, false>
                      (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, 0, NAN, make_vec2<Dtype>(0,0),
                        normalization_term_fudge_);
                  } else {
                    ggemm_readc_gpu
                      <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                      sim_l2_normalized_backward_weights<Dtype, true, false>, add_vec2<Dtype>, true, no_op<typename vec<Dtype>::vec2>, false>
                      (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, 0, 0, make_vec2<Dtype>(0,0),
                        normalization_term_fudge_);
                  }
                } else {
                  if (ignore_nan_input_) {
                    ggemm_readc_gpu
                      <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                      sim_l2_normalized_backward_weights<Dtype, false, true>, add_vec2<Dtype>, true, no_op<typename vec<Dtype>::vec2>, false>
                      (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, 0, NAN, make_vec2<Dtype>(0,0),
                        normalization_term_fudge_);
                  } else {
                    ggemm_readc_gpu
                      <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                      sim_l2_normalized_backward_weights<Dtype, false, false>, add_vec2<Dtype>, true, no_op<typename vec<Dtype>::vec2>, false>
                      (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, 0, 0, make_vec2<Dtype>(0,0),
                        normalization_term_fudge_);
                  }
                }
              } else {
                if (use_log_space_weight_param_) {
                  ggemm_readc_gpu
                    <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                    sim_l2_backward_weights<Dtype, true>, add_vec2<Dtype>, true, no_op<typename vec<Dtype>::vec2>, false>
                    (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, 0, 0, make_vec2<Dtype>(0,0), 0);
                } else {
                  ggemm_readc_gpu
                    <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                    sim_l2_backward_weights<Dtype, false>, add_vec2<Dtype>, true, no_op<typename vec<Dtype>::vec2>, false>
                    (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, 0, 0, make_vec2<Dtype>(0,0), 0);
                }
              }
              break;
            default:
              break;
          }
        }

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[top_idx]) {
          Dtype* col_diff_buff = NULL;
          if (is_1x1_) {
            col_diff_buff = bottom[top_idx]->mutable_gpu_diff() + bottom[top_idx]->offset(n);
          } else {
            col_diff_buff = col_buffer_.mutable_gpu_diff();
          }

          switch (this->layer_param_.similarity_param().similarity_function()) {
            case SimilarityParameter_SimilarityFunction_CONVOLUTION:
              ggemm_readc_gpu
                <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                sim_linear_backward_bottom<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                (K_, N_, M_, inter_params_transposed, top_diff, col_buff, col_diff_buff, make_vec2<Dtype>(0,0), 0, 0, 0);
              break;
            case SimilarityParameter_SimilarityFunction_L1:
              ggemm_readc_gpu
                <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                sim_l1_backward_bottom<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                (K_, N_, M_, inter_params_transposed, top_diff, col_buff, col_diff_buff, make_vec2<Dtype>(0,0), 0, 0, 0);
              break;
            case SimilarityParameter_SimilarityFunction_L2:
              if (normalization_term_) {
                if (use_log_space_weight_param_) {
                  if (ignore_nan_input_) {
                    ggemm_readc_gpu
                      <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                      sim_l2_normalized_backward_bottom<Dtype, true, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                      (K_, N_, M_, inter_params_transposed, top_diff, col_buff, col_diff_buff,
                       make_vec2<Dtype>(0,0), 0, 0, normalization_term_fudge_);
                  } else {
                    ggemm_readc_gpu
                      <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                      sim_l2_normalized_backward_bottom<Dtype, true, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                      (K_, N_, M_, inter_params_transposed, top_diff, col_buff, col_diff_buff,
                       make_vec2<Dtype>(0,0), 0, 0, normalization_term_fudge_);
                  }
                } else {
                  if (ignore_nan_input_) {
                    ggemm_readc_gpu
                      <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                      sim_l2_normalized_backward_bottom<Dtype, false, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                      (K_, N_, M_, inter_params_transposed, top_diff, col_buff, col_diff_buff,
                       make_vec2<Dtype>(0,0), 0, 0, normalization_term_fudge_);
                  } else {
                    ggemm_readc_gpu
                      <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                      sim_l2_normalized_backward_bottom<Dtype, false, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                      (K_, N_, M_, inter_params_transposed, top_diff, col_buff, col_diff_buff,
                       make_vec2<Dtype>(0,0), 0, 0, normalization_term_fudge_);
                  }
                }
              } else {
                if (use_log_space_weight_param_) {
                  ggemm_readc_gpu
                    <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                    sim_l2_backward_bottom<Dtype, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                    (K_, N_, M_, inter_params_transposed, top_diff, col_buff, col_diff_buff, make_vec2<Dtype>(0,0), 0, 0, 0);
                } else {
                  ggemm_readc_gpu
                    <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                    sim_l2_backward_bottom<Dtype, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                    (K_, N_, M_, inter_params_transposed, top_diff, col_buff, col_diff_buff, make_vec2<Dtype>(0,0), 0, 0, 0);
                }
              }
              break;
            default:
              break;
          }
          if (normalize_patches_) {
            caffe_gpu_transpose(K_, N_, col_diff_buff, col_buff);
            caffe_gpu_normalize_patches_rows_backward(K_, N_, normalization_fudge_factor_,
              row_buffer_.gpu_diff(), row_buffer_.gpu_data(), col_buff, normalize_variance_);
            caffe_gpu_transpose(N_, K_, col_buff, col_diff_buff);
          }

          // col2im back to the data
          if (!is_1x1_) {
            col2im_3d_gpu(
                col_diff_buff,
                channels_, height_, width_,
                block_c_, block_h_, block_w_,
                pad_c_, pad_h_, pad_w_,
                stride_c_, stride_h_, stride_w_,
                bottom_diff + bottom[top_idx]->offset(n));
          }
        }
      }
    }
  }
  if (this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
    const int params_size = M_ * K_;
    deinterlace_gpu<Dtype>(params_size,
      interlaced_params_diff, templates_diff, weights_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SimilarityLayer);
INSTANTIATE_LAYER_GPU_INIT_STEP(SimilarityLayer);

}  // namespace caffe
