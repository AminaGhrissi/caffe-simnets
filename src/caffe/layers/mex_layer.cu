#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mex_layer.hpp"
#include "math_constants.h"
#include "caffe/util/ggemm.cuh"
#include "caffe/layers/mex_layer_shared.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
namespace caffe {

template <typename Dtype>
__global__ void linear_offsets_gradient_kernel(const int n, const Dtype* logspace_offsets,
    const Dtype* logspace_offsets_diff, const Dtype fudge_factor, Dtype* offsets_diff) {
  CUDA_KERNEL_LOOP(i, n) {
    offsets_diff[i] += logspace_offsets_diff[i] / max(exp(logspace_offsets[i]), fudge_factor);
  }
}

template <typename Dtype>
void linear_offsets_gradient(const int n, const Dtype* logspace_offsets,
    const Dtype* logspace_offsets_diff, const Dtype fudge_factor, Dtype* offsets_diff) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  linear_offsets_gradient_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, logspace_offsets, logspace_offsets_diff, fudge_factor, offsets_diff);
}

template <typename Dtype>
Dtype MEXLayer<Dtype>::test_init_step_objective_gpu(const vector<Blob<Dtype>*>& bottom) {
  if (!needs_unsupervised_init()) {
    return INFINITY;
  }
  int batch_size = 0;
  for (int i = 0; i < bottom.size(); ++i) {
    if (expects_labels_ && i % 2 == 1) continue;
    batch_size += N_ * bottom[i]->num();
  }

  input_for_learner_[0]->Reshape(batch_size, K_, 1, 1);
  if (expects_labels_) {
    input_for_learner_[1]->Reshape(batch_size, 1, 1, 1);
  }

  Dtype* patches_data = input_for_learner_[0]->mutable_gpu_data();
  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    if (expects_labels_ && bottom_idx % 2 == 1) continue;
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
          stride_c_, stride_h_, stride_w_,
          col_buff,
          blocks_round_down_, blocks_out_of_bounds_value_);
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
  if (expects_labels_) {
    Dtype* labels_data = input_for_learner_[1]->mutable_gpu_data();
    for (int bottom_idx = 1; bottom_idx < bottom.size(); bottom_idx += 2) {
      const Dtype* labels = bottom[bottom_idx]->gpu_data();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, N_, 1,
        Dtype(1), labels, one_zero_vec_.gpu_data(),
        Dtype(0), labels_data + ((bottom_idx - 1) / 2) * num_ * N_);
    }
  }
  return unsupervised_learner_->objective_gpu(input_for_learner_);
}

template <typename Dtype>
bool MEXLayer<Dtype>::init_step_gpu(const vector<Blob<Dtype>*>& bottom, Dtype* objective) {
  if (!needs_unsupervised_init()) {
    return false;
  }
  int batch_size = 0;
  for (int i = 0; i < bottom.size(); ++i) {
    if (expects_labels_ && i % 2 == 1) continue;
    batch_size += N_ * bottom[i]->num();
  }

  input_for_learner_[0]->Reshape(batch_size, K_, 1, 1);
  if (expects_labels_) {
    input_for_learner_[1]->Reshape(batch_size, 1, 1, 1);
  }

  Dtype* patches_data = input_for_learner_[0]->mutable_gpu_data();
  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    if (expects_labels_ && bottom_idx % 2 == 1) continue;
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
          stride_c_, stride_h_, stride_w_,
          col_buff,
          blocks_round_down_, blocks_out_of_bounds_value_);
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
  if (expects_labels_) {
    Dtype* labels_data = input_for_learner_[1]->mutable_gpu_data();
    for (int bottom_idx = 1; bottom_idx < bottom.size(); bottom_idx += 2) {
      const Dtype* labels = bottom[bottom_idx]->gpu_data();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, N_, 1,
        Dtype(1), labels, one_zero_vec_.gpu_data(),
        Dtype(0), labels_data + ((bottom_idx - 1) / 2) * num_ * N_);
    }
  }
  bool not_finished = unsupervised_learner_->step_gpu(input_for_learner_, objective);
  if (!not_finished) {
    const vector<shared_ptr<Blob<Dtype> > > blobs(1, this->blobs_[1]);
    unsupervised_learner_->fill_gpu(blobs);

    for (int i = 0; i < input_for_learner_.size(); ++i) {
      input_for_learner_[i].reset();
    }
    input_for_learner_.clear();
    unsupervised_learner_.reset();

    param_initialized_ = true;
  }
  return not_finished;
}

template <typename Dtype, bool REVERSE>
__global__ void split_patches_kernel(const int num_kernels, const int N, const int Dim,
                      const int W, const int H, const int C,
                      const int W_Gs, const int H_Gs, const int C_Gs,
                      const int W_Step, const int H_Step, const int C_Step,
                      typename std::conditional<REVERSE, Dtype*, const Dtype*>::type in,
                      Dtype* out, const bool use_unshared_regions_) {
  const int step_out = C_Step * H_Step * W_Step;
  const int group_step_w = !use_unshared_regions_ ? W_Step : 1;
  const int group_step_h = !use_unshared_regions_ ? H_Step : 1;
  const int group_step_c = !use_unshared_regions_ ? C_Step : 1;
  const int region_step_w = !use_unshared_regions_ ? 1 : W_Gs;
  const int region_step_h = !use_unshared_regions_ ? 1 : H_Gs;
  const int region_step_c = !use_unshared_regions_ ? 1 : C_Gs;
  Dtype* in_unconst = NULL;
  if (REVERSE) {
    in_unconst = (Dtype*)in;
  }
  CUDA_KERNEL_LOOP(index, num_kernels) {
    const int i = index % W_Step;
    const int i_index = index / W_Step;
    const int j = i_index % H_Step;
    const int j_index = i_index / H_Step;
    const int l = j_index % C_Step;
    const int l_index = j_index / C_Step;
    const int w_g = l_index % W_Gs;
    const int w_index = l_index / W_Gs;
    const int h_g = w_index % H_Gs;
    const int h_index = w_index / H_Gs;
    const int c_g = h_index;

    // "inner loop"
    Dtype* o = out + ((c_g * H_Gs + h_g) * W_Gs + w_g) * step_out * Dim;
    const int group_addr = (c_g * group_step_c * H + h_g * group_step_h) * W + w_g * group_step_w;
    const int base_addr_out = (l * H_Step + j) * W_Step + i;
    const int base_addr_in  = group_addr + (l * region_step_c * H + j * region_step_h) * W  + i * region_step_w;
    if (w_g * W_Step + i < W &&
        h_g * H_Step + j < H &&
        c_g * C_Step + l < C) {
      for (int k = 0; k < Dim; ++k) {
        if (!REVERSE) {
          o[base_addr_out + k * step_out] = in[base_addr_in + k * N];
        } else {
          in_unconst[base_addr_in + k * N] = o[base_addr_out + k * step_out];
        }
      }
    }
  }
}


template <typename Dtype, bool REVERSE> 
void split_patches_gpu(const int N, const int Dim,
                      const int W, const int H, const int C,
                      const int W_Gs, const int H_Gs, const int C_Gs,
                      const int W_Step, const int H_Step, const int C_Step,
                      typename std::conditional<REVERSE, Dtype*, const Dtype*>::type in,
                      Dtype* out, const bool use_unshared_regions) {
  const int num_kernels = W_Step * H_Step * C_Step * W_Gs * H_Gs * C_Gs;
  // NOLINT_NEXT_LINE(whitespace/operators)
  split_patches_kernel<Dtype, REVERSE><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, N, Dim, W, H, C, W_Gs, H_Gs, C_Gs, W_Step, H_Step, C_Step, in, out, use_unshared_regions);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void mex_forward_gpu(const int M, const int N, const int K, const bool softmax_mode,
    const Dtype epsilon, const Dtype* offsets, const Dtype* in, Dtype* out, const int batch_size = 1) {
  const Dtype init_value = epsilon > 0 ? -INFINITY : INFINITY;
  if (epsilon > 0) {
    ggemm_gpu
      <Dtype, Dtype, Dtype, uint8_t,
       ggemm_add<Dtype, uint8_t>, ggemm_max<Dtype>, false,
       true, true, true>
      (M, N, K, offsets, in, out,
      init_value, init_value, init_value, 0, batch_size);
  } else {
    ggemm_gpu
      <Dtype, Dtype, Dtype, uint8_t,
       ggemm_add<Dtype, uint8_t>, ggemm_min<Dtype>, false,
       true, true, true>
      (M, N, K, offsets, in, out,
      init_value, init_value, init_value, 0, batch_size);
  }
  if (std::isfinite(epsilon)) {
    ggemm_readc_gpu
      <false, false, Dtype, Dtype, Dtype, typename vec<Dtype>::vec2,
       mex_forward_exp<Dtype>, ggemm_add<Dtype>, true, mex_forward_out<Dtype>, true,
       true, true, true>
      (M, N, K, offsets, in, out, out,
      init_value, init_value, 0, make_vec2<Dtype>(epsilon, softmax_mode ? Dtype(0) : (Dtype)-std::log(K)), batch_size);
  }
}

template <typename Dtype>
void MEXLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* col_buff = NULL;
  if (!is_1x1_ || normalize_patches_) {
    col_buff = col_buffer_.mutable_gpu_data();
  }
  const Dtype epsilon = this->blobs_[0]->cpu_data()[0];
  Dtype* split_patches_in = NULL;
  Dtype* split_patches_out = NULL;
  const Dtype* offsets = this->blobs_[1]->gpu_data();
  if (!use_log_space_parameters_) {
    caffe_gpu_clip_min<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_gpu_data(), linear_space_min_value_);
    caffe_gpu_log<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_gpu_data());
  }
  if (normalize_offsets_) {
    mex_forward_gpu<Dtype>(M_ * num_regions_, 1, K_, softmax_mode_, epsilon,
      offsets, one_zero_vec_.gpu_diff(), offsets_norm_factor_.mutable_gpu_data());
    Dtype* offsets_mutable = NULL;
    if (!normalize_offsets_projected_) {
      caffe_copy<Dtype>(num_regions_ * M_ * K_, offsets, normed_offsets_.mutable_gpu_data());
      offsets = normed_offsets_.gpu_data();
      offsets_mutable = normed_offsets_.mutable_gpu_data();
    } else {
      offsets_mutable = this->blobs_[1]->mutable_gpu_data();
    }
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_ * num_regions_, K_, 1,
      -1, offsets_norm_factor_.gpu_data(), one_zero_vec_.gpu_data(),
       1, offsets_mutable);
  }

  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    const int top_idx = expects_labels_ ? bottom_idx / 2 : bottom_idx;
    const Dtype* bottom_data = bottom[bottom_idx]->gpu_data();
    Dtype* top_data = top[top_idx]->mutable_gpu_data();

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
            col_buff,
            blocks_round_down_, blocks_out_of_bounds_value_);
      } else {  // special case for 1x1 convolution
        if (!normalize_patches_) {
          col_buff = bottom[bottom_idx]->mutable_gpu_data() + bottom[bottom_idx]->offset(n);
        } else {
          caffe_copy(N_ * K_, bottom[bottom_idx]->gpu_data() + bottom[bottom_idx]->offset(n), col_buff);
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
      // Prepare input
      Dtype* current_top = top_data + top[top_idx]->offset(n);
      if (num_regions_ > 1) {
        split_patches_in = split_patches_in_.mutable_gpu_data();
        split_patches_out = split_patches_out_.mutable_gpu_data();
        split_patches_gpu<Dtype, false>(N_, K_,
                        width_out_, height_out_, channels_out_,
                        offsets_w_, offsets_h_, offsets_c_,
                        shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                        col_buff, split_patches_in, use_unshared_regions_);
      } else {
        split_patches_in = col_buff;
        split_patches_out = current_top;
      }

      // Calculate
      mex_forward_gpu<Dtype>(M_, region_size_, K_, softmax_mode_, epsilon,
        offsets, split_patches_in, split_patches_out, num_regions_);

      // Copy to output if needed
      if (num_regions_ > 1) {
        split_patches_gpu<Dtype, true>(N_, M_,
                        width_out_, height_out_, channels_out_,
                        offsets_w_, offsets_h_, offsets_c_,
                        shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                        current_top, split_patches_out, use_unshared_regions_);
      }
    }
  }
  if (!use_log_space_parameters_) {
    caffe_gpu_exp<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_gpu_data());
    caffe_gpu_clip_min<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_gpu_data(), linear_space_min_value_);
  }
}


template <typename Dtype>
void MEXLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  Dtype* split_patches_in = NULL;
  Dtype* split_patches_in_diff = NULL;
  Dtype* split_patches_out = NULL;
  Dtype* split_patches_out_diff = NULL;
  typename vec<Dtype>::vec2* split_patches_out_inter = NULL;

  const Dtype epsilon = this->blobs_[0]->cpu_data()[0];
  Dtype epsilon_diff = 0;
  Dtype* epsilon_helper = NULL;
  if (this->param_propagate_down_[0]) {
    epsilon_helper = static_cast<Dtype*>(epsilon_helper_->mutable_gpu_data());
  }

  const Dtype* offsets = this->blobs_[1]->gpu_data();
  if (!use_log_space_parameters_) {
    caffe_gpu_clip_min<Dtype>(num_regions_ * M_ * K_, offsets, this->blobs_[1]->mutable_gpu_data(), linear_space_min_value_);
    caffe_gpu_log<Dtype>(num_regions_ * M_ * K_, offsets, this->blobs_[1]->mutable_gpu_data());
  }
  if (normalize_offsets_) {
    mex_forward_gpu<Dtype>(M_ * num_regions_, 1, K_, softmax_mode_, epsilon,
      offsets, one_zero_vec_.gpu_diff(), offsets_norm_factor_.mutable_gpu_data());
    Dtype* offsets_mutable = NULL;
    if (!normalize_offsets_projected_) {
      caffe_copy<Dtype>(num_regions_ * M_ * K_, offsets, normed_offsets_.mutable_gpu_data());
      offsets = normed_offsets_.gpu_data();
      offsets_mutable = normed_offsets_.mutable_gpu_data();
    } else {
      offsets_mutable = this->blobs_[1]->mutable_gpu_data();
    }
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_ * num_regions_, K_, 1,
      -1, offsets_norm_factor_.gpu_data(), one_zero_vec_.gpu_data(),
       1, offsets_mutable);
  }
  Dtype* offsets_diff = NULL;
  if (this->param_propagate_down_[1]) {
    if (use_log_space_parameters_) {
      offsets_diff = this->blobs_[1]->mutable_gpu_diff();
    } else {
      offsets_diff = normed_offsets_.mutable_gpu_diff();
    }
  }

  bool propagate_down_any = false;
  for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
    if (propagate_down[top_idx]) {
      propagate_down_any = true;
      break;
    }
  }
  const Dtype* transposed_offsets = NULL;
  if (propagate_down_any) {
    transposed_offsets = static_cast<const Dtype*>(transposed_offsets_->gpu_data());
    for (int r = 0; r < num_regions_; ++r) {
      const int offsets_idx = r * M_ * K_;
      caffe_gpu_transpose(M_, K_,
                  offsets + offsets_idx,
                  static_cast<Dtype*>(transposed_offsets_->mutable_gpu_data()) + offsets_idx);
    }
  }
  for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
    const int bottom_idx = expects_labels_ ? top_idx * 2 : top_idx;
    if (this->param_propagate_down_[0] ||
        this->param_propagate_down_[1] ||
        propagate_down[top_idx]) {
      const Dtype* top_diff = top[top_idx]->gpu_diff();
      const Dtype* top_data = top[top_idx]->gpu_data();
      Dtype* col_buff = NULL;
      Dtype* col_diff = NULL;
      if (!is_1x1_ || normalize_patches_) {
        col_buff = col_buffer_.mutable_gpu_data();
      }
      if (!is_1x1_) {
        col_diff = col_buffer_.mutable_gpu_diff();
      }
      const Dtype* bottom_data = bottom[bottom_idx]->gpu_data();
      Dtype* bottom_diff = bottom[bottom_idx]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        if (!is_1x1_) {
          im2col_3d_gpu(
              bottom_data + bottom[bottom_idx]->offset(n),
              channels_, height_, width_,
              block_c_, block_h_, block_w_,
              pad_c_, pad_h_, pad_w_,
              stride_c_, stride_h_, stride_w_,
              col_buff,
              blocks_round_down_, blocks_out_of_bounds_value_);
        } else {  // special case for 1x1 convolution
          col_diff = bottom_diff + bottom[bottom_idx]->offset(n);
          if (!normalize_patches_) {
            col_buff = bottom[bottom_idx]->mutable_gpu_data() + bottom[bottom_idx]->offset(n);
          } else {
            caffe_copy(N_ * K_, bottom[bottom_idx]->mutable_gpu_data() + bottom[bottom_idx]->offset(n), col_buff);
          }
        }
        if (normalize_patches_) {
          caffe_gpu_transpose(K_, N_,
            col_buff,
            row_buffer_.mutable_gpu_data());
          caffe_copy(K_ * N_,
            row_buffer_.gpu_data(),
            row_buffer_.mutable_gpu_diff());
          caffe_gpu_normalize_patches_rows_forward(K_, N_, normalization_fudge_factor_,
            row_buffer_.mutable_gpu_data(), normalize_variance_);
          caffe_gpu_transpose(N_, K_,
            row_buffer_.gpu_data(),
            col_buff);
        }
        // Prepare input for backprop
        const Dtype* current_top_data = top_data + n * M_ * N_;
        const Dtype* current_top_diff = top_diff + n * M_ * N_;
        if (num_regions_ > 1) {
          split_patches_in = split_patches_in_.mutable_gpu_data();
          split_patches_in_diff = split_patches_in_.mutable_gpu_diff();
          split_patches_out = split_patches_out_.mutable_gpu_data();
          split_patches_out_diff = split_patches_out_.mutable_gpu_diff();
          split_patches_gpu<Dtype, false>(N_, K_,
                          width_out_, height_out_, channels_out_,
                          offsets_w_, offsets_h_, offsets_c_,
                          shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                          col_buff, split_patches_in, use_unshared_regions_);
          split_patches_gpu<Dtype, false>(N_, M_,
                          width_out_, height_out_, channels_out_,
                          offsets_w_, offsets_h_, offsets_c_,
                          shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                          current_top_data, split_patches_out, use_unshared_regions_);
          split_patches_gpu<Dtype, false>(N_, M_,
                          width_out_, height_out_, channels_out_,
                          offsets_w_, offsets_h_, offsets_c_,
                          shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                          current_top_diff, split_patches_out_diff, use_unshared_regions_);
        } else {
          split_patches_in = col_buff;
          split_patches_in_diff = col_diff;
          split_patches_out = (Dtype*)current_top_data;
          split_patches_out_diff = (Dtype*)current_top_diff;
        }
        split_patches_out_inter = static_cast<typename vec<Dtype>::vec2 *>(
            split_patches_out_inter_->mutable_gpu_data());
        interlace_gpu(num_regions_ * M_ * region_size_, split_patches_out, split_patches_out_diff,
            split_patches_out_inter);
        // Caculate backprop
        if (this->param_propagate_down_[0] && std::isfinite(epsilon)) { // epsilon = ±inf => epsilon_diff = 0
          if (!normalize_offsets_ || normalize_offsets_projected_) {
            ggemm_readc_gpu
              <false, true, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
               mex_backward_epsilon<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype, Dtype>, false,
               true, true, true>
              (M_, K_, region_size_, split_patches_out_inter, split_patches_in,
               offsets, epsilon_helper,
              make_vec2<Dtype>(0, 0), 0, 0, epsilon, num_regions_);
          } else {
            ggemm_readc_gpu
              <false, true, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
               mex_backward_epsilon_with_normalized_offsets<Dtype>, ggemm_add<Dtype>, false,
               no_op<Dtype, Dtype>, false,
               true, true, true>
              (M_, K_, region_size_, split_patches_out_inter, split_patches_in,
               offsets, epsilon_helper,
              make_vec2<Dtype>(0, 0), 0, 0, epsilon, num_regions_);
          }
          thrust::device_ptr<Dtype> cptr = thrust::device_pointer_cast(epsilon_helper);
          const Dtype sum_offsets_diff = thrust::reduce(cptr, cptr + num_regions_ * M_ * K_);
          epsilon_diff += sum_offsets_diff / (epsilon * K_);
        }
        if (this->param_propagate_down_[1]) {
          if (!use_log_space_parameters_) {
            caffe_gpu_set(M_ * K_ * num_regions_, Dtype(0), offsets_diff);
          }

          if (!normalize_offsets_ || normalize_offsets_projected_) {
            if (std::isfinite(epsilon)) {
              ggemm_readc_gpu
                <false, true, typename vec<Dtype>::vec2, Dtype, Dtype, typename vec<Dtype>::vec2,
                 mex_backward_offsets_finite<Dtype>, ggemm_add<Dtype>, true, no_op<Dtype, typename vec<Dtype>::vec2>, false,
                 true, true, true>
                (M_, K_, region_size_, split_patches_out_inter, split_patches_in,
                 offsets, offsets_diff,
                make_vec2<Dtype>(epsilon > 0 ? INFINITY : -INFINITY, 0), 0, 0,
                make_vec2<Dtype>(epsilon, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
            } else {
              ggemm_readc_gpu
                <false, true, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                 mex_backward_offsets_infinite<Dtype>, ggemm_add<Dtype>, true, no_op<Dtype, uint8_t>, false,
                 true, true, true>
                (M_, K_, region_size_, split_patches_out_inter, split_patches_in,
                 offsets, offsets_diff,
                make_vec2<Dtype>(0, 0), 0, 0, 0, num_regions_);
            }
          } else {
            if (std::isfinite(epsilon)) {
              ggemm_readc_gpu
                <false, true, typename vec<Dtype>::vec2, Dtype, Dtype, typename vec<Dtype>::vec2,
                 mex_backward_normalized_offsets_finite<Dtype>, ggemm_add<Dtype>, true,
                 no_op<Dtype, typename vec<Dtype>::vec2>, false,
                 true, true, true>
                (M_, K_, region_size_, split_patches_out_inter, split_patches_in,
                 offsets, offsets_diff,
                make_vec2<Dtype>(epsilon > 0 ? INFINITY : -INFINITY, 0), 0, 0,
                make_vec2<Dtype>(epsilon, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
            } else {
              ggemm_readc_gpu
                <false, true, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                 mex_backward_normalized_offsets_infinite<Dtype>, ggemm_add<Dtype>, true,
                 no_op<Dtype, uint8_t>, false,
                 true, true, true>
                (M_, K_, region_size_, split_patches_out_inter, split_patches_in,
                 offsets, offsets_diff,
                make_vec2<Dtype>(0, 0), 0, 0, 0, num_regions_);
            }
          }
        }
        if (propagate_down[top_idx]) {
          if (std::isfinite(epsilon)) {
            ggemm_readc_gpu
              <false, false, Dtype, typename vec<Dtype>::vec2, Dtype, typename vec<Dtype>::vec2,
               mex_backward_bottom_finite<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype, typename vec<Dtype>::vec2>, false,
               true, true, true>
              (K_, region_size_, M_, transposed_offsets, split_patches_out_inter,
               split_patches_in, split_patches_in_diff, 0, make_vec2<Dtype>(epsilon > 0 ? INFINITY : -INFINITY, 0), 0,
              make_vec2<Dtype>(epsilon, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
          } else {
            ggemm_readc_gpu
              <false, false, Dtype, typename vec<Dtype>::vec2, Dtype, uint8_t,
               mex_backward_bottom_infinite<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype, uint8_t>, false,
               true, true, true>
              (K_, region_size_, M_, transposed_offsets, split_patches_out_inter,
               split_patches_in, split_patches_in_diff, 0, make_vec2<Dtype>(0, 0), 0, 0, num_regions_);
          }
        }
        // Copy to bottom if needed
        if (num_regions_ > 1) {
          split_patches_gpu<Dtype, true>(N_, K_,
                          width_out_, height_out_, channels_out_,
                          offsets_w_, offsets_h_, offsets_c_,
                          shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                          col_diff, split_patches_in_diff, use_unshared_regions_);
        }

        // Backprop for patch normalization
        if (normalize_patches_ && propagate_down[top_idx]) {
          caffe_gpu_transpose(K_, N_, col_diff, col_buff);
          caffe_gpu_normalize_patches_rows_backward(K_, N_, normalization_fudge_factor_,
            row_buffer_.gpu_diff(), row_buffer_.gpu_data(), col_buff, normalize_variance_);
          caffe_gpu_transpose(N_, K_, col_buff, col_diff);
        }

        if (propagate_down[top_idx] && !is_1x1_) {
          col2im_3d_gpu(
            col_diff,
            channels_, height_, width_,
            block_c_, block_h_, block_w_,
            pad_c_, pad_h_, pad_w_,
            stride_c_, stride_h_, stride_w_,
            bottom_diff + bottom[bottom_idx]->offset(n),
            blocks_round_down_);
        }
        if (!use_log_space_parameters_ && this->param_propagate_down_[1]) {
          const Dtype* original_logspace_offsets = this->blobs_[1]->gpu_data();
          Dtype* original_offsets_diff = this->blobs_[1]->mutable_gpu_diff();
          linear_offsets_gradient<Dtype>(num_regions_ * M_ * K_, original_logspace_offsets,
            offsets_diff, linear_space_min_value_, original_offsets_diff);
        }
      }
    }
  }
  if (this->param_propagate_down_[0]) {
    this->blobs_[0]->mutable_cpu_diff()[0] = epsilon_diff;
  }
  if (use_log_space_parameters_ && this->param_propagate_down_[1] && this->maximum_entropy_regularization_coeff_ > Dtype(0)) {
    caffe_gpu_maximum_entropy_regularization(num_regions_ * M_, K_, offsets, normed_offsets_.mutable_gpu_diff());
    caffe_gpu_axpy(num_regions_ * M_ * K_, maximum_entropy_regularization_coeff_, normed_offsets_.gpu_diff(), offsets_diff);
  }
  if (!use_log_space_parameters_) {
    caffe_gpu_exp<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->gpu_data(), this->blobs_[1]->mutable_gpu_data());
    caffe_gpu_clip_min<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->gpu_data(), this->blobs_[1]->mutable_gpu_data(), linear_space_min_value_);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(MEXLayer);
INSTANTIATE_LAYER_GPU_INIT_STEP(MEXLayer);
}  // namespace caffe
