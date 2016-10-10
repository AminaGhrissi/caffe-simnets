#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ggemm_cpu.hpp"
#include "caffe/layers/similarity_layer.hpp"
#include "caffe/layers/similarity_layer_shared.cuh"

namespace caffe {

template <typename Dtype>
void SimilarityLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SimilarityParameter sim_param = this->layer_param_.similarity_param();
  ignore_nan_input_ = sim_param.ignore_nan_input();
  normalize_patches_ = sim_param.normalize_patches();
  if (normalize_patches_) {
    normalization_fudge_factor_ = sim_param.normalization_fudge_factor();
    normalize_variance_ = sim_param.normalize_variance();
  }
  use_log_space_weight_param_ = sim_param.use_log_space_weight_param();
  BlockParameter block_param = sim_param.block_param();
  block_out_of_bounds_value_ = block_param.out_of_bounds_value();
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_instances_ = sim_param.num_instances();
  CHECK_GT(num_instances_, 0);

  normalization_term_ = sim_param.normalization_term();
  if (normalization_term_) {
    CHECK(this->layer_param_.similarity_param().similarity_function() == SimilarityParameter_SimilarityFunction_L2)
      << "The normalization term can only be used with L2 similarity function.";
    normalization_term_fudge_ = sim_param.normalization_term_fudge();
    CHECK_GE(normalization_term_fudge_, 0) << "Normalization term fudge factor must be non-negative.";
  }

  // Configure the block size, padding, stride, and inputs.
  CHECK(!block_param.has_block_size() !=
      !(block_param.has_block_h() && block_param.has_block_w()))
      << "Filter size is block_size OR block_h and block_w; not both";
  CHECK(block_param.has_block_size() ||
      (block_param.has_block_h() && block_param.has_block_w()))
      << "For non-square filters both block_h and block_w are required.";
  CHECK((!block_param.has_pad() && block_param.has_pad_h()
      && block_param.has_pad_w())
      || (!block_param.has_pad_h() && !block_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!block_param.has_stride() && block_param.has_stride_h()
      && block_param.has_stride_w())
      || (!block_param.has_stride_h() && !block_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (block_param.has_block_size()) {
    block_h_ = block_w_ = block_param.block_size();
  } else {
    block_h_ = block_param.block_h();
    block_w_ = block_param.block_w();
  }
  block_c_ = block_param.block_c();
  if (block_c_ < 0) {
    block_c_ = channels_;
  }

  CHECK_GT(block_c_, 0) << "Filter dimensions cannot be zero.";
  CHECK_LE(block_c_, channels_) 
     << "Filter dimensions cannot be exceed channel dimention";
  CHECK_GT(block_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(block_w_, 0) << "Filter dimensions cannot be zero.";

  if (block_param.has_pad()) {
    pad_h_ = pad_w_ = block_param.pad();
  } else {
    pad_h_ = block_param.pad_h();
    pad_w_ = block_param.pad_w();
  }
  pad_c_ = block_param.pad_c();

  if (!block_param.has_stride_h()) {
    stride_h_ = stride_w_ = block_param.stride();
  } else {
    stride_h_ = block_param.stride_h();
    stride_w_ = block_param.stride_w();
  }
  stride_c_ = block_param.stride_c();
  if (stride_c_ < 0) {
    stride_c_ = block_c_;
  }

  // Special case: im2col is the identity for 1x1xchannels convolution
  // with stride 1 in 2D and no padding, so flag for skipping the buffer
  // and transformation.
  is_1x1_ = block_c_ == channels_ && block_w_ == 1 && block_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 
      && pad_c_ == 0 && pad_h_ == 0 && pad_w_ == 0;

  // Handle the parameters: templatess and biases.
  // - blobs_[0] holds the filter templates
  // - blobs_[1] holds the filter weights
  // - blobs_[2] holds the biases (optional)
  param_initialized_ = false;
  bias_term_ = this->layer_param_.similarity_param().bias_term();
  const int params_size = num_instances_ * block_c_ * block_h_ * block_w_;
  const int padding_size = ggemm_padded_output_size(num_instances_, block_c_ * block_h_ * block_w_);
  weights_transposed_.reset(new SyncedMemory((params_size + padding_size) * sizeof(Dtype)));
  templates_transposed_.reset(new SyncedMemory((params_size + padding_size) * sizeof(Dtype)));
  interlaced_params_.reset(new SyncedMemory((params_size + padding_size) * sizeof(typename vec<Dtype>::vec2)));
  interlaced_params_transposed_.reset(new SyncedMemory((params_size + padding_size) * sizeof(typename vec<Dtype>::vec2)));
  interlaced_params_diff_.reset(new SyncedMemory((params_size + padding_size) * sizeof(typename vec<Dtype>::vec2)));
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
    param_initialized_ = true;
  } else {
    if (bias_term_) {
      this->blobs_.resize(3);
    } else {
      this->blobs_.resize(2);
    }
    // Initialize and fill the templates:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        num_instances_, block_c_, block_h_, block_w_));
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[1].reset(new Blob<Dtype>(
        num_instances_, block_c_, block_h_, block_w_));
    if (needs_unsupervised_init()) {
      SimilarityParameter sim_param = this->layer_param_.similarity_param();
      UnsupervisedInitialization init_param = sim_param.unsupervised_init();
      const std::string& type = init_param.type();
      if (type == "kmeans") {
        // only templates are init with kmeans -> init weights
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.similarity_param().weight_filler()));
        weight_filler->Fill(this->blobs_[1].get());

        unsupervised_learner_.reset(new KmeansLearner<Dtype>(
          num_instances_, init_param.max_iterations(), init_param.num_batches(),
          init_param.prob_choose_centroid(), init_param.use_kmeans_plus_plus(), init_param.fudge_factor()));
      } else if (type == "gmm") {
        unsupervised_learner_.reset(new GMMLearner<Dtype>(
          num_instances_, init_param.max_iterations(), init_param.num_batches(),
          init_param.fudge_factor(), init_param.soft_kmeans()));
      } else {
        LOG(FATAL) << "Layer " << this->layer_param_.name() << " uses unsupported unsupervised initialization type";
      }
      input_for_learner_.resize(1);
      input_for_learner_[0].reset(new Blob<Dtype>(1,1,1,1));
    } else {
      shared_ptr<Filler<Dtype> > template_filler(GetFiller<Dtype>(
        this->layer_param_.similarity_param().template_filler()));
      template_filler->Fill(this->blobs_[0].get());
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.similarity_param().weight_filler()));
      weight_filler->Fill(this->blobs_[1].get());
      param_initialized_ = true;
    }
    // Used to store consolidated weight and template blobs for faster forward
    // If necessary, initialize and fill the biases:
    // 1 x 1 x 1 x output channels
    if (bias_term_) {
      this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, num_instances_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.similarity_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
bool SimilarityLayer<Dtype>::needs_unsupervised_init() {
  if (param_initialized_) {
    return false;
  }
  SimilarityParameter sim_param = this->layer_param_.similarity_param();
  if (!sim_param.has_unsupervised_init()) {
    return false;
  }
  UnsupervisedInitialization init_param = sim_param.unsupervised_init();
  const std::string& type = init_param.type();
  if (type == "none") {
    return false;
  } else {
    return true;
  }
}

template <typename Dtype>
void SimilarityLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  height_out_ = (height_ + 2 * pad_h_ - block_h_) / stride_h_ + 1;
  width_out_ = (width_ + 2 * pad_w_ - block_w_) / stride_w_ + 1;
  channels_out_ = (channels_ + 2 * pad_c_ - block_c_) / stride_c_ + 1;
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(
      num_, num_instances_*channels_out_, height_out_, width_out_);
  }
  // Prepare the matrix multiplication computation.
  // Each input will be convolved as a single GEMM.
  M_ = num_instances_;
  K_ = block_c_ * block_h_ * block_w_;
  N_ = height_out_ * width_out_ * channels_out_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_.Reshape(1,
    block_c_ * block_h_ * block_w_, channels_out_ * height_out_, width_out_);
  col_buffer_.SetPadding(ggemm_padded_output_size(block_c_ * block_h_ * block_w_,
    channels_out_ * height_out_ * width_out_));
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_ || normalization_term_) {
    bias_multiplier_.Reshape(1, 1, 1, std::max(num_instances_, std::max(N_, K_)));
    caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  row_buffer_.Reshape(block_c_ * block_h_ * block_w_,
      channels_out_, height_out_, width_out_);
}

template <typename Dtype>
Dtype SimilarityLayer<Dtype>::test_init_step_objective_cpu(const vector<Blob<Dtype>*>& bottom) {
  if (!needs_unsupervised_init()) {
    return INFINITY;
  }
  int batch_size = 0;
  for (int i = 0; i < bottom.size(); ++i) {
    batch_size += N_ * bottom[i]->num();
  }
  input_for_learner_[0]->Reshape(batch_size, K_, 1, 1);
  Dtype* patches_data = input_for_learner_[0]->mutable_cpu_data();
  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    const Dtype* bottom_data = bottom[bottom_idx]->cpu_data();
    Dtype* col_buff = NULL;
    if (!is_1x1_ || normalize_patches_) {
      col_buff = col_buffer_.mutable_cpu_data();
    }
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      if (!is_1x1_) {
        im2col_3d_cpu(
            bottom_data + bottom[bottom_idx]->offset(n),
            channels_, height_, width_,
            block_c_, block_h_, block_w_,
            pad_c_, pad_h_, pad_w_,
            stride_c_, 1, 1, // For init it is best to densly sample patches for translation invariance
            col_buff, true, std::isnan(block_out_of_bounds_value_) ? 0 : block_out_of_bounds_value_);
      } else {  // special case for 1x1 convolution
        if (!normalize_patches_) {
          col_buff = bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n);
        } else {
          caffe_copy(N_ * K_, bottom[bottom_idx]->cpu_data() + bottom[bottom_idx]->offset(n), col_buff);
        }
      }

      if (normalize_patches_) {
        caffe_cpu_transpose(K_, N_,
          col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
        caffe_cpu_normalize_patches_rows_forward(K_, N_,
          normalization_fudge_factor_, patches_data + (bottom_idx * num_ + n) * K_ * N_, normalize_variance_);
      } else {
        caffe_cpu_transpose(K_, N_,
          col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
      }
    }
  }
  return unsupervised_learner_->objective_cpu(input_for_learner_);
}

template <typename Dtype>
bool SimilarityLayer<Dtype>::init_step_cpu(const vector<Blob<Dtype>*>& bottom, Dtype* objective) {
  if (!needs_unsupervised_init()) {
    return false;
  }
  int batch_size = 0;
  for (int i = 0; i < bottom.size(); ++i) {
    batch_size += N_ * bottom[i]->num();
  }
  input_for_learner_[0]->Reshape(batch_size, K_, 1, 1);
  Dtype* patches_data = input_for_learner_[0]->mutable_cpu_data();
  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    const Dtype* bottom_data = bottom[bottom_idx]->cpu_data();
    Dtype* col_buff = NULL;
    if (!is_1x1_ || normalize_patches_) {
      col_buff = col_buffer_.mutable_cpu_data();
    }
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      if (!is_1x1_) {
        im2col_3d_cpu(
            bottom_data + bottom[bottom_idx]->offset(n),
            channels_, height_, width_,
            block_c_, block_h_, block_w_,
            pad_c_, pad_h_, pad_w_,
            stride_c_, 1, 1, // For init it is best to densly sample patches for translation invariance
            col_buff, true, std::isnan(block_out_of_bounds_value_) ? 0 : block_out_of_bounds_value_);
      } else {  // special case for 1x1 convolution
        if (!normalize_patches_) {
          col_buff = bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n);
        } else {
          caffe_copy(N_ * K_, bottom[bottom_idx]->cpu_data() + bottom[bottom_idx]->offset(n), col_buff);
        }
      }
      if (normalize_patches_) {
        caffe_cpu_transpose(K_, N_,
          col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
        caffe_cpu_normalize_patches_rows_forward(K_, N_,
          normalization_fudge_factor_, patches_data + (bottom_idx * num_ + n) * K_ * N_, normalize_variance_);
      } else {
        caffe_cpu_transpose(K_, N_,
          col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
      }
    }
  }
  bool not_finished = unsupervised_learner_->step_cpu(input_for_learner_, objective);
  if (!not_finished) {
    SimilarityParameter sim_param = this->layer_param_.similarity_param();
    UnsupervisedInitialization init_param = sim_param.unsupervised_init();
    if (init_param.use_centroids_variance() || init_param.type() == "gmm") {
      unsupervised_learner_->fill_cpu(this->blobs_);
      if (bias_term_ && init_param.type() == "gmm") {
        caffe_log<Dtype>(this->blobs_[2]->count(),
          this->blobs_[2]->mutable_cpu_data(), this->blobs_[2]->mutable_cpu_data());
        if (!normalization_term_) {
          caffe_add_scalar<Dtype>(this->blobs_[2]->count(),
            Dtype(-0.5 * K_ * log(2.0 * M_PI)), this->blobs_[2]->mutable_cpu_data());
          caffe_log<Dtype>(this->blobs_[1]->count(),
            this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_diff());
          caffe_cpu_gemv<Dtype>(CblasNoTrans, num_instances_, K_,
            Dtype(-0.5), this->blobs_[1]->cpu_diff(), bias_multiplier_.cpu_data(),
            Dtype(1), this->blobs_[2]->mutable_cpu_data());
          caffe_set(this->blobs_[1]->count(),
            Dtype(0), this->blobs_[1]->mutable_cpu_diff());
        }
      }
      if (this->layer_param_.similarity_param().similarity_function() == SimilarityParameter_SimilarityFunction_L2) {
        caffe_cpu_inv(num_instances_* K_, this->blobs_[1]->mutable_cpu_data(), this->blobs_[1]->mutable_cpu_data());
        if (!normalization_term_) {
          caffe_scal(num_instances_* K_, Dtype(0.5), this->blobs_[1]->mutable_cpu_data());
        }
        if (use_log_space_weight_param_) {
          caffe_log(num_instances_* K_,
                    this->blobs_[1]->mutable_gpu_data(), this->blobs_[1]->mutable_gpu_data(),
                    normalization_term_fudge_);
        }
      }
    } else {
      const vector<shared_ptr<Blob<Dtype> > > blobs(1, this->blobs_[0]);
      unsupervised_learner_->fill_cpu(blobs);
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
void SimilarityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* col_buff = NULL;
  if (!is_1x1_ || normalize_patches_) {
    col_buff = col_buffer_.mutable_cpu_data();
  }
  const Dtype* templates = this->blobs_[0]->cpu_data();
  const Dtype* weights = this->blobs_[1]->cpu_data();

  const int params_size = num_instances_ * block_w_ * block_h_ * block_c_;
  typename vec<Dtype>::vec2 * inter_params = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_->mutable_cpu_data());
  interlace_cpu<Dtype>(params_size, templates, weights, inter_params);

  for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
    const Dtype* bottom_data = bottom[bottom_idx]->cpu_data();
    Dtype* top_data = top[bottom_idx]->mutable_cpu_data();
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      if (!is_1x1_) {
        im2col_3d_cpu(
            bottom_data + bottom[bottom_idx]->offset(n),
            channels_, height_, width_,
            block_c_, block_h_, block_w_,
            pad_c_, pad_h_, pad_w_,
            stride_c_, stride_h_, stride_w_,
            col_buff, true, block_out_of_bounds_value_);
      } else {  // special case for 1x1 convolution
        if (!normalize_patches_) {
          col_buff = bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n);
        } else {
          caffe_copy(K_ * N_, bottom[bottom_idx]->cpu_data() + bottom[bottom_idx]->offset(n), col_buff);
        }
      }

      if (normalize_patches_) {
        caffe_cpu_transpose(K_, N_,
          col_buff,
          row_buffer_.mutable_cpu_data());
        caffe_cpu_normalize_patches_rows_forward(K_, N_, normalization_fudge_factor_,
          row_buffer_.mutable_cpu_data(), normalize_variance_);
        caffe_cpu_transpose(N_, K_,
          row_buffer_.cpu_data(),
          col_buff);
      }
      
      switch (this->layer_param_.similarity_param().similarity_function()) {
        case SimilarityParameter_SimilarityFunction_CONVOLUTION:
          ggemm_cpu
            <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
            sim_linear_forward<Dtype>, ggemm_add<Dtype>, false>
            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, 0);
          break;
        case SimilarityParameter_SimilarityFunction_L1:
          ggemm_cpu
            <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
             sim_l1_forward<Dtype>, ggemm_add<Dtype>, false>
            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, 0);
          break;
        case SimilarityParameter_SimilarityFunction_L2:
          if (normalization_term_) {
            if (use_log_space_weight_param_) {
              if (ignore_nan_input_) {
                ggemm_cpu
                  <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                  sim_l2_normalized_forward<Dtype, true, true>, ggemm_add<Dtype>, false>
                  (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, normalization_term_fudge_);
              } else {
                ggemm_cpu
                  <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                  sim_l2_normalized_forward<Dtype, true, false>, ggemm_add<Dtype>, false>
                  (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, normalization_term_fudge_);
                caffe_add_scalar<Dtype>(M_ * N_, Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI),
                  top_data + top[bottom_idx]->offset(n));
              }
            } else {
              if (ignore_nan_input_) {
                ggemm_cpu
                  <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                  sim_l2_normalized_forward<Dtype, false, true>, ggemm_add<Dtype>, false>
                  (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, normalization_term_fudge_);
              } else {
                ggemm_cpu
                  <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                  sim_l2_normalized_forward<Dtype, false, false>, ggemm_add<Dtype>, false>
                  (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, normalization_term_fudge_);
                caffe_add_scalar<Dtype>(M_ * N_, Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI),
                  top_data + top[bottom_idx]->offset(n));
              }
            }
          } else {
            if (use_log_space_weight_param_) {
              ggemm_cpu
                <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                sim_l2_forward<Dtype, true>, ggemm_add<Dtype>, false>
                (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, 0);
            } else {
              ggemm_cpu
                <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                sim_l2_forward<Dtype, false>, ggemm_add<Dtype>, false>
                (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, 0);
            }

          }
          break;
        default:
          break;
      }
      // Add bias.
      if (bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_instances_,
            N_, 1, (Dtype)1., this->blobs_[2]->cpu_data(),
            bias_multiplier_.cpu_data(),
            (Dtype)1., top_data + top[bottom_idx]->offset(n));
      }
    }
  }
}

template <typename Dtype>
void SimilarityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* templates = this->blobs_[0]->cpu_data();
  const Dtype* weights = this->blobs_[1]->cpu_data();

  bool propagate_down_any = false;
  for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
    if (propagate_down[top_idx]) {
      propagate_down_any = true;
      break;
    }
  }
  typename vec<Dtype>::vec2 * inter_params = NULL;
  if (propagate_down_any || this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
    inter_params = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_->mutable_cpu_data());
    interlace_cpu<Dtype>(M_ * K_,
      templates, weights,
      inter_params);
  }

  Dtype* templates_diff = NULL;
  Dtype* weights_diff = NULL;
  typename vec<Dtype>::vec2 * interlaced_params_diff = NULL;
  if (this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
    templates_diff = this->blobs_[0]->mutable_cpu_diff();
    weights_diff = this->blobs_[1]->mutable_cpu_diff();
    interlaced_params_diff = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_diff_->mutable_cpu_data());
    const int params_size = M_ * K_;
    interlace_cpu<Dtype>(params_size,
      templates_diff, weights_diff,
      interlaced_params_diff);
  }

  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[2]) {
    bias_diff = this->blobs_[2]->mutable_cpu_diff();
  }

  for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[2]) {
      top_diff = top[top_idx]->cpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_instances_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_.cpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || this->param_propagate_down_[1] || propagate_down[top_idx]) {
      Dtype* col_buff = NULL;
      if (!is_1x1_ || normalize_patches_) {
        col_buff = col_buffer_.mutable_cpu_data();
      }
      const Dtype* bottom_data = bottom[top_idx]->cpu_data();
      Dtype* bottom_diff = bottom[top_idx]->mutable_cpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        if (!is_1x1_) {
          im2col_3d_cpu(
              bottom_data + bottom[top_idx]->offset(n),
              channels_, height_, width_,
              block_c_, block_h_, block_w_,
              pad_c_, pad_h_, pad_w_,
              stride_c_, stride_h_, stride_w_,
              col_buff, true, block_out_of_bounds_value_);
        } else {
          if (!normalize_patches_) {
            col_buff = bottom[top_idx]->mutable_cpu_data() + bottom[top_idx]->offset(n);
          } else {
            caffe_copy(N_ * K_, bottom[top_idx]->mutable_cpu_data() + bottom[top_idx]->offset(n), col_buff);
          }
        }
        Dtype* row_buff = row_buffer_.mutable_cpu_data();
        caffe_cpu_transpose(K_, N_,
            col_buff,
            row_buff);
        if (normalize_patches_) {
          caffe_copy(K_ * N_,
            row_buff,
            row_buffer_.mutable_cpu_diff());
          caffe_cpu_normalize_patches_rows_forward(K_, N_, normalization_fudge_factor_,
            row_buff, normalize_variance_);
          caffe_cpu_transpose(N_, K_,
            row_buff,
            col_buff);
        }
        top_diff = top[top_idx]->cpu_diff() + top[0]->offset(n);
        // gradient w.r.t. weights and templates. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
          switch (this->layer_param_.similarity_param().similarity_function()) {
            case SimilarityParameter_SimilarityFunction_CONVOLUTION:
              ggemm_readc_cpu
                <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                sim_linear_backward_weights<Dtype>, add_vec2<Dtype>, true,
                no_op<typename vec<Dtype>::vec2>, false>
                (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
              break;
            case SimilarityParameter_SimilarityFunction_L1:
              ggemm_readc_cpu
                <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                sim_l1_backward_weights<Dtype>, add_vec2<Dtype>, true,
                no_op<typename vec<Dtype>::vec2>, false>
                (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
              break;
            case SimilarityParameter_SimilarityFunction_L2:
              if (normalization_term_) {
                if (use_log_space_weight_param_) {
                  if (ignore_nan_input_) {
                    ggemm_readc_cpu
                      <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                      sim_l2_normalized_backward_weights<Dtype, true, true>, add_vec2<Dtype>, true,
                      no_op<typename vec<Dtype>::vec2>, false>
                      (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0),
                        normalization_term_fudge_);
                  } else {
                    ggemm_readc_cpu
                      <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                      sim_l2_normalized_backward_weights<Dtype, true, false>, add_vec2<Dtype>, true,
                      no_op<typename vec<Dtype>::vec2>, false>
                      (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0),
                        normalization_term_fudge_);
                  }
                } else {
                  if (ignore_nan_input_) {
                    ggemm_readc_cpu
                      <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                      sim_l2_normalized_backward_weights<Dtype, false, true>, add_vec2<Dtype>, true,
                      no_op<typename vec<Dtype>::vec2>, false>
                      (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0),
                        normalization_term_fudge_);
                  } else {
                    ggemm_readc_cpu
                      <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                      sim_l2_normalized_backward_weights<Dtype, false, false>, add_vec2<Dtype>, true,
                      no_op<typename vec<Dtype>::vec2>, false>
                      (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0),
                        normalization_term_fudge_);
                  }
                }
              } else {
                if (use_log_space_weight_param_) {
                  ggemm_readc_cpu
                    <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                    sim_l2_backward_weights<Dtype, true>, add_vec2<Dtype>, true,
                    no_op<typename vec<Dtype>::vec2>, false>
                    (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
                } else {
                  ggemm_readc_cpu
                    <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                    sim_l2_backward_weights<Dtype, false>, add_vec2<Dtype>, true,
                    no_op<typename vec<Dtype>::vec2>, false>
                    (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
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
            col_diff_buff = bottom[top_idx]->mutable_cpu_diff() + bottom[top_idx]->offset(n);
          } else {
            col_diff_buff = col_buffer_.mutable_cpu_diff();
          }

          switch (this->layer_param_.similarity_param().similarity_function()) {
            case SimilarityParameter_SimilarityFunction_CONVOLUTION:
              ggemm_readc_cpu
                <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                sim_linear_backward_bottom<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, 0);
              break;
            case SimilarityParameter_SimilarityFunction_L1:
              ggemm_readc_cpu
                <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                sim_l1_backward_bottom<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, 0);
              break;
            case SimilarityParameter_SimilarityFunction_L2:
              if (normalization_term_) {
                if (use_log_space_weight_param_) {
                  if (ignore_nan_input_) {
                    ggemm_readc_cpu
                      <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                      sim_l2_normalized_backward_bottom<Dtype, true, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                      (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, normalization_term_fudge_);
                  } else {
                    ggemm_readc_cpu
                      <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                      sim_l2_normalized_backward_bottom<Dtype, true, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                      (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, normalization_term_fudge_);
                  } 
                } else {
                  if (ignore_nan_input_) {
                    ggemm_readc_cpu
                      <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                      sim_l2_normalized_backward_bottom<Dtype, false, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                      (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, normalization_term_fudge_);
                  } else {
                    ggemm_readc_cpu
                      <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                      sim_l2_normalized_backward_bottom<Dtype, false, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                      (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, normalization_term_fudge_);
                  }
                }
              } else {
                if (use_log_space_weight_param_) {
                  ggemm_readc_cpu
                    <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                    sim_l2_backward_bottom<Dtype, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                    (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, 0);
                } else {
                  ggemm_readc_cpu
                    <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                    sim_l2_backward_bottom<Dtype, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                    (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, 0);
                }
              }
              break;
            default:
              break;
          }
          if (normalize_patches_) {
            caffe_cpu_transpose(K_, N_, col_diff_buff, col_buff);
            caffe_cpu_normalize_patches_rows_backward(K_, N_, normalization_fudge_factor_,
              row_buffer_.cpu_diff(), row_buffer_.cpu_data(), col_buff, normalize_variance_);
            caffe_cpu_transpose(N_, K_, col_buff, col_diff_buff);
          }

          // col2im back to the data
          if (!is_1x1_) {
            col2im_3d_cpu(
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
    deinterlace_cpu<Dtype>(params_size,
      interlaced_params_diff, templates_diff, weights_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SimilarityLayer);
template <typename Dtype>
void SimilarityLayer<Dtype>::init_step_gpu(const vector<Blob<Dtype>*>& bottom) {
  NO_GPU;
}
#endif

INSTANTIATE_CLASS(SimilarityLayer);
}  // namespace caffe
