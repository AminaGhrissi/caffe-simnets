#include <cstring>
#include <vector>
//#include <cmath>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mex_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#define MEX_IDX(w,h,c,k,W,H,C) ((((k)*(C) + (c)) * (H) + (h)) * (W) + (w))

// Reference similarity operator for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_mex(const Dtype* in, const Dtype epsilon, 
  const int width, const int height, const int channels, const int K,
  Dtype* out, const bool additive = false, const Dtype mul = 1,
  const bool ignore_inf_values = false, const bool softmax_mode = false) {
  for (int w = 0; w < width; ++w) {
    for (int h = 0; h < height; ++h) {
      for (int c = 0; c < channels; ++c) {
        Dtype m = in[MEX_IDX(w,h,c,0,width,height,channels)];
        for (int k = 0; k < K; ++k) {
          const Dtype x = in[MEX_IDX(w,h,c,k,width,height,channels)];
          m = epsilon > 0 ? std::max(m, x) : std::min(m, x);
        }
        if (std::isfinite(epsilon)) {
          int finite_values_count = 0;
          Dtype sum = 0;
          for (int k = 0; k < K; ++k) {
            const Dtype x = in[MEX_IDX(w,h,c,k,width,height,channels)];
            sum += std::exp(epsilon * (x - m));
            finite_values_count += std::isfinite(x);
          }
          int sum_mul = K;
          if (softmax_mode) {
            sum_mul = 1;
          } else if (ignore_inf_values) {
            sum_mul = finite_values_count;
          }
          if (additive) {
            out[MEX_IDX(w,h,c,0,width,height,channels)] += mul * ((std::log(sum / sum_mul) / epsilon) + m);
          } else {
            out[MEX_IDX(w,h,c,0,width,height,channels)] = (std::log(sum / sum_mul) / epsilon) + m;
          }
        } else if (isinf(epsilon)) {
          if (additive) {
            out[MEX_IDX(w,h,c,0,width,height,channels)] += mul * m;
          } else {
            out[MEX_IDX(w,h,c,0,width,height,channels)] = m;
          }
        }
      }
    }
  }
}

template void caffe_mex(const float* in, const float epsilon, 
  const int width, const int height, const int channels, const int K,
  float* out, const bool additive, const float mul,
  const bool ignore_inf_values, const bool softmax_mode);
template void caffe_mex(const double* in, const double epsilon, 
  const int width, const int height, const int channels, const int K,
  double* out, const bool additive, const double mul,
  const bool ignore_inf_values, const bool softmax_mode);

template <typename Dtype>
void caffe_mex_forward(const Blob<Dtype>* in, MEXParameter* mex_param,
  Blob<Dtype>* col_buffer, const vector<shared_ptr<Blob<Dtype> > >& blobs, Blob<Dtype>* out) {
  const int num_instances = mex_param->num_instances();
  const int width = in->width();
  const int height = in->height();
  const int channels = in->channels();
  const int num = in->num();
  const bool normalize_patches = mex_param->normalize_patches();
  const Dtype normalization_fudge_factor = mex_param->normalization_fudge_factor();
  const bool normalize_variance = mex_param->normalize_variance();
  const bool normalize_offsets = mex_param->normalize_offsets();
  const bool softmax_mode = mex_param->softmax_mode();
  const bool ignore_inf_values = false;//mex_param->ignore_inf_values();
  const bool use_log_space_parameters = mex_param->use_log_space_parameters();
  const Dtype linear_space_min_value = mex_param->linear_space_min_value();

  BlockParameter block_param = mex_param->block_param();
  int block_h, block_c, block_w;
  if (block_param.has_block_size()) {
    block_h = block_w = block_param.block_size();
  } else {
    block_h = block_param.block_h();
    block_w = block_param.block_w();
  }
  block_c = block_param.block_c();
  if (block_c < 0) {
    block_c = in->channels();
  }
  int stride_h, stride_c, stride_w;
  if (block_param.has_stride()) {
    stride_h = stride_w = block_param.stride();
  } else {
    stride_h = block_param.stride_h();
    stride_w = block_param.stride_w();
  }
  stride_c = block_param.stride_c();
  if (stride_c < 0) {
    stride_c = block_c;
  }
  int pad_h, pad_w, pad_c;
  if (!block_param.has_pad()) {
    pad_h = pad_w = block_param.pad();
  } else {
    pad_h = block_param.pad_h();
    pad_w = block_param.pad_w();
  }
  pad_c = block_param.pad_c();
  const bool round_down = block_param.round_down();
  const Dtype out_of_bounds_value = block_param.out_of_bounds_value();
  const int height_out = dimension_out_size(height, pad_h, block_h, stride_h, round_down);
  const int width_out = dimension_out_size(width, pad_w, block_w, stride_w, round_down);;
  const int channels_out = dimension_out_size(channels, pad_c, block_c, stride_c, round_down);
  const int K = block_c * block_h * block_w;
  const int N = channels_out * width_out * height_out;
  col_buffer->Reshape(K, channels_out, height_out, width_out);
  Dtype* col_buff = col_buffer->mutable_cpu_data();
  Dtype* temp_buff = col_buffer->mutable_cpu_diff();
  const Dtype* bottom_data = in->cpu_data();
  Dtype* top_data = out->mutable_cpu_data();

  const Dtype epsilon = blobs[0]->cpu_data()[0];
  const Dtype* offsets_data = blobs[1]->cpu_data();
  int offsets_h=0, offsets_w=0, offsets_c=0;
  int shared_h=0, shared_w=0, shared_c=0;
  bool use_unshared_regions = mex_param->use_unshared_regions();
  if (!use_unshared_regions) {
    if (mex_param->has_shared_offsets_region_h() && mex_param->has_shared_offsets_region_w()) {
      shared_h = mex_param->shared_offsets_region_h();
      shared_w = mex_param->shared_offsets_region_w();
    } else {
      shared_w = shared_h = mex_param->shared_offsets_region_size();
    }
    shared_c = mex_param->shared_offsets_region_c();
    if (shared_h < 0) {
      shared_h = height_out;
    }
    if (shared_w < 0) {
      shared_w = width_out;
    }
    if (shared_c < 0) {
      shared_c = channels_out;
    }
    offsets_w = (width_out / shared_w) + ((width_out % shared_w) > 0);
    offsets_h = (height_out / shared_h) + ((height_out % shared_h) > 0);
    offsets_c = (channels_out / shared_c) + ((channels_out % shared_c) > 0);
  } else {
    if (mex_param->has_unshared_offsets_region_h() && mex_param->has_unshared_offsets_region_w()) {
      offsets_h = mex_param->unshared_offsets_region_h();
      offsets_w = mex_param->unshared_offsets_region_w();
    } else {
      offsets_w = offsets_h = mex_param->unshared_offsets_region_size();
    }
    offsets_c = mex_param->unshared_offsets_region_c();
    if (offsets_h < 0) {
      offsets_h = height_out;
    }
    if (offsets_w < 0) {
      offsets_w = width_out;
    }
    if (offsets_c < 0) {
      offsets_c = channels_out;
    }
    shared_w = (width_out / offsets_w) + ((width_out % offsets_w) > 0);
    shared_h = (height_out / offsets_h) + ((height_out % offsets_h) > 0);
    shared_c = (channels_out / offsets_c) + ((channels_out % offsets_c) > 0);
  }

  for (int n = 0; n < num; ++n) {
    for (int inst = 0; inst < num_instances; ++inst) {
      im2col_3d_cpu(
        bottom_data + in->offset(n),
        channels, height, width,
        block_c, block_h, block_w,
        pad_c, pad_h, pad_w,
        stride_c, stride_h, stride_w,
        col_buff, round_down, out_of_bounds_value);
      if (normalize_patches) {
        caffe_cpu_transpose(K, N, col_buff, temp_buff);
        caffe_cpu_normalize_patches_rows_forward(K, N, normalization_fudge_factor,
          temp_buff, normalize_variance);
        caffe_cpu_transpose(N, K, temp_buff, col_buff);
      }
      for (int w = 0; w < width_out; ++w) {
        for (int h = 0; h < height_out; ++h) {
          for (int c = 0; c < channels_out; ++c) {
            for (int k = 0; k < K; ++k) {
              const int ow = use_unshared_regions ? w % offsets_w : w / shared_w;
              const int oh = use_unshared_regions ? h % offsets_h : h / shared_h;
              const int oc = use_unshared_regions ? c % offsets_c : c / shared_c;
              const int off_set_idx = MEX_IDX(ow, oh, oc, 0, offsets_w, offsets_h, offsets_c);
              const Dtype off_temp = offsets_data[off_set_idx * num_instances * K + inst * K + k];
              const Dtype off = !use_log_space_parameters ?
                  std::log(std::max(off_temp, linear_space_min_value)) : off_temp;
              col_buff[MEX_IDX(w,h,c,k,width_out,height_out,channels_out)] += off;
              temp_buff[MEX_IDX(w,h,c,k,width_out,height_out,channels_out)] = off;
            }
          }
        }
      }
      caffe_mex(col_buff,
        epsilon, width_out, height_out, channels_out, K,
        top_data + out->offset(n) + (width_out * height_out * channels_out) * inst,
        false, Dtype(1), ignore_inf_values, softmax_mode);
      if (normalize_offsets) {
        caffe_mex(temp_buff,
          epsilon, width_out, height_out, channels_out, K,
          top_data + out->offset(n) + (width_out * height_out * channels_out) * inst,
          true, Dtype(-1), ignore_inf_values, softmax_mode);
      }
    }
  }
}

// template <>
// void caffe_mex_forward(const Blob<float>* in, MEXParameter* mex_param,
//   const vector<shared_ptr<Blob<float> > >& blobs, Blob<float>* out);
// template <>
// void caffe_mex_forward(const Blob<double>* in, MEXParameter* mex_param,
//   const vector<shared_ptr<Blob<double> > >& blobs,Blob<double>* out);

template <typename TypeParam>
class MEXLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MEXLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()),
        col_buffer_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MEXLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
    delete col_buffer_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  Blob<Dtype>* const col_buffer_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MEXLayerTest, TestDtypesAndDevices);

TYPED_TEST(MEXLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(MEXLayerTest, TestSetup2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_block_c(1);
  block_param->set_stride(2);
  mex_param->set_num_instances(1);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(MEXLayerTest, TestSimpleBasicMEX) {
  typedef typename TypeParam::Dtype Dtype;
  // FillerParameter filler_param;
  // filler_param.set_value(1.);
  // ConstantFiller<Dtype> filler(filler_param);
  // filler.Fill(this->blob_bottom_);
  // filler.Fill(this->blob_bottom_2_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_block_c(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleBasicMEXPositiveEpsilon) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(4);
  mex_param->mutable_epsilon_filler()->set_type("gaussian");
  mex_param->mutable_epsilon_filler()->set_mean(1);
  mex_param->mutable_epsilon_filler()->set_std(0.3);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleBasicMEXNegativeEpsilon) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(4);
  mex_param->mutable_epsilon_filler()->set_type("gaussian");
  mex_param->mutable_epsilon_filler()->set_mean(-1);
  mex_param->mutable_epsilon_filler()->set_std(0.3);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleBasicMEXLargeNumbers) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_value(1e30);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom_2_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_block_c(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleBasic3DMEX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_block_c(2);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleMEXWithOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(4);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleMEX2DPoolingWithFullySharedOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_block_c(1);
  block_param->set_stride(2);
  block_param->set_round_down(false);
  block_param->set_out_of_bounds_value(-INFINITY);
  mex_param->set_num_instances(1);
  // mex_param->set_ignore_inf_values(true);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(1e-3);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->set_shared_offsets_region_c(-1);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}


TYPED_TEST(MEXLayerTest, TestSimpleMEX2DPoolingWithFullySharedOffsetsMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_block_c(1);
  block_param->set_stride(2);
  mex_param->set_num_instances(1);
  block_param->set_round_down(false);
  block_param->set_out_of_bounds_value(-INFINITY);
  // mex_param->set_ignore_inf_values(true);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->set_shared_offsets_region_c(-1);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleMEX2DPoolingWithPartiallySharedOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_block_c(1);
  block_param->set_stride(2);
  mex_param->set_num_instances(1);
  block_param->set_round_down(false);
  block_param->set_out_of_bounds_value(-INFINITY);
  // mex_param->set_ignore_inf_values(true);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(1e-3);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  mex_param->set_shared_offsets_region_size(2);
  mex_param->set_shared_offsets_region_c(2);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleMEX2DPoolingWithPartiallySharedOffsetsMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_block_c(1);
  block_param->set_stride(2);
  mex_param->set_num_instances(1);
  block_param->set_round_down(false);
  block_param->set_out_of_bounds_value(-INFINITY);
  // mex_param->set_ignore_inf_values(true);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  mex_param->set_shared_offsets_region_size(2);
  mex_param->set_shared_offsets_region_c(2);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleMEXWith2DSharedOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(3);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleMEXWith2DPartiallyUnsharedOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_use_unshared_regions(true);
  mex_param->set_unshared_offsets_region_size(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestSimpleMEXWith3DSharedOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_block_c(1);
  block_param->set_stride(2);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(3);
  mex_param->set_shared_offsets_region_c(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}


// Simple
TYPED_TEST(MEXLayerTest, TestMEX3x3Forward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX3x3FullyShared_Epsilon1) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
TYPED_TEST(MEXLayerTest, TestMEX3x3FullySharedNormedOffsets_Epsilon1) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX3x3ForwardNormalizeOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX3x3ForwardNormalizeOffsetsPartialSharingLinearSpace) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(2);
  mex_param->set_use_log_space_parameters(false);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  mex_param->mutable_offsets_filler()->set_type("uniform");
  mex_param->mutable_offsets_filler()->set_min(0);
  mex_param->mutable_offsets_filler()->set_max(10);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX3x3ForwardNormalizeOffsetsPartialUnsharedLinearSpace) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_use_unshared_regions(true);
  mex_param->set_unshared_offsets_region_size(2);
  mex_param->set_use_log_space_parameters(false);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  mex_param->mutable_offsets_filler()->set_type("uniform");
  mex_param->mutable_offsets_filler()->set_min(0);
  mex_param->mutable_offsets_filler()->set_max(10);
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX3x3ForwardNormalization) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->set_normalize_patches(true);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX3x3ForwardMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX3x3ForwardMAXNormalizeOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX3x3ForwardMIN) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(-INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

// 1x1
TYPED_TEST(MEXLayerTest, TestMEX1x1Forward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardPartialSharing) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardUnshared) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1FullyShared_Epsilon1) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
TYPED_TEST(MEXLayerTest, TestMEX1x1FullySharedNormedOffsets_Epsilon1) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardNormalizeOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardNormalizeOffsetsPartialSharing) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardNormalizeOffsetsUnshared) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardNormalization) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->set_normalize_patches(true);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardMAX) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardMAXNormalizeOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(MEXLayerTest, TestMEX1x1ForwardMIN) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(4);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(-INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new MEXLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_mex_forward(this->blob_bottom_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_mex_forward(this->blob_bottom_2_, mex_param, this->col_buffer_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

///////////// Test Gradient /////////////
TYPED_TEST(MEXLayerTest, TestGradientBasicMEX) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestGradientBasicMEXWithOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestGradientBasicMEXPoolingWithOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_block_c(1);
  block_param->set_stride(2);
  block_param->set_out_of_bounds_value(0);
  block_param->set_round_down(false);
  // mex_param->set_ignore_inf_values(true);
  mex_param->set_num_instances(1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(1e-1);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestGradientMEXWithOffsetsNormalization) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(2);
  mex_param->set_normalize_patches(true);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(MEXLayerTest, TestMEX3x3GradientNormalizeOffsetsPartialSharingLinearSpace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(2);
  mex_param->set_normalize_offsets(true);
  mex_param->set_use_log_space_parameters(false);
  mex_param->set_shared_offsets_region_size(2);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  mex_param->mutable_offsets_filler()->set_type("uniform");
  mex_param->mutable_offsets_filler()->set_min(0.1);
  mex_param->mutable_offsets_filler()->set_max(1);
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX3x3GradientNormalizeOffsetsPartialUnsharedLinearSpace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(2);
  mex_param->set_normalize_offsets(true);
  mex_param->set_use_log_space_parameters(false);
  mex_param->set_use_unshared_regions(true);
  mex_param->set_unshared_offsets_region_size(2);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(1);
  mex_param->mutable_offsets_filler()->set_type("uniform");
  mex_param->mutable_offsets_filler()->set_min(0.1);
  mex_param->mutable_offsets_filler()->set_max(1);
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestGradientMEXWithOffsetsMAX) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(2);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestGradientMEXWithOffsetsMIN) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(2);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(-INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(MEXLayerTest, TestMEX1x1Gradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientPartialSharing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_shared_offsets_region_size(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientUnshared) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_shared_offsets_region_size(1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientNormalization) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->set_normalize_patches(true);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 5e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, Test1x1MEXGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientNormalizeOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientNormalizeOffsetsPartialSharing) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientNormalizeOffsetsUnshared) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(1);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientMAX) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientMAXNormalizeOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_normalize_offsets(true);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestMEX1x1GradientMIN) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->set_shared_offsets_region_size(-1);
  mex_param->mutable_epsilon_filler()->set_type("constant");
  mex_param->mutable_epsilon_filler()->set_value(-INFINITY);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, Test1x1GradientMEX) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  mex_param->set_num_instances(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MEXLayerTest, TestGradientMEXSharedOffsets) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MEXParameter* mex_param =
      layer_param.mutable_mex_param();
  BlockParameter* block_param = mex_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  mex_param->set_num_instances(2);
  mex_param->set_shared_offsets_region_size(2);
  mex_param->mutable_epsilon_filler()->set_type("uniform");
  mex_param->mutable_epsilon_filler()->set_min(-10);
  mex_param->mutable_epsilon_filler()->set_max(10);
  mex_param->mutable_offsets_filler()->set_type("gaussian");
  MEXLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
