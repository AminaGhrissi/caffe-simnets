#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/similarity_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype> Dtype similarity_function(SimilarityParameter* similarity_param,
  const Dtype x, const Dtype z, const int K) {
  switch (similarity_param->similarity_function()) {
    case SimilarityParameter_SimilarityFunction_CONVOLUTION:
      return x * z;
      break;
    case SimilarityParameter_SimilarityFunction_L1:
      return -std::abs(x-z);
      break;
    case SimilarityParameter_SimilarityFunction_L2:
      return  -(x-z) * (x-z);
      break;
    default:
      return 0;
  }
}

// template float similarity_function(SimilarityParameter* similarity_param, const float x, const float z);
// template double similarity_function(SimilarityParameter* similarity_param, const double x, const double z);

// Reference similarity operator for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_sim(const Blob<Dtype>* in, SimilarityParameter* similarity_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int block_h, block_w;
  BlockParameter* block_param = similarity_param->mutable_block_param();
  const Dtype out_of_bounds_value = block_param->out_of_bounds_value();
  if (block_param->has_block_size()) {
    block_h = block_w = block_param->block_size();
  } else {
    block_h = block_param->block_h();
    block_w = block_param->block_w();
  }
  int pad_h, pad_w;
  if (!block_param->has_pad_h()) {
    pad_h = pad_w = block_param->pad();
  } else {
    pad_h = block_param->pad_h();
    pad_w = block_param->pad_w();
  }
  int stride_h, stride_w;
  if (!block_param->has_stride_h()) {
    stride_h = stride_w = block_param->stride();
  } else {
    stride_h = block_param->stride_h();
    stride_w = block_param->stride_w();
  }

  const bool normalize_patches = similarity_param->normalize_patches();
  const Dtype normalization_fudge_factor = similarity_param->normalization_fudge_factor();
  const bool normalize_variance = similarity_param->normalize_variance();
  const bool normalization_term = similarity_param->normalization_term();
  const Dtype normalization_term_fudge = similarity_param->normalization_term_fudge();
  const bool use_log_space_weight_param = similarity_param->use_log_space_weight_param();
  const bool ignore_nan_input = similarity_param->ignore_nan_input();
  // Groups
  int groups = 1;
  int o_g = out->channels() / groups;
  int k_g = in->channels() / groups;
  int o_head, k_head;
  const int patch_size = block_h * block_w * k_g;

  // Similarity
  const Dtype* in_data = in->cpu_data();
  const Dtype* template_data = weights[0]->cpu_data();
  const Dtype* weight_data = weights[1]->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int y = 0; y < out->height(); y++) {
          for (int x = 0; x < out->width(); x++) {
            Dtype mean = 0;
            Dtype var = 0;
            if (normalize_patches) {
              for (int k = 0; k < k_g; k++) {
                for (int p = 0; p < block_h; p++) {
                  for (int q = 0; q < block_w; q++) {
                    int in_y = y * stride_h - pad_h + p;
                    int in_x = x * stride_w - pad_w + q;
                    if (in_y >= 0 && in_y < in->height()
                      && in_x >= 0 && in_x < in->width()) {
                      mean += in_data[in->offset(n, k + k_head, in_y, in_x)];
                    } else {
                      mean += out_of_bounds_value;
                    }
                  }
                }
              }
              mean /= Dtype(patch_size);
              for (int k = 0; k < k_g; k++) {
                for (int p = 0; p < block_h; p++) {
                  for (int q = 0; q < block_w; q++) {
                    int in_y = y * stride_h - pad_h + p;
                    int in_x = x * stride_w - pad_w + q;
                    Dtype x_data = out_of_bounds_value;
                    if (in_y >= 0 && in_y < in->height()
                      && in_x >= 0 && in_x < in->width()) {
                      x_data = in_data[in->offset(n, k + k_head, in_y, in_x)];
                    }
                    var += (x_data - mean) * (x_data - mean);
                  }
                }
              }
              var /= Dtype(patch_size - 1);
            }

            for (int k = 0; k < k_g; k++) {
              for (int p = 0; p < block_h; p++) {
                for (int q = 0; q < block_w; q++) {
                  const Dtype u = weight_data[weights[1]->offset(o + o_head, k, p, q)];
                  Dtype u_weight = u;
                  if (use_log_space_weight_param) {
                    u_weight = std::exp(u);
                  }
                  const Dtype z = template_data[weights[0]->offset(o + o_head, k, p, q)];
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  Dtype pixel = out_of_bounds_value;
                  if (in_y >= 0 && in_y < in->height()
                    && in_x >= 0 && in_x < in->width()) {
                    pixel = in_data[in->offset(n, k + k_head, in_y, in_x)];
                  }
                  if (ignore_nan_input && std::isnan(pixel)) {
                    continue;
                  }
                  if (normalize_patches) {
                    pixel = pixel - mean;
                    if (normalize_variance) {
                      pixel = pixel / std::sqrt(var + normalization_fudge_factor);
                    } 
                  }
                  Dtype value = u_weight * similarity_function(similarity_param, pixel, z, k_g * block_h * block_w);
                  if (normalization_term) {
                    value *= 0.5;
                    if (use_log_space_weight_param) {
                      value += 0.5 * u
                             - 0.5 * std::log(2.0 * M_PI);
                    } else {
                      value += 0.5 * std::log(u + normalization_term_fudge) 
                             - 0.5 * std::log(2.0 * M_PI);
                    }
                  }
                  out_data[out->offset(n, o + o_head, y, x)] += value;
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (similarity_param->bias_term()) {
    const Dtype* bias_data = weights[2]->cpu_data();
    for (int n = 0; n < out->num(); n++) {
      for (int o = 0; o < out->channels(); o++) {
        for (int y = 0; y < out->height(); y++) {
          for (int x = 0; x < out->width(); x++) {
            out_data[out->offset(n, o, y, x)] += bias_data[o];
          }
        }
      }
    }
  }
}

template void caffe_sim(const Blob<float>* in,
    SimilarityParameter* similarity_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_sim(const Blob<double>* in,
    SimilarityParameter* similarity_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class SimilarityLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SimilarityLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SimilarityLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
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
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SimilarityLayerTest, TestDtypesAndDevices);

TYPED_TEST(SimilarityLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
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

TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("constant");
  similarity_param->mutable_weight_filler()->set_value(1.0);
  similarity_param->mutable_bias_filler()->set_type("constant");
  similarity_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityConvolution2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityL1) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L1);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityL2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityL2WithNormTerm) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->set_normalization_term(true);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityL2WithNormTermIgnoreNaN) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  block_param->set_pad(3);
  block_param->set_out_of_bounds_value(NAN);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->set_normalization_term(true);
  similarity_param->set_ignore_nan_input(true);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityL2LogSpaceWeight) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(4);
  similarity_param->set_use_log_space_weight_param(true);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("gaussian");
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityL2WithNormTermLogSpaceWeight) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(4);
  similarity_param->set_use_log_space_weight_param(true);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->set_normalization_term(true);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("gaussian");
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityL2WithNormTermLogSpaceWeightIgnoreNaN) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  block_param->set_pad(3);
  block_param->set_out_of_bounds_value(NAN);
  similarity_param->set_num_instances(4);
  similarity_param->set_use_log_space_weight_param(true);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->set_normalization_term(true);
  similarity_param->set_ignore_nan_input(true);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("gaussian");
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityNormConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("constant");
  similarity_param->mutable_weight_filler()->set_value(1.0);
  similarity_param->mutable_bias_filler()->set_type("constant");
  similarity_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityNormConvolution2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityNormL1) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L1);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityNormL2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestSimpleSimilarityNormNoVarianceL2) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_normalize_variance(false);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_sim(this->blob_bottom_2_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, Test1x1SimilarityConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("constant");
  similarity_param->mutable_weight_filler()->set_value(1.0);
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, Test1x1SimilarityConvolution2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, Test1x1SimilarityL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L1);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, Test1x1SimilarityL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(1);
  block_param->set_stride(1);
  similarity_param->set_num_instances(4);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_sim(this->blob_bottom_, similarity_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestSobelConvolutionSimilarity) {
  // Test separable convolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the convolution of two rectangular filters.
  typedef typename TypeParam::Dtype Dtype;
  // Fill bottoms with identical Gaussian noise.
  shared_ptr<GaussianFiller<Dtype> > filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new GaussianFiller<Dtype>(filler_param));
  filler->Fill(this->blob_bottom_);
  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
  // Compute Sobel G_x operator as 3 x 3 convolution.
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(1);
  similarity_param->set_bias_term(false);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->blobs().resize(2);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 3));
  layer->blobs()[1].reset(new Blob<Dtype>(1, 3, 3, 3));
  Dtype* templates = layer->blobs()[0]->mutable_cpu_data();
  Dtype* weights = layer->blobs()[1]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 9;  // 3 x 3 filter
    templates[i +  0] = -1;
    templates[i +  1] =  0;
    templates[i +  2] =  1;
    templates[i +  3] = -2;
    templates[i +  4] =  0;
    templates[i +  5] =  2;
    templates[i +  6] = -1;
    templates[i +  7] =  0;
    templates[i +  8] =  1;
    weights[i +  0] = 1;
    weights[i +  1] = 1;
    weights[i +  2] = 1;
    weights[i +  3] = 1;
    weights[i +  4] = 1;
    weights[i +  5] = 1;
    weights[i +  6] = 1;
    weights[i +  7] = 1;
    weights[i +  8] = 1;
  }
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convolutions.
  // (1) the [1 2 1] column filter
  vector<Blob<Dtype>*> sep_blob_bottom_vec;
  vector<Blob<Dtype>*> sep_blob_top_vec;
  shared_ptr<Blob<Dtype> > blob_sep(new Blob<Dtype>());
  sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
  sep_blob_top_vec.push_back(this->blob_top_2_);
  block_param->clear_block_size();
  block_param->clear_stride();
  block_param->set_block_h(3);
  block_param->set_block_w(1);
  block_param->set_stride_h(2);
  block_param->set_stride_w(1);
  similarity_param->set_num_instances(1);
  similarity_param->set_bias_term(false);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  layer.reset(new SimilarityLayer<Dtype>(layer_param));
  layer->blobs().resize(2);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 1));
  layer->blobs()[1].reset(new Blob<Dtype>(1, 3, 3, 1));
  Dtype* templates_1 = layer->blobs()[0]->mutable_cpu_data();
  Dtype* weights_1 = layer->blobs()[1]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 3 x 1 filter
    templates_1[i +  0] = 1;
    templates_1[i +  1] = 2;
    templates_1[i +  2] = 1;
    weights_1[i +  0] = 1;
    weights_1[i +  1] = 1;
    weights_1[i +  2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
  // (2) the [-1 0 1] row filter
  blob_sep->CopyFrom(*this->blob_top_2_, false, true);
  sep_blob_bottom_vec.clear();
  sep_blob_bottom_vec.push_back(blob_sep.get());
  block_param->set_block_h(1);
  block_param->set_block_w(3);
  block_param->set_stride_h(1);
  block_param->set_stride_w(2);
  similarity_param->set_num_instances(1);
  similarity_param->set_bias_term(false);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  layer.reset(new SimilarityLayer<Dtype>(layer_param));
  layer->blobs().resize(22);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 1, 3));
  layer->blobs()[1].reset(new Blob<Dtype>(1, 3, 1, 3));
  Dtype* templates_2 = layer->blobs()[0]->mutable_cpu_data();
  Dtype* weights_2 = layer->blobs()[1]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 1 x 3 filter
    templates_2[i +  0] = -1;
    templates_2[i +  1] =  0;
    templates_2[i +  2] =  1;
    weights_2[i +  0] = 1;
    weights_2[i +  1] = 1;
    weights_2[i +  2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
  // Test equivalence of full and separable filters.
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* sep_top_data = this->blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
  }
}

TYPED_TEST(SimilarityLayerTest, TestManualL1Similarity) {
  // Test separable convolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the convolution of two rectangular filters.
  typedef typename TypeParam::Dtype Dtype;
  // Fill bottoms with identical Gaussian noise.
  shared_ptr<GaussianFiller<Dtype> > filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new GaussianFiller<Dtype>(filler_param));
  filler->Fill(this->blob_bottom_);
  Dtype* bottom = this->blob_bottom_->mutable_cpu_data();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) { // 2, 3, 6, 4
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      //  1  2  3  4
      //  1  2  3  4
      //  1  2  3  4
      //  1  2  3  4
      //  1  2  3  4
      //  1  2  3  4
      for (int i = 0; i < this->blob_bottom_->width() * this->blob_bottom_->height(); ++i) {
        bottom[this->blob_bottom_->offset(n, c) + i] = (i % 4) + 1;
      }
    }
  }
  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
  // Compute Sobel G_x operator as 3 x 3 convolution.
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  block_param->set_block_size(2);
  block_param->set_stride(1);
  similarity_param->set_num_instances(1);
  similarity_param->set_bias_term(false);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L1);
  shared_ptr<Layer<Dtype> > layer(
      new SimilarityLayer<Dtype>(layer_param));
  layer->blobs().resize(2);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 2, 2));
  layer->blobs()[1].reset(new Blob<Dtype>(1, 3, 2, 2));
  Dtype* templates = layer->blobs()[0]->mutable_cpu_data();
  Dtype* weights = layer->blobs()[1]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 4;  // 2 x 2 filter
    weights[i +  0] = 1;
    weights[i +  1] =  1;
    weights[i +  2] =  0.5;
    weights[i +  3] = 0.5;
    templates[i +  0] = -1;
    templates[i +  1] = 1;
    templates[i +  2] = -2;
    templates[i +  3] = 3;
  }
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  // Test equivalence of full and separable filters.
  const Dtype* top_data = this->blob_top_->cpu_data();
  //  -15  -18  -30
  //  -15  -18  -30
  //  -15  -18  -30
  //  -15  -18  -30
  //  -15  -18  -30
  for (int i = 0; i < this->blob_top_->count(); ++i) { // 2, 1, 3, 2
    if (i % 3 == 0) {
      EXPECT_NEAR(top_data[i], -15, 1e-4);
    } else if (i % 3 == 1){ 
      EXPECT_NEAR(top_data[i], -21, 1e-4);
    } else {
      EXPECT_NEAR(top_data[i], -30, 1e-4);
    }
  }
}

///////////// Test Convolution /////////////
TYPED_TEST(SimilarityLayerTest, TestGradientConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientNormConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, Test1x1GradientConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_CONVOLUTION);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

///////////// Test L1 Similarity /////////////
TYPED_TEST(SimilarityLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L1);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-2, 1701, 0., 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientNormL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L1);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 5e-2, 2234, 0., 0.1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, Test1x1GradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L1);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-2, 1701, 0., 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

///////////// Test L2 Similarity /////////////
TYPED_TEST(SimilarityLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientL2WithNormTerm) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->set_normalization_term(true);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientL2WithNormTermIgnoreNaN) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  block_param->set_pad(3);
  block_param->set_out_of_bounds_value(NAN);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->set_normalization_term(true);
  similarity_param->set_ignore_nan_input(true);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientL2LogSpaceWeight) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(2);
  similarity_param->set_use_log_space_weight_param(true);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("gaussian");
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 5e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientL2WithNormTermLogSpaceWeight) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_num_instances(2);
  similarity_param->set_use_log_space_weight_param(true);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->set_normalization_term(true);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("gaussian");
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 5e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientL2WithNormTermLogSpaceWeightIgnoreNaN) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  block_param->set_pad(3);
  block_param->set_out_of_bounds_value(NAN);
  similarity_param->set_num_instances(2);
  similarity_param->set_use_log_space_weight_param(true);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->set_normalization_term(true);
  similarity_param->set_ignore_nan_input(true);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("gaussian");
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 5e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientNormL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, TestGradientNormNoVarianceL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(3);
  block_param->set_stride(2);
  similarity_param->set_normalize_patches(true);
  similarity_param->set_normalize_variance(false);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
  similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(SimilarityLayerTest, Test1x1GradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SimilarityParameter* similarity_param =
      layer_param.mutable_similarity_param();
  BlockParameter* block_param = similarity_param->mutable_block_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  block_param->set_block_size(1);
  block_param->set_stride(1);
  similarity_param->set_num_instances(2);
  similarity_param->set_similarity_function(SimilarityParameter_SimilarityFunction_L2);
  similarity_param->mutable_template_filler()->set_type("gaussian");
    similarity_param->mutable_weight_filler()->set_type("uniform");
  similarity_param->mutable_weight_filler()->set_min(0.1);
  similarity_param->mutable_weight_filler()->set_max(1.9);
  similarity_param->mutable_bias_filler()->set_type("gaussian");
  SimilarityLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
