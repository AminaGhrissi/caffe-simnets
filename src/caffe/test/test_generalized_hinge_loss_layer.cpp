#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/generalized_hinge_loss_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class GeneralizedHingeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GeneralizedHingeLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 10, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
    ref_blob_top_.reset(new Blob<Dtype>());
    ref_blob_top_vec_.push_back(ref_blob_top_.get());
  }
  virtual ~GeneralizedHingeLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> ref_blob_top_vec_;
};

TYPED_TEST_CASE(GeneralizedHingeLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(GeneralizedHingeLossLayerTest, TestSoftmaxCase) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  //layer_param.add_loss_weight(3);
  GeneralizedHingeLossParameter* loss_param =
      layer_param.mutable_generalized_hinge_loss_param();
  loss_param->set_epsilon(1);
  loss_param->set_margin(0);
  GeneralizedHingeLossLayer<Dtype> layer(layer_param);
  SoftmaxWithLossLayer<Dtype> softmax(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against softmax layer.
  this->ref_blob_top_vec_.clear();
  this->ref_blob_top_vec_.push_back(this->MakeReferenceTop(this->blob_top_loss_));
  softmax.SetUp(this->blob_bottom_vec_, this->ref_blob_top_vec_);
  softmax.Forward(this->blob_bottom_vec_, this->ref_blob_top_vec_);
  const Dtype* top_data = this->blob_top_loss_->cpu_data();
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_loss_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}


TYPED_TEST(GeneralizedHingeLossLayerTest, TestSimpleGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  GeneralizedHingeLossParameter* loss_param =
      layer_param.mutable_generalized_hinge_loss_param();
  loss_param->set_epsilon(1);
  loss_param->set_margin(0);
  GeneralizedHingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GeneralizedHingeLossLayerTest, TestSimpleGradientWithNegativeEpsilon) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  GeneralizedHingeLossParameter* loss_param =
      layer_param.mutable_generalized_hinge_loss_param();
  loss_param->set_epsilon(-1);
  loss_param->set_margin(0);
  GeneralizedHingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GeneralizedHingeLossLayerTest, TestGradientWithMargin) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  GeneralizedHingeLossParameter* loss_param =
      layer_param.mutable_generalized_hinge_loss_param();
  loss_param->set_epsilon(1);
  loss_param->set_margin(1);
  GeneralizedHingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(GeneralizedHingeLossLayerTest, TestGradientRandomParameters) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  GeneralizedHingeLossParameter* loss_param =
      layer_param.mutable_generalized_hinge_loss_param();
  const float epsilon = 0.01 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(400.0)));
  const float margin = static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(100.0)));
  loss_param->set_epsilon(epsilon);
  loss_param->set_margin(margin);
  GeneralizedHingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
