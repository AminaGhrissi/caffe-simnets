#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/permutation_layer.hpp"

// #ifdef USE_CUDNN
// #include "caffe/layers/cudnn_permutation_layer.hpp"
// #endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PermutationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PermutationLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PermutationLayerTest() {
    delete blob_bottom_;
    delete blob_top_;

  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
//   // Test 2x2 quadrant pooling on rectangular input w>h
  void TestQuad2RectWide() {
    LayerParameter layer_param;
    // PermutationParameter* permutation_param = layer_param.mutable_permutation_param();

    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [0 1 2 3 4]
    //     [5 6 7 8 9]
    //     [10 11 12 13 14]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 0;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 2;
      blob_bottom_->mutable_cpu_data()[i +  3] = 3;
      blob_bottom_->mutable_cpu_data()[i +  4] = 4;
      blob_bottom_->mutable_cpu_data()[i +  5] = 5;
      blob_bottom_->mutable_cpu_data()[i +  6] = 6;
      blob_bottom_->mutable_cpu_data()[i +  7] = 7;
      blob_bottom_->mutable_cpu_data()[i +  8] = 8;
      blob_bottom_->mutable_cpu_data()[i +  9] = 9;
      blob_bottom_->mutable_cpu_data()[i + 10] = 10;
      blob_bottom_->mutable_cpu_data()[i + 11] = 11;
      blob_bottom_->mutable_cpu_data()[i + 12] = 12;
      blob_bottom_->mutable_cpu_data()[i + 13] = 13;
      blob_bottom_->mutable_cpu_data()[i + 14] = 14;
    }
    PermutationLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [  0   4   1   3   2],
    // [ 10  14  11  13  12],
    // [  5   9   6   8   7]

    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 14);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 13);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 7);
    }

  }
//   // Test 2x2 quadrant pooling on rectangular input h > w
  void TestQuad2RectHigh() {
    LayerParameter layer_param;
    // PermutationParameter* permutation_param = layer_param.mutable_permutation_param();

    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 5, 3);
    // Input: 2x 2 channels of:
    //     [0 1 2 ]
    //     [3 4 5 ]
    //     [6 7 8 ]
    //     [9 10 11]
    //     [12 13 14]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 0;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 2;
      blob_bottom_->mutable_cpu_data()[i +  3] = 3;
      blob_bottom_->mutable_cpu_data()[i +  4] = 4;
      blob_bottom_->mutable_cpu_data()[i +  5] = 5;
      blob_bottom_->mutable_cpu_data()[i +  6] = 6;
      blob_bottom_->mutable_cpu_data()[i +  7] = 7;
      blob_bottom_->mutable_cpu_data()[i +  8] = 8;
      blob_bottom_->mutable_cpu_data()[i +  9] = 9;
      blob_bottom_->mutable_cpu_data()[i + 10] = 10;
      blob_bottom_->mutable_cpu_data()[i + 11] = 11;
      blob_bottom_->mutable_cpu_data()[i + 12] = 12;
      blob_bottom_->mutable_cpu_data()[i + 13] = 13;
      blob_bottom_->mutable_cpu_data()[i + 14] = 14;
    }
    PermutationLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 5);
    EXPECT_EQ(blob_top_->width(), 3);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
//  [ 0  2  1]
//  [12 14 13]
//  [ 3  5  4]
//  [ 9 11 10]
//  [ 6  8  7]

    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 14);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 13);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 7);
    }

  }
  void TestQuad2Square() {
    LayerParameter layer_param;
    // PermutationParameter* permutation_param = layer_param.mutable_permutation_param();

    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [ 0  1  2  3  4  5]
    //  [ 6  7  8  9 10 11]
    //  [12 13 14 15 16 17]
    //  [18 19 20 21 22 23]
    //  [24 25 26 27 28 29]
    //  [30 31 32 33 34 35]
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 0;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 2;
      blob_bottom_->mutable_cpu_data()[i +  3] = 3;
      blob_bottom_->mutable_cpu_data()[i +  4] = 4;
      blob_bottom_->mutable_cpu_data()[i +  5] = 5;
      blob_bottom_->mutable_cpu_data()[i +  6] = 6;
      blob_bottom_->mutable_cpu_data()[i +  7] = 7;
      blob_bottom_->mutable_cpu_data()[i +  8] = 8;
      blob_bottom_->mutable_cpu_data()[i +  9] = 9;
      blob_bottom_->mutable_cpu_data()[i + 10] = 10;
      blob_bottom_->mutable_cpu_data()[i + 11] = 11;
      blob_bottom_->mutable_cpu_data()[i + 12] = 12;
      blob_bottom_->mutable_cpu_data()[i + 13] = 13;
      blob_bottom_->mutable_cpu_data()[i + 14] = 14;
      blob_bottom_->mutable_cpu_data()[i + 15] = 15;
      blob_bottom_->mutable_cpu_data()[i + 16] = 16;
      blob_bottom_->mutable_cpu_data()[i + 17] = 17;
      blob_bottom_->mutable_cpu_data()[i + 18] = 18;
      blob_bottom_->mutable_cpu_data()[i + 19] = 19;
      blob_bottom_->mutable_cpu_data()[i + 20] = 20;
      blob_bottom_->mutable_cpu_data()[i + 21] = 21;
      blob_bottom_->mutable_cpu_data()[i + 22] = 22;
      blob_bottom_->mutable_cpu_data()[i + 23] = 23;
      blob_bottom_->mutable_cpu_data()[i + 24] = 24;
      blob_bottom_->mutable_cpu_data()[i + 25] = 25;
      blob_bottom_->mutable_cpu_data()[i + 26] = 26;
      blob_bottom_->mutable_cpu_data()[i + 27] = 27;
      blob_bottom_->mutable_cpu_data()[i + 28] = 28;
      blob_bottom_->mutable_cpu_data()[i + 29] = 29;
      blob_bottom_->mutable_cpu_data()[i + 30] = 30;
      blob_bottom_->mutable_cpu_data()[i + 31] = 31;
      blob_bottom_->mutable_cpu_data()[i + 32] = 32;
      blob_bottom_->mutable_cpu_data()[i + 33] = 33;
      blob_bottom_->mutable_cpu_data()[i + 34] = 34;
      blob_bottom_->mutable_cpu_data()[i + 35] = 35;
    }
    PermutationLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 6);
    EXPECT_EQ(blob_top_->width(), 6);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //  [ 0  5  1  4  2  3]
    //  [30 35 31 34 32 33]
    //  [ 6 11  7 10  8  9]
    //  [24 29 25 28 26 27]
    //  [12 17 13 16 14 15]
    //  [18 23 19 22 20 21]
    for (int i = 0; i < 20 * num * channels; i += 36) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 30);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 24);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 29);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 25);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 28);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 24], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 25], 17);
      EXPECT_EQ(blob_top_->cpu_data()[i + 26], 13);
      EXPECT_EQ(blob_top_->cpu_data()[i + 27], 16);
      EXPECT_EQ(blob_top_->cpu_data()[i + 28], 14);
      EXPECT_EQ(blob_top_->cpu_data()[i + 29], 15);
      EXPECT_EQ(blob_top_->cpu_data()[i + 30], 18);
      EXPECT_EQ(blob_top_->cpu_data()[i + 31], 23);
      EXPECT_EQ(blob_top_->cpu_data()[i + 32], 19);
      EXPECT_EQ(blob_top_->cpu_data()[i + 33], 22);
      EXPECT_EQ(blob_top_->cpu_data()[i + 34], 20);
      EXPECT_EQ(blob_top_->cpu_data()[i + 35], 21);

    }

  }


//   // Test general permutation on rectangular input w>h
  void TestGenRectWide() {
    LayerParameter layer_param;
    PermutationParameter* permutation_param = layer_param.mutable_permutation_param();
    permutation_param->set_type(PermutationParameter_PermutationType_GEN);
    permutation_param->set_permute_string("6,5,3,4,13,12,1,11,7,8,10,0,2,14,9");
    PermutationLayer<Dtype> layer(layer_param);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [0 1 2 3 4]
    //     [5 6 7 8 9]
    //     [10 11 12 13 14]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 0;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 2;
      blob_bottom_->mutable_cpu_data()[i +  3] = 3;
      blob_bottom_->mutable_cpu_data()[i +  4] = 4;
      blob_bottom_->mutable_cpu_data()[i +  5] = 5;
      blob_bottom_->mutable_cpu_data()[i +  6] = 6;
      blob_bottom_->mutable_cpu_data()[i +  7] = 7;
      blob_bottom_->mutable_cpu_data()[i +  8] = 8;
      blob_bottom_->mutable_cpu_data()[i +  9] = 9;
      blob_bottom_->mutable_cpu_data()[i + 10] = 10;
      blob_bottom_->mutable_cpu_data()[i + 11] = 11;
      blob_bottom_->mutable_cpu_data()[i + 12] = 12;
      blob_bottom_->mutable_cpu_data()[i + 13] = 13;
      blob_bottom_->mutable_cpu_data()[i + 14] = 14;
    }

    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 3);
    EXPECT_EQ(blob_top_->width(), 5);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [ 6,  5,  3,  4, 13],
    // [12,  1, 11,  7,  8],
    // [10,  0,  2, 14,  9]

    for (int i = 0; i < 15 * num * channels; i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 13);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 14);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 9);
    }

  }
//   // Test general permutation on square input
 
  void TestGenSquare() {
    LayerParameter layer_param;
    PermutationParameter* permutation_param = layer_param.mutable_permutation_param();
    permutation_param->set_type(PermutationParameter_PermutationType_GEN);
    permutation_param->set_permute_string("6,1,7,9,4,10,0,2,15,3,5,16,13,12,14,8,11,17,24,20,19,22,21,28,18,30,31,32,23,35,25,26,27,34,33,29");
    PermutationLayer<Dtype> layer(layer_param);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [ 0  1  2  3  4  5]
    //  [ 6  7  8  9 10 11]
    //  [12 13 14 15 16 17]
    //  [18 19 20 21 22 23]
    //  [24 25 26 27 28 29]
    //  [30 31 32 33 34 35]
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 0;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 2;
      blob_bottom_->mutable_cpu_data()[i +  3] = 3;
      blob_bottom_->mutable_cpu_data()[i +  4] = 4;
      blob_bottom_->mutable_cpu_data()[i +  5] = 5;
      blob_bottom_->mutable_cpu_data()[i +  6] = 6;
      blob_bottom_->mutable_cpu_data()[i +  7] = 7;
      blob_bottom_->mutable_cpu_data()[i +  8] = 8;
      blob_bottom_->mutable_cpu_data()[i +  9] = 9;
      blob_bottom_->mutable_cpu_data()[i + 10] = 10;
      blob_bottom_->mutable_cpu_data()[i + 11] = 11;
      blob_bottom_->mutable_cpu_data()[i + 12] = 12;
      blob_bottom_->mutable_cpu_data()[i + 13] = 13;
      blob_bottom_->mutable_cpu_data()[i + 14] = 14;
      blob_bottom_->mutable_cpu_data()[i + 15] = 15;
      blob_bottom_->mutable_cpu_data()[i + 16] = 16;
      blob_bottom_->mutable_cpu_data()[i + 17] = 17;
      blob_bottom_->mutable_cpu_data()[i + 18] = 18;
      blob_bottom_->mutable_cpu_data()[i + 19] = 19;
      blob_bottom_->mutable_cpu_data()[i + 20] = 20;
      blob_bottom_->mutable_cpu_data()[i + 21] = 21;
      blob_bottom_->mutable_cpu_data()[i + 22] = 22;
      blob_bottom_->mutable_cpu_data()[i + 23] = 23;
      blob_bottom_->mutable_cpu_data()[i + 24] = 24;
      blob_bottom_->mutable_cpu_data()[i + 25] = 25;
      blob_bottom_->mutable_cpu_data()[i + 26] = 26;
      blob_bottom_->mutable_cpu_data()[i + 27] = 27;
      blob_bottom_->mutable_cpu_data()[i + 28] = 28;
      blob_bottom_->mutable_cpu_data()[i + 29] = 29;
      blob_bottom_->mutable_cpu_data()[i + 30] = 30;
      blob_bottom_->mutable_cpu_data()[i + 31] = 31;
      blob_bottom_->mutable_cpu_data()[i + 32] = 32;
      blob_bottom_->mutable_cpu_data()[i + 33] = 33;
      blob_bottom_->mutable_cpu_data()[i + 34] = 34;
      blob_bottom_->mutable_cpu_data()[i + 35] = 35;
    }

    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 6);
    EXPECT_EQ(blob_top_->width(), 6);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
      // [ 6,  1,  7,  9,  4, 10],
      // [ 0,  2, 15,  3,  5, 16],
      // [13, 12, 14,  8, 11, 17],
      // [24, 20, 19, 22, 21, 28],
      // [18, 30, 31, 32, 23, 35],
      // [25, 26, 27, 34, 33, 29]
    for (int i = 0; i < 20 * num * channels; i += 36) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 15);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 16);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 13);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 14);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 17);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 24);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 20);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 19);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 22);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 21);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 28);
      EXPECT_EQ(blob_top_->cpu_data()[i + 24], 18);
      EXPECT_EQ(blob_top_->cpu_data()[i + 25], 30);
      EXPECT_EQ(blob_top_->cpu_data()[i + 26], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 27], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i + 28], 23);
      EXPECT_EQ(blob_top_->cpu_data()[i + 29], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i + 30], 25);
      EXPECT_EQ(blob_top_->cpu_data()[i + 31], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i + 32], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 33], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 34], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 35], 29);

    }

  }

};  

TYPED_TEST_CASE(PermutationLayerTest, TestDtypesAndDevices);

TYPED_TEST(PermutationLayerTest, TestSetupQUAD_2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermutationParameter* permutation_param = layer_param.mutable_permutation_param();
  permutation_param->set_type(PermutationParameter_PermutationType_QUAD2);
  PermutationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(PermutationLayerTest, TestSetup_GEN) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PermutationParameter* permutation_param = layer_param.mutable_permutation_param();
  permutation_param->set_type(PermutationParameter_PermutationType_GEN);
  permutation_param->set_permute_string("0,1,2,3,4,5,6,7,8,10,11,9,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29");
  PermutationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}




TYPED_TEST(PermutationLayerTest, TestForward_Quad2) {
  this->TestQuad2Square();
  this->TestQuad2RectHigh();
  this->TestQuad2RectWide();
}


TYPED_TEST(PermutationLayerTest, TestGradientQuad2) {
  typedef typename TypeParam::Dtype Dtype;
  
  LayerParameter layer_param;
  PermutationParameter* permutation_param = layer_param.mutable_permutation_param();
  permutation_param->set_type(PermutationParameter_PermutationType_QUAD2);
  PermutationLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);



}

TYPED_TEST(PermutationLayerTest, TestForward_Gen) {
  this->TestGenSquare();
  this->TestGenRectWide();
}


TYPED_TEST(PermutationLayerTest, TestGradientGen) {
  typedef typename TypeParam::Dtype Dtype;
  
  LayerParameter layer_param;
  PermutationParameter* permutation_param = layer_param.mutable_permutation_param();
  permutation_param->set_type(PermutationParameter_PermutationType_GEN);
  permutation_param->set_permute_string("0,1,2,3,4,5,6,7,8,10,11,9,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29");
  PermutationLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);



}



}  // namespace caffe