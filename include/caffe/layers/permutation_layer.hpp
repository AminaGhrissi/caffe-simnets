#ifndef CAFFE_PERMUTATION_LAYER_HPP_
#define CAFFE_PERMUTATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Permutes the image pixels acording to the scheme specified.
 *
 * 
 */
template <typename Dtype>
class PermutationLayer : public Layer<Dtype> {
 public:
  explicit PermutationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Permutation"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }
  

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// @brief the type of permutation operation performed by the layer
  PermutationParameter_PermutationType type_;

  // To hold lookup tables for permutation, for use where permutation is explicitly 
  // listed, such as in GEN case.
  // src location to dst location (e.g, send location 1 to location 6)
  vector<int> permute_ltable_; 
  // dst location to src location (this is exactly the permute string input 
  // by the user- "1,3,2,0" means send 1 to location 0, 3 to loc. 1, 2->2 etc)
  vector<int> inv_permute_ltable_; 
  
};

}  // namespace caffe

#endif  // CAFFE_PERMUTATION_LAYER_HPP_
