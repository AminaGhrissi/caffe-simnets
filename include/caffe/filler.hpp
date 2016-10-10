// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  using std::min;
  using std::max;

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
  virtual void Fill_gpu(Blob<Dtype>* blob) {
    Fill(blob);
  };
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
  virtual void Fill_gpu(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_gpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    caffe_gpu_set(count, value, data);
    CHECK_EQ(this->filler_param_.sparse(), -1)
    << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
  virtual void Fill_gpu(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_gpu_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
                                 Dtype(this->filler_param_.max()), blob->mutable_gpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
    << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with randomly generated rectangles. Only supports use in
/// 4-D blobs (NxDxHxW), currently CPU-only.
template <typename Dtype>
class RectFiller : public Filler<Dtype> {
 public:
  explicit RectFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    CHECK_EQ(blob->num_axes(), 4)
      << "Filler only supports 4-D blobs.";
    CHECK_GT(blob->shape(0),0)
      << "Number of channels must be > 0";

    int min_rects = this->filler_param_.min_rects();
    int max_rects = this->filler_param_.max_rects();
    int min_width = this->filler_param_.min_width();
    int max_width = this->filler_param_.max_width();

    CHECK_GE(min_width, 0);
    CHECK_GE(min_rects, 0);
    CHECK_LE(min_width,max_width);
    CHECK_LE(min_rects,max_rects);

    Dtype* data = blob->mutable_cpu_data();
    for (int i = 0; i < blob->count(); ++i) {
      data[i] = 0;
    }

    if( max_rects > 0 && max_width > 0) {
      int num = blob->shape(0);
      int channels = blob->shape(1);
      int height = blob->shape(2);
      int width = blob->shape(3);
      int num_rects;
      caffe_rng_uniform_int(1,  min_rects,  max_rects, &num_rects);    
      for (int n = 0; n < num; ++n) {
        for (int i = 0; i < num_rects; ++i) {
          int px1, py1, px2, py2;
          int dx, dy;
          caffe_rng_uniform_int(1, 0, int(width - min(max(min_width, 1), width)) , &px1);
          caffe_rng_uniform_int(1, 0, int(height - min(max(min_width, 1), height)) , &py1);
          caffe_rng_uniform_int(1, 0, int(max(min(width - px1 - min_width, max_width - min_width), 0)) , &dx);
          caffe_rng_uniform_int(1, 0, int(max(min(height - py1 - min_width, max_width - min_width), 0)) , &dy);
          px2 = px1 + (min_width - 1) + dx;
          py2 = py1 + (min_width - 1) + dy;

          if (px1 <= px2 && py1 <= py2) {
            int top_index_n = n * channels;
            for (int c = 0; c < channels; ++c) {
              int top_index_c = (top_index_n + c) * height;
              for (int h = py1; h < py2; ++h) {
                int top_index_h = (top_index_c + h) * width;        
                for (int w = px1; w < px2; ++w) {
                  data[top_index_h + w] = 1;
                  
                }
              }
            }
          }  
        }
      }
    } 
    

    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
    if (this->filler_param_.to_log()) {
        caffe_scal<Dtype>(blob->count(), this->filler_param_.fudge_factor(), data);
    }
  }

};


/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class BernoulliFiller : public Filler<Dtype> {
 public:
  explicit BernoulliFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    Dtype* data = blob->mutable_cpu_data();
    
    Dtype non_zero_probability = this->filler_param_.non_zero_probability();
    if (this->filler_param_.use_range()) {
      CHECK_GE(this->filler_param_.min(), 0) << "Range min must be >= 0.";
      CHECK_GE(this->filler_param_.max(), this->filler_param_.min()) << "Range max must be >= min.";
      CHECK_LE(this->filler_param_.max(), 1) << "Range max must be <= 1.";
      caffe_rng_uniform(1, Dtype(this->filler_param_.min()), Dtype(this->filler_param_.max()), &non_zero_probability); 
    }
    if (this->filler_param_.to_log()) {
      non_zero_probability = Dtype(1) - non_zero_probability;
    }
    caffe_rng_bernoulli<Dtype, Dtype>(blob->count(), non_zero_probability, data);
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
    if (this->filler_param_.to_log()) {
        caffe_scal<Dtype>(blob->count(), this->filler_param_.fudge_factor(), data);
    }
  }
  virtual void Fill_gpu(Blob<Dtype>* blob) {
    CHECK(blob->count());
    Dtype* data = blob->mutable_gpu_data();
    Dtype non_zero_probability = this->filler_param_.non_zero_probability();
    if (this->filler_param_.use_range()) {
      CHECK_GE(this->filler_param_.min(), 0) << "Range min must be >= 0.";
      CHECK_GE(this->filler_param_.max(), this->filler_param_.min()) << "Range max must be >= min.";
      CHECK_LE(this->filler_param_.max(), 1) << "Range max must be <= 1.";
      caffe_gpu_rng_uniform(1, Dtype(this->filler_param_.min()), Dtype(this->filler_param_.max()), &non_zero_probability); 
    }
    if (this->filler_param_.to_log()) {
      non_zero_probability = Dtype(1) - non_zero_probability;
    }
    caffe_gpu_rng_bernoulli(blob->count(), non_zero_probability, data);
    CHECK_EQ(this->filler_param_.sparse(), -1)
    << "Sparsity not supported by this Filler.";
    if (this->filler_param_.to_log()) {
        caffe_gpu_scal<Dtype>(blob->count(), this->filler_param_.fudge_factor(), data);
    }
  }
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    if (this->filler_param_.std() == 0) {
      caffe_set<Dtype>(blob->count(), Dtype(this->filler_param_.mean()), data);
    } else {
      caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
          Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    }
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }
  virtual void Fill_gpu(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_gpu_data();
    CHECK(blob->count());
    if (this->filler_param_.std() == 0) {
      caffe_gpu_set<Dtype>(blob->count(), Dtype(this->filler_param_.mean()), data);
    } else {
      caffe_gpu_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
                                    Dtype(this->filler_param_.std()), data);
    }
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; height is number of inputs; width is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_EQ(blob->num(), 1);
      CHECK_EQ(blob->channels(), 1);
      int num_inputs = blob->height();
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_inputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(Dtype)));
      Dtype* mask = reinterpret_cast<Dtype*>(rand_vec_->mutable_gpu_data());
      caffe_gpu_rng_uniform<Dtype>(blob->count(), 0, 1, mask);
      caffe_gpu_mask<Dtype>(blob->count(), data, mask, non_zero_probability, data);
    }
  }
 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim, num;
    if (this->filler_param_.primal_dim() > 0) {
        dim = this->filler_param_.primal_dim();
        CHECK_EQ(blob->count() % dim, 0) << "The primal dimension must divide the number of parameters";
        num = blob->count() / dim;
    } else {
        dim = blob->count() / blob->num();
        num = blob->num();
    }
    CHECK(dim);CHECK(num);
    for (int i = 0; i < num; ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }
    if (this->filler_param_.to_log()) {
        for (int i = 0; i < blob->count(); ++i) {
            data[i] = std::log(data[i] + this->filler_param_.fudge_factor());
        }
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class DirichletFiller : public Filler<Dtype> {
 public:
  explicit DirichletFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_gamma<Dtype>(blob->count(), this->filler_param_.alpha(), blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim, num, filled_size;
    if (this->filler_param_.primal_dim() > 0) {
        dim = this->filler_param_.primal_dim();
        CHECK_EQ(blob->count() % dim, 0) << "The primal dimension must divide the number of parameters";
        num = blob->count() / dim;
        filled_size = this->filler_param_.filled_size();
        if (filled_size < 0) {
          filled_size = dim;
        }
        CHECK_LE(filled_size, dim) << "Filled size must be less than or equal to primal dimesion.";
    } else {
        dim = blob->count() / blob->num();
        num = blob->num();
        filled_size = dim;
    }
    for (int i = 0; i < num; ++i) {
      Dtype sum = 0;
      for (int j = 0; j < filled_size; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < filled_size; ++j) {
        data[i * dim + j] /= sum;
      }
      for (int j = filled_size; j < dim; ++j) {
        data[i * dim + j] = 0;
      }
    }
    if (this->filler_param_.to_log()) {
        caffe_log<Dtype>(blob->count(), data, data, this->filler_param_.fudge_factor());
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/** @brief Fills a Blob with values of indicator vectors in a cyclic fashion.
 *         If the blob is an sqaure matrix than this would be the identity transform.
 */
template <typename Dtype>
class IdentityFiller : public Filler<Dtype> {
 public:
  explicit IdentityFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_set<Dtype>(blob->count(), Dtype(0), data);
    int dim, num;
    if (this->filler_param_.primal_dim() > 0) {
        dim = this->filler_param_.primal_dim();
        CHECK_EQ(blob->count() % dim, 0) << "The primal dimension must divide the number of parameters";
        num = blob->count() / dim;
    } else {
        dim = blob->count() / blob->num();
        num = blob->num();
    }
    int cyclic_length = this->filler_param_.cyclic_length();
    if (cyclic_length < 0) {
      cyclic_length = dim;
    }
    for (int i = 0, k = 0; i < num; ++i) {
      data[i * dim + k] = Dtype(1);
      k = (k + 1) % cyclic_length;
    }
    if (this->filler_param_.to_log()) {
        caffe_log<Dtype>(blob->count(), data, data, this->filler_param_.fudge_factor());
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype scale = sqrt(Dtype(3) / n);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
  virtual void Fill_gpu(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
               FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype scale = sqrt(Dtype(3) / n);
    caffe_gpu_rng_uniform<Dtype>(blob->count(), -scale, scale,
                                 blob->mutable_gpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
    << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
 *        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
 *        nodes, outgoing nodes, or their average.
 *
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
 * the variance_norm option. You should make sure the input blob has shape (num,
 * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
 * is currently not the case for inner product layers.
 */
template <typename Dtype>
class MSRAFiller : public Filler<Dtype> {
 public:
  explicit MSRAFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype std = sqrt(Dtype(2) / n);
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/*!
@brief Fills a Blob with coefficients for bilinear interpolation.

A common use case is with the DeconvolutionLayer acting as upsampling.
You can upsample a feature map with shape of (B, C, H, W) by any integer factor
using the following proto.
\code
layer {
  name: "upsample", type: "Deconvolution"
  bottom: "{{bottom_name}}" top: "{{top_name}}"
  convolution_param {
    kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    num_output: {{C}} group: {{C}}
    pad: {{ceil((factor - 1) / 2.)}}
    weight_filler: { type: "bilinear" } bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
\endcode
Please use this by replacing `{{}}` with your values. By specifying
`num_output: {{C}} group: {{C}}`, it behaves as
channel-wise convolution. The filter shape of this deconvolution layer will be
(C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
interpolation kernel for every channel of the filter identically. The resulting
shape of the top feature map will be (B, C, factor * H, factor * W).
Note that the learning rate and the
weight decay are set to 0 in order to keep coefficient values of bilinear
interpolation unchanged during training. If you apply this to an image, this
operation is equivalent to the following call in Python with Scikit.Image.
\code{.py}
out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
\endcode
 */
template <typename Dtype>
class BilinearFiller : public Filler<Dtype> {
 public:
  explicit BilinearFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) {
      float x = i % blob->width();
      float y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else if (type == "msra") {
    return new MSRAFiller<Dtype>(param);
  } else if (type == "bilinear") {
    return new BilinearFiller<Dtype>(param);
  } else if (type == "dirichlet") {
    return new DirichletFiller<Dtype>(param);
  } else if (type == "identity") {
    return new IdentityFiller<Dtype>(param);
  } else if (type == "bernoulli") {
    return new BernoulliFiller<Dtype>(param);
  } else if (type == "rects") {
    return new RectFiller<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
