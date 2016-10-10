// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_UNSUPERVISED_LEARNER_HPP
#define CAFFE_UNSUPERVISED_LEARNER_HPP

#include <vector>
#include "caffe/blob.hpp"

namespace caffe {

/// @brief Learns parameters using unsupervised methods (K-Means, GMM, etc.)
template <typename Dtype>
class UnsupervisedLearner {
 public:
  explicit UnsupervisedLearner() {}
  virtual ~UnsupervisedLearner() {}
  /**
   * Performs a single step of the unsupervised learner with the given mini-batch.
   * @param  input     Mini-batch of examples. Each row an example.
   * @param  objective An output variable for returning the mini-batch objective. Can be NULL.
   * @return           True if the learner hasn't fininshed. False if the learner is done.
   */
  virtual bool step_cpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective) = 0;
  virtual bool step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective) = 0;
  virtual void fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) = 0;
  virtual void fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs) = 0;
  virtual Dtype objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input) = 0;
  virtual Dtype objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input) = 0;
 protected:
  virtual void setup(const vector<shared_ptr<Blob<Dtype> > >& input) = 0;
};  // class Filler


/// @brief Learn K clusters using mini-batch k-means algorithm.
template <typename Dtype>
class KmeansLearner : public UnsupervisedLearner<Dtype> {
 public:
  explicit KmeansLearner(const int num_clusters, const int max_kmeans_iterations, const int num_batches_, 
      const float prob_choose_centroid, const bool use_kmeans_plus_plus, const Dtype fudge_factor = 0.1) : 
      num_clusters_(num_clusters), max_iterations_(max_kmeans_iterations), num_batches_(num_batches_), 
      prob_choose_centroid_(prob_choose_centroid), iter_(0), num_init_clusters_(0), init_clusters_(false), 
      called_setup_(false), use_kmeans_plus_plus_(use_kmeans_plus_plus), fudge_factor_(fudge_factor) {}
  virtual bool step_cpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective);
  virtual bool step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective);
  virtual void fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs);
  virtual void fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs);
  virtual Dtype objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input);
  virtual Dtype objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input);
 protected:
  virtual void setup(const vector<shared_ptr<Blob<Dtype> > >& input);
  bool random_initialization_cpu(const shared_ptr<Blob<Dtype> >& input);
  bool sample_kmeans_plus_plus_cpu(const shared_ptr<Blob<Dtype> >& input);
  void assign_to_clusters_cpu(const shared_ptr<Blob<Dtype> >& input, Dtype* objective);
  void update_clusters_cpu(const shared_ptr<Blob<Dtype> >& input);
  void update_variances_cpu(const shared_ptr<Blob<Dtype> >& input);
  bool random_initialization_gpu(const shared_ptr<Blob<Dtype> >& input);
  bool sample_kmeans_plus_plus_gpu(const shared_ptr<Blob<Dtype> >& input);
  void assign_to_clusters_gpu(const shared_ptr<Blob<Dtype> >& input, Dtype* objective);
  void update_clusters_gpu(const shared_ptr<Blob<Dtype> >& input);
  void update_variances_gpu(const shared_ptr<Blob<Dtype> >& input);
  Blob<Dtype> clusters_;
  Blob<Dtype> assignments_;
  Blob<Dtype> per_center_count_;
  Blob<Dtype> batch_size_ones_;
  const int num_clusters_;
  const int max_iterations_;
  const int num_batches_;
  const float prob_choose_centroid_;
  int iter_;
  int batch_size_;
  int dim_;
  int num_init_clusters_;
  bool init_clusters_;
  bool called_setup_;
  const bool use_kmeans_plus_plus_;
  const Dtype fudge_factor_;
};

/// @brief Learn K clusters with diagnoal covariance matrix using FREM (Fast and Robust EM) algorithm.
/// @remark Assumes inputs are whitened
template <typename Dtype>
class GMMLearner : public UnsupervisedLearner<Dtype> {
 public:
  explicit GMMLearner(const int num_clusters, const int max_epoch_iterations, const int num_batches_, 
    const Dtype fudge_factor = 0.01, const bool soft_kmeans = false, const int mstep_frequency = -1.0,
    const Dtype split_cluster_threshold = 0.2, const Dtype init_cluster_scale_factor = 1e-1,
    const Dtype convergence_threshold = 5e-3,
    const int kmeans_iterations = 1000, const float prob_choose_centroid = 1e-5) : 
      num_clusters_(num_clusters), max_iterations_(max_epoch_iterations), num_batches_(num_batches_), 
      fudge_factor_(fudge_factor), soft_kmeans_(soft_kmeans), mstep_frequency_(mstep_frequency),
      split_cluster_threshold_(split_cluster_threshold),
      init_cluster_scale_factor_(init_cluster_scale_factor), convergence_threshold_(convergence_threshold),
      iter_(0), epoch_iter_(0), last_mstep_iter_(0), called_setup_(false),
      did_finished_kmeans_init_(false),
      kmeans_init_(num_clusters, kmeans_iterations, 0, prob_choose_centroid, true) {}
  virtual bool step_cpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective);
  virtual bool step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective);
  virtual void fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs);
  virtual void fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs);
  virtual Dtype objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input);
  virtual Dtype objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input);
 protected:
  // Allocate matrices and initialize model parameters
  virtual void setup(const vector<shared_ptr<Blob<Dtype> > >& input);
  // Perform E step by updating sufficient statistics
  virtual void estep_cpu(const shared_ptr<Blob<Dtype> >& input, Dtype* objective);
  virtual void estep_gpu(const shared_ptr<Blob<Dtype> >& input, Dtype* objective);
  // Perform M step using the sufficient statistics
  virtual void mstep_cpu();
  virtual void mstep_gpu();
  // Split large clusters if needed
  virtual void split_clusters_cpu();
  virtual void split_clusters_gpu();
  // Tie all variances to the average variances per cluster
  virtual void tie_variances_cpu();
  virtual void tie_variances_gpu();

  Blob<Dtype> clusters_; // Matrix containing the cluster's centers
  Blob<Dtype> variances_; // Matrix containin the per-dimension cluster's variances
  Blob<Dtype> log_norm_factor_; // The normalization factor for each cluster in log space
  Blob<Dtype> cluster_weights_; // Matrix containing the probability weights of the mixture

  Blob<Dtype> M_; // sum of inputs per cluster
  Blob<Dtype> Q_; // sum of squared inputs per cluster
  Blob<Dtype> N_; // number of inputs per cluster

  Blob<Dtype> distances_;
  Blob<Dtype> assignments_;
  Blob<Dtype> clusters_helper_;
  Blob<Dtype> batch_size_ones_;

  // Parameters
  const int num_clusters_; // Number of desired clusters
  const int max_iterations_; // Maximum number of epoch iterations
  const int num_batches_; // Number of batches per epoch
  const Dtype fudge_factor_; // Fudge factor used to smooth variances.
  const bool soft_kmeans_;
  int mstep_frequency_; // After how many esteps should mstep be applied
  const Dtype split_cluster_threshold_; // The minimum threshold the weight 
                                        // of a cluster can before splitting occurs
  Dtype init_cluster_scale_factor_; // How much pertubation should be used to init the clusters
  const int convergence_threshold_; // The threshold for convergence
  
  int iter_; // Iteration per example
  int epoch_iter_; // Iteration per epoch
  int last_mstep_iter_; // The last iteration mstep was calculated
  bool called_setup_;
  Dtype previous_log_likelihood_;
  Dtype current_log_likelihood_;
  Dtype current_kmeans_objective_;
  int batch_size_; // The number of examples in each batch
  int dim_; // The dimension of the data
  bool did_finished_kmeans_init_;
  KmeansLearner<Dtype> kmeans_init_;
};

/// @brief Learn PCA 
template <typename Dtype>
class PCALearner : public UnsupervisedLearner<Dtype> {
 public:
  explicit PCALearner(const int out_dim, const int num_batches, const bool apply_whitening,
    const bool zca_whitening, const Dtype fudge_factor) : 
      out_dim_(out_dim), num_batches_(num_batches), iter_(0),
      apply_whitening_(apply_whitening), zca_whitening_(zca_whitening),
      calculated_mean_(false), called_setup_(false), fudge_factor_(fudge_factor) {}
  virtual bool step_cpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective);
  virtual bool step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective);
  virtual void fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs);
  virtual void fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs);
  virtual Dtype objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input);
  virtual Dtype objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input);
 protected:
  virtual void setup(const vector<shared_ptr<Blob<Dtype> > >& input);
  void update_mean_cpu(const shared_ptr<Blob<Dtype> >& input);
  void update_mean_gpu(const shared_ptr<Blob<Dtype> >& input);
  void update_covariance_cpu(const shared_ptr<Blob<Dtype> >& input);
  void update_covariance_gpu(const shared_ptr<Blob<Dtype> >& input);
  void norm_cpu(const shared_ptr<Blob<Dtype> >& input);
  void norm_gpu(const shared_ptr<Blob<Dtype> >& input);
  void calc_pca_cpu();
  void calc_pca_gpu();

  const int out_dim_;
  const int num_batches_;
  int iter_;
  const bool apply_whitening_;
  const bool zca_whitening_;
  bool calculated_mean_;
  bool called_setup_;
  Dtype fudge_factor_;
private:
  Blob<Dtype> mean_;
  Blob<Dtype> cov_;
  Blob<Dtype> P_;
  Blob<Dtype> sum_multiplier_;
  int batch_size_;
  int dim_;
};

/// @brief Learn PCA 
template <typename Dtype>
class LabelingDensityLearner : public UnsupervisedLearner<Dtype> {
 public:
  explicit LabelingDensityLearner(const int num_labels, const int num_batches, const int max_iterations,
    const Dtype fudge_factor, const bool soft_assignment, const Dtype lambda) : 
      num_labels_(num_labels), num_batches_(num_batches), max_iterations_(max_iterations), 
      fudge_factor_(fudge_factor), soft_assignment_(soft_assignment), lambda_(lambda), called_setup_(false), iter_(0) {}
  virtual bool step_cpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective);
  virtual bool step_gpu(const vector<shared_ptr<Blob<Dtype> > >& input, Dtype* objective);
  virtual void fill_cpu(const vector<shared_ptr<Blob<Dtype> > >& blobs);
  virtual void fill_gpu(const vector<shared_ptr<Blob<Dtype> > >& blobs);
  virtual Dtype objective_cpu(const vector<shared_ptr<Blob<Dtype> > >& input);
  virtual Dtype objective_gpu(const vector<shared_ptr<Blob<Dtype> > >& input);
 protected:
  virtual void setup(const vector<shared_ptr<Blob<Dtype> > >& input);
  void update_densities_cpu(const vector<shared_ptr<Blob<Dtype> > >& input);
  void update_densities_gpu(const vector<shared_ptr<Blob<Dtype> > >& input);

  const int num_labels_;
  const int num_batches_;
  const int max_iterations_;
  const Dtype fudge_factor_;
  const bool soft_assignment_;
  const Dtype lambda_;
  bool called_setup_;
  int iter_;
  int batch_size_;
  int dim_;

  Blob<Dtype> densities_;
  Blob<Dtype> per_label_densities_;
  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> per_label_count_;
};

}  // namespace caffe

#endif  // CAFFE_UNSUPERVISED_LEARNER_HPP_
