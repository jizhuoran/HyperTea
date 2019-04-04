#ifndef HYPERTEA_BATCHNORM_OP_HPP_
#define HYPERTEA_BATCHNORM_OP_HPP_

#include <vector>

#include "hypertea/operator.hpp"

namespace hypertea {

/**
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch.
 *
 * This layer computes Batch Normalization as described in [1]. For each channel
 * in the data (i.e. axis 1), it subtracts the mean and divides by the variance,
 * where both statistics are computed across both spatial dimensions and across
 * the different examples in the batch.
 *
 * By default, during training time, the network is computing global
 * mean/variance statistics via a running average, which is then used at test
 * time to allow deterministic outputs for each input. You can manually toggle
 * whether the network is accumulating or using the statistics via the
 * use_global_stats option. For reference, these statistics are kept in the
 * layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
 *
 * Note that the original paper also included a per-channel learned bias and
 * scaling factor. To implement this in Hypertea, define a `ScaleLayer` configured
 * with `bias_term: true` after each `BatchNormOp` to handle both the bias
 * and scaling factor.
 *
 * [1] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network
 *     Training by Reducing Internal Covariate Shift." arXiv preprint
 *     arXiv:1502.03167 (2015).
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BatchNormOp_CPU : public CPUFunctor<Dtype> {
 public:
  explicit BatchNormOp_CPU(int top_count, int num,
                       int channels,
                       float eps, float scale_factor,
                       bool use_global_stats,
                       Dtype* mean, Dtype* variance,
                       Dtype* weight, Dtype* bias,
                       bool inplace = false)
      : CPUFunctor<Dtype>(), top_count_(top_count), num_(num),
        channels_(channels),
        eps_(eps), scale_factor_(scale_factor),
        use_global_stats_(use_global_stats),
        mean_(mean), variance_(variance),
        weight_(weight), bias_(bias),
        inplace_(inplace) {

          temp_ = new Dtype[top_count];

          spatial_dim_ = top_count/(num*channels);

          spatial_sum_multiplier_ = new Dtype[spatial_dim_];
          hypertea_set(spatial_dim_, float(1), spatial_sum_multiplier_);

          num_by_chans_ = new Dtype[num*channels];

          batch_sum_multiplier_ = new Dtype[num];
          hypertea_set(num, float(1), batch_sum_multiplier_);

          if(!use_global_stats) {
            mean_ = new Dtype[channels];
            variance_ = new Dtype[channels];
          }

          if (bias != NULL) {
            bias_multiplier_ = new Dtype[top_count_ / (channels_ * num_)];
            hypertea_set(top_count_ / (channels_ * num_), Dtype(1), bias_multiplier_);
          }

        }

  ~BatchNormOp_CPU() {
    delete [] temp_;
    delete [] spatial_sum_multiplier_;
    delete [] num_by_chans_;
    delete [] batch_sum_multiplier_;

    if(!use_global_stats_) {
      delete [] mean_;
      delete [] variance_;
    }

    if(bias_ != NULL) {
      delete [] bias_multiplier_;
    }

  }


  virtual inline const char* type() const { return "BatchNorm"; }

 // protected:
  // virtual void Forward(const std::vector<Dtype*> bottom_datas,
  //     const std::vector<Dtype*> top_datas);

  virtual TensorCPU<Dtype> Forward(TensorCPU<Dtype> &input_tensor);
  TensorCPU<Dtype> TForward(TensorCPU<Dtype> &input_tensor);



  Dtype *mean_, *variance_;
  Dtype *weight_, *bias_;
  Dtype *temp_, *bias_multiplier_;
  bool use_global_stats_;
  
  float eps_;

  float scale_factor_;


  int channels_, spatial_dim_;
  int top_count_, num_;

  bool inplace_;



  // extra temporarary variables is used to carry out sums/broadcasting
  // using BLAS
  Dtype* batch_sum_multiplier_;
  Dtype* num_by_chans_;
  Dtype* spatial_sum_multiplier_;
};



#ifdef USE_OPENCL


template <typename Dtype>
class BatchNormOp_GPU : public GPUFunctor<Dtype> {
 public:

  explicit BatchNormOp_GPU(
    int top_count, int num,
    int channels,
    float eps, float scale_factor,
    bool use_global_stats,
    const TensorGPU<Dtype>& mean, const TensorGPU<Dtype>& variance,
    const TensorGPU<Dtype>& weight, const TensorGPU<Dtype>& bias,
    float sum_shift_num = 1.0, float top_shift_num = 1.0,
    bool inplace = false)
      : GPUFunctor<Dtype>(), top_count_(top_count), num_(num),
        channels_(channels),
        eps_(eps), scale_factor_(scale_factor),
        use_global_stats_(use_global_stats),
        // mean_(mean), variance_(variance),
        tmean_(mean), tvariance_(variance),
        tweight_(weight), tbias_(bias),
        sum_shift_num_(sum_shift_num), top_shift_num_(top_shift_num),
        inplace_(inplace), ttemp_(TensorGPU<Dtype>(top_count)) {

          has_bias_ = (bias.count() != 0);
          affine = (weight.count() != 0);

          temp_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(Dtype) * top_count, NULL, NULL);

          spatial_dim_ = top_count/(num*channels);

          spatial_sum_multiplier_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(Dtype) * spatial_dim_, NULL, NULL);
          hypertea_gpu_set<Dtype>(spatial_dim_, float(1.), spatial_sum_multiplier_);

          num_by_chans_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(Dtype) * num * channels, NULL, NULL);

          batch_sum_multiplier_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(Dtype) * num, NULL, NULL);
          hypertea_gpu_set<Dtype>(num, float(1.), batch_sum_multiplier_);

        }

 


  virtual inline const char* type() const { return "BatchNorm"; }

  // virtual void Forward(const std::vector<cl_mem> bottom_datas,
  //     const std::vector<cl_mem> top_datas);
  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> input_tensor);

  // cl_mem mean_, variance_;
  // cl_mem weight_ = NULL;
  // cl_mem bias_ = NULL;
  cl_mem temp_;



  TensorGPU<Dtype> tmean_;
  TensorGPU<Dtype> tvariance_;
  TensorGPU<Dtype> tweight_;
  TensorGPU<Dtype> tbias_;
  TensorGPU<Dtype> ttemp_;


  bool use_global_stats_;
  
  float eps_;

  float scale_factor_;


  int channels_, spatial_dim_;
  int top_count_, num_;

  // extra temporarary variables is used to carry out sums/broadcasting
  // using BLAS
  cl_mem batch_sum_multiplier_;
  cl_mem num_by_chans_;
  cl_mem spatial_sum_multiplier_;

  float sum_shift_num_, top_shift_num_;

  
  bool inplace_;

  bool has_bias_;
  bool affine;
};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_BATCHNORM_OP_HPP_
