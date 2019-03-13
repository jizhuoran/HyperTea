#ifndef HYPERTEA_MIO_BATCHNORM_OP_HPP_
#define HYPERTEA_MIO_BATCHNORM_OP_HPP_

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

#ifdef USE_OPENCL


template <typename Dtype>
class MIOpenBatchNormOp_GPU : public GPUFunctor<Dtype> {
 public:

  explicit MIOpenBatchNormOp_GPU(
    std::string kernel_name,
    cl_mem mean, cl_mem variance,
    cl_mem weight, cl_mem bias,
    std::vector<size_t> local,
    std::vector<size_t> global,
    int channels,
    bool inplace = false)
      : GPUFunctor<Dtype>(), kernel_name_(kernel_name),
        mean_(mean), variance_(variance),
        weight_(weight), bias_(bias),
        local_size_(local), global_size_(global),
        inplace_(inplace) {

          if(weight_ == NULL) {
            weight_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(Dtype) * channels, NULL, NULL);
            hypertea_gpu_set<float>(channels, float(1.), weight_);
          }

          if(bias_ == NULL) {
            bias_ = clCreateBuffer(OpenCLHandler::Get().context, CL_MEM_READ_WRITE, sizeof(Dtype) * channels, NULL, NULL);
            hypertea_gpu_set<float>(channels, float(0.), bias_);
          }


        }


  // std::string MIOpen_BN_code(std::string params);
  // void build_program();
  std::vector<size_t> local_size_;
  std::vector<size_t> global_size_;
  std::string kernel_name_;
  bool single_ = true;
  float inhw_ = 1.0;


  virtual inline const char* type() const { return "BatchNorm"; }

  // virtual void Forward(const std::vector<cl_mem> bottom_datas,
  //     const std::vector<cl_mem> top_datas);
  virtual TensorGPU<Dtype> Forward(TensorGPU<Dtype> input_tensor);

  cl_mem mean_ = NULL;
  cl_mem variance_ = NULL;
  cl_mem weight_ = NULL;
  cl_mem bias_ = NULL;
  
  bool inplace_;
};

#endif //USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_MIO_BATCHNORM_OP_HPP_
