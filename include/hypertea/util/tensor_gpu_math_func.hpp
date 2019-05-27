#ifndef HYPERTEA_UTIL_TENSOR_GPU_MATH_FUNC_H_
#define HYPERTEA_UTIL_TENSOR_GPU_MATH_FUNC_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <clblast.h>
#include <cblas.h>

using int16 = short;

namespace hypertea {

template<typename Dtype> class Tensor;
template<typename Dtype> class TensorCPU;
template<typename Dtype> class TensorGPU;


#ifdef USE_OPENCL  // GPU

template<typename T> T to_dtype(const float x);
template<> inline float to_dtype<float>(const float x) {return x;}
template<> inline half to_dtype<half> (const float x) {return float2half_impl(x);}
template<> inline int16 to_dtype<int16> (const float x) {return static_cast<int16>(x);}


template <typename Dtype>
TensorGPU<Dtype>& unary_math_gpu_inplace(TensorGPU<Dtype> &x, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype>& unary_scalar_math_gpu_inplace(TensorGPU<Dtype> &x, const float scalar, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype>& binary_math_gpu_inplace(const TensorGPU<Dtype> &x, TensorGPU<Dtype> &y, const std::string& op_name);

template <typename Dtype>
TensorGPU<Dtype>& binary_scalar_math_gpu_inplace(const TensorGPU<Dtype> &x, TensorGPU<Dtype> &y, const float scalar, const std::string& op_name);


template <typename Dtype>
TensorGPU<Dtype> inplace_gemm(
	const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB,
	const int M, const int N, const int K,
    const float alpha, 
    const TensorGPU<Dtype>& A, 
    const TensorGPU<Dtype>& B, 
    const float beta,
    TensorGPU<Dtype> C) {

	size_t lda = (TransA == CblasNoTrans) ? K : M;
  	size_t ldb = (TransB == CblasNoTrans) ? N : K;
  	size_t ldc = N;

	Dtype alpha_(to_dtype<Dtype>(alpha));
  	Dtype beta_(to_dtype<Dtype>(beta));

	auto A_data = A.immutable_data();
  	auto B_data = B.immutable_data();
  	auto C_data = C.mutable_data();

	auto blastTransA =
      (TransA == CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;
  	auto blastTransB =
      (TransB == CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

	// CLBLAST_CPP_CHECK(clblast::Gemm<Dtype>(
	//     clblast::Layout::kRowMajor,
	//     blastTransA, blastTransB,
	//     M, N, K,
	//     alpha_,
	//     (cl_mem) A_data, 0, lda,
	//     (cl_mem) B_data, 0, ldb,
	//     beta_,
	//     (cl_mem) C_data, 0, ldc,
	//     &OpenCLHandler::Get().commandQueue, NULL)
 //  );

	return C;
}



template <typename Dtype>
TensorGPU<Dtype> inplace_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorGPU<Dtype>& A, 
    const TensorGPU<Dtype>& x, 
    const float beta,
    TensorGPU<Dtype> y) {

	Dtype alpha_(to_dtype<Dtype>(alpha));
  	Dtype beta_(to_dtype<Dtype>(beta));

	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto y_data = y.immutable_data();


	auto blastTransA =
	(TransA != CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

	// CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(
	// 	clblast::Layout::kColMajor,
	// 	blastTransA, 
	// 	N, M,
	// 	alpha_,
	// 	A_data, 0, N,
	// 	x_data, 0, 1,
	// 	beta_,
	// 	y_data, 0, 1,
	// 	&OpenCLHandler::Get().commandQueue, NULL)
	// );

	return y;
}


template <typename Dtype>
inline TensorGPU<Dtype> inplace_set(TensorGPU<Dtype> &x, const Dtype alpha) {
	size_t x_size = x.count() * sizeof(Dtype);
	auto x_data = x.mutable_data();
	OPENCL_CHECK(clEnqueueFillBuffer(OpenCLHandler::Get().commandQueue, x_data, &alpha, sizeof(Dtype), 0, x_size, 0, NULL, NULL));
	return x;
}


template <typename Dtype>
inline TensorGPU<Dtype> inplace_add(TensorGPU<Dtype> x, const TensorGPU<Dtype>& y) {
	binary_math_gpu_inplace(y, x, "add_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_sub(TensorGPU<Dtype> x, const TensorGPU<Dtype>& y) {
	binary_math_gpu_inplace(y, x, "sub_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_mul(TensorGPU<Dtype> x, const TensorGPU<Dtype>& y) {
	binary_math_gpu_inplace(y, x, "mul_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_div(TensorGPU<Dtype> x, const TensorGPU<Dtype>& y) {
	binary_math_gpu_inplace(y, x, "div_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_add_scalar(TensorGPU<Dtype> y, const float a) {
	unary_scalar_math_gpu_inplace(y, a, "add_scalar_kernel");
	return y;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_sub_scalar(TensorGPU<Dtype> y, const float a) {
	unary_scalar_math_gpu_inplace(y, static_cast<float>(-a), "add_scalar_kernel");
	return y;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_mul_scalar(TensorGPU<Dtype> y, const float a) {
	unary_scalar_math_gpu_inplace(y, a, "scal_scalar_kernel");
	return y;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_div_scalar(TensorGPU<Dtype> y, const float a) {
	unary_scalar_math_gpu_inplace(y, static_cast<float>(1/a), "scal_scalar_kernel");
	return y;
}


template <typename Dtype>
inline TensorGPU<Dtype> inplace_sigmoid(TensorGPU<Dtype> x) {
  unary_math_gpu_inplace(x, "sigmoid_kernel");
  return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_tanh(TensorGPU<Dtype> x) {
  unary_math_gpu_inplace(x, "tanh_kernel");
  return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_abs(TensorGPU<Dtype> x) {
	unary_math_gpu_inplace(x, "abs_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_exp(TensorGPU<Dtype> x) {
	unary_math_gpu_inplace(x, "exp_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_log(TensorGPU<Dtype> x) {
	unary_math_gpu_inplace(x, "log_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_sqr(TensorGPU<Dtype> x) {
	unary_math_gpu_inplace(x, "sqr_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_sqrt(TensorGPU<Dtype> x) {
	unary_math_gpu_inplace(x, "sqrt_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_inv(TensorGPU<Dtype> x) {
	unary_math_gpu_inplace(x, "inv_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_powx(TensorGPU<Dtype> x, const float a) {
	unary_scalar_math_gpu_inplace(x, a, "powx_kernel");
	return x;
}

template <typename Dtype>
inline TensorGPU<Dtype> inplace_elu(TensorGPU<Dtype> x, const float a = 1.) {
	unary_scalar_math_gpu_inplace(x, a, "elu_kernel");
	return x;
}


template <typename Dtype>
inline TensorGPU<Dtype> inplace_relu(TensorGPU<Dtype> x, const float a = .0) {
	unary_scalar_math_gpu_inplace(x, a, "relu_kernel");
	return x;
}


template <typename Dtype>
TensorGPU<Dtype> inplace_channeled_scal(
	TensorGPU<Dtype> x, 
	const TensorGPU<Dtype>& weight,
	int channels,
	int inner_dim
);


template <typename Dtype>
TensorGPU<Dtype> inplace_channeled_add(
	TensorGPU<Dtype> x, 
	const TensorGPU<Dtype>& bias,
	int channels,
	int inner_dim
);

template <typename Dtype>
TensorGPU<Dtype> inplace_channeled_sub(
	TensorGPU<Dtype> x, 
	const TensorGPU<Dtype>& bias,
	int channels,
	int inner_dim
);

template <typename Dtype>
TensorGPU<Dtype> inplace_channeled_scaladd(
	TensorGPU<Dtype> x, 
	const TensorGPU<Dtype>& weight,
	const TensorGPU<Dtype>& bias,
	int channels,
	int inner_dim
);




template <typename Dtype>
TensorGPU<Dtype> outplace_gemm(
	const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB,
	const int M, const int N, const int K,
    const float alpha, 
    const TensorGPU<Dtype>& A, 
    const TensorGPU<Dtype>& B, 
    const float beta) {


	TensorGPU<Dtype> nC(M*N, 0);


	size_t lda = (TransA == CblasNoTrans) ? K : M;
  	size_t ldb = (TransB == CblasNoTrans) ? N : K;
  	size_t ldc = N;

	Dtype alpha_(to_dtype<Dtype>(alpha));
  	Dtype beta_(to_dtype<Dtype>(beta));

	auto A_data = A.immutable_data();
  	auto B_data = B.immutable_data();
  	auto C_data = nC.immutable_data();

	auto blastTransA =
      (TransA == CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;
  	auto blastTransB =
      (TransB == CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

	CLBLAST_CPP_CHECK(clblast::Gemm<Dtype>(
    clblast::Layout::kRowMajor,
    blastTransA, blastTransB,
    M, N, K,
    alpha_,
    (cl_mem) A_data, 0, lda,
    (cl_mem) B_data, 0, ldb,
    beta_,
    (cl_mem) C_data, 0, ldc,
    &OpenCLHandler::Get().commandQueue, NULL)
  );

	return nC;
}



template <typename Dtype>
TensorGPU<Dtype> outplace_gemv(
	const CBLAS_TRANSPOSE TransA, 
	const int M, const int N,
    const float alpha, 
    const TensorGPU<Dtype>& A, 
    const TensorGPU<Dtype>& x, 
    const float beta,
    const TensorGPU<Dtype>& y) {

	TensorGPU<Dtype> ny(y.count());
	ny.copy_data(y);

	Dtype alpha_(to_dtype<Dtype>(alpha));
  	Dtype beta_(to_dtype<Dtype>(beta));

  	auto A_data = A.immutable_data();
  	auto x_data = x.immutable_data();
  	auto ny_data = ny.immutable_data();


	auto blastTransA =
	(TransA != CblasNoTrans) ? clblast::Transpose::kNo : clblast::Transpose::kYes;

	CLBLAST_CPP_CHECK(clblast::Gemv<Dtype>(
		clblast::Layout::kColMajor,
		blastTransA, 
		N, M,
		alpha_,
		A_data, 0, N,
		x_data, 0, 1,
		beta_,
		ny_data, 0, 1,
		&OpenCLHandler::Get().commandQueue, NULL)
	);

	return ny;
}



template <typename Dtype>
TensorGPU<Dtype> channeled_sum(
	TensorGPU<Dtype> x, 
	int spatial_dim
);


template <typename Dtype>
std::vector<int> batched_argmax(
	TensorGPU<Dtype> x, 
	int spatial_dim
);


template <typename Dtype>
TensorGPU<Dtype> concate(std::vector<TensorGPU<Dtype>* > xs) {

	int total_count = 0;

	for (auto const&x: xs) { total_count += x->count(); }

	TensorGPU<Dtype> y(total_count);
	auto y_data = y.mutable_data();

	int pos = 0;
	for (auto const&x: xs) {
		OPENCL_CHECK(clEnqueueCopyBuffer(OpenCLHandler::Get().commandQueue, 
    		x->immutable_data(), 
    		y_data, 
    		0, sizeof(Dtype) * pos, sizeof(Dtype) * x->count(), 0, NULL, NULL)
		);

		pos += x->count();
	}

	return y;


}


template <typename Dtype>
TensorGPU<Dtype> hconcate(std::vector<TensorGPU<Dtype>* > xs, int top_dim) {

	int total_count = 0;

	for (auto const&x: xs) { total_count += x->count(); }

	TensorGPU<Dtype> y(total_count);
	auto y_data = y.mutable_data();

	int pos = 0;

	for (int i = 0; i < top_dim; ++i) {
		for (auto const&x: xs) {

			OPENCL_CHECK(clEnqueueCopyBuffer(OpenCLHandler::Get().commandQueue, 
	    		x->immutable_data(), 
	    		y_data, 
	    		sizeof(Dtype) * x->count() / top_dim * i, sizeof(Dtype) * pos, 
	    		sizeof(Dtype) * x->count() / top_dim, 0, NULL, NULL)
			);
			pos += (x->count() / top_dim);
		}
	}

	return y;
}


 
template<typename Dtype> 
TensorGPU<Dtype> operator+ (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {return inplace_add(lhs.duplicate() ,rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator+ (const TensorGPU<Dtype>& lhs, const float rhs) {return inplace_add_scalar(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator+ (const float lhs, const TensorGPU<Dtype>& rhs) {return inplace_add_scalar(rhs.duplicate(), lhs); }

template<typename Dtype>
TensorGPU<Dtype> operator- (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {return inplace_sub(lhs.duplicate() ,rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator- (const TensorGPU<Dtype>& lhs, const float rhs) {return inplace_sub_scalar(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator- (const float lhs, const TensorGPU<Dtype>& rhs) {return inplace_add_scalar(rhs.duplicate(), -lhs); }


template<typename Dtype>
TensorGPU<Dtype> operator* (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {return inplace_mul(lhs.duplicate() ,rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator* (const TensorGPU<Dtype>& lhs, const float rhs) {return inplace_mul_scalar(lhs.duplicate(), rhs); }
template<typename Dtype>
TensorGPU<Dtype> operator* (const float lhs, const TensorGPU<Dtype>& rhs) {return inplace_mul_scalar(rhs.duplicate(), lhs); }


template<typename Dtype>
TensorGPU<Dtype> operator/ (const TensorGPU<Dtype>& lhs, const TensorGPU<Dtype>& rhs) {return inplace_div(lhs.duplicate() ,rhs); }
template<typename Dtype> 
TensorGPU<Dtype> operator/ (const TensorGPU<Dtype>& lhs, const float rhs) {return inplace_div_scalar(lhs.duplicate(), rhs); }
// template<typename Dtype>
// TensorGPU<Dtype> operator/ (const float lhs, const TensorGPU<Dtype>& rhs) {return outplace_div_scalar(rhs.duplicate(), lhs); }

#endif  // USE_OPENCL

}  // namespace hypertea

#endif  // HYPERTEA_UTIL_TENSOR_GPU_MATH_FUNC_H_
