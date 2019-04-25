#include "hypertea/common.hpp"
#include "hypertea/operators/sampling_op.hpp"

namespace hypertea {



template <typename Dtype>
TensorCPU<Dtype> upsampling_2d(
	TensorCPU<Dtype>& x,
	int scale,
	int height,
	int width,
	int spatial_dim) {


	int nums = x.count() / spatial_dim;

	TensorCPU<Dtype> y(x.count() * scale * scale);

	auto x_data = x.mutable_data();
	auto y_data = y.mutable_data();

	int index = 0;

	for (int n = 0; n < nums; ++n) {

		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {

				auto val = x_data[n * spatial_dim + i * width + j];

				for (int is = 0; is < scale; ++is) {
					for (int js = 0; js < scale; ++js) {
						y_data[n * spatial_dim * scale * scale + (i * scale + is) * width * scale + j * scale + js]= val;
					}				
				}		
			}
		}
	}

	return y;

}

template TensorCPU<float> upsampling_2d(
  TensorCPU<float>& x,
  int scale,
  int height,
  int width,
  int spatial_dim
);


template <typename Dtype>
TensorGPU<Dtype> upsampling_2d(
  TensorGPU<Dtype>& x,
  int scale,
  int height,
  int width,
  int spatial_dim
) {

  size_t num = static_cast<size_t>(x.count() / spatial_dim);

  TensorGPU<Dtype> y(x.count() * scale * scale);

  auto x_data = x.mutable_data();
  auto y_data = y.mutable_data();



  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "up_sampling_nearest_neighbor_2d_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_mem), (void *)&x_data),
      std::make_pair(sizeof(cl_mem), (void *)&y_data),
      std::make_pair(sizeof(cl_int), (void *)&num),
      std::make_pair(sizeof(cl_int), (void *)&spatial_dim),
      std::make_pair(sizeof(cl_int), (void *)&width),
      std::make_pair(sizeof(cl_int), (void *)&scale),
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(spatial_dim)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );


  return y;
}

template TensorGPU<float> upsampling_2d(
  TensorGPU<float>& x,
  int scale,
  int height,
  int width,
  int spatial_dim
);

template TensorGPU<half> upsampling_2d(
  TensorGPU<half>& x,
  int scale,
  int height,
  int width,
  int spatial_dim
);


template<typename DeviceTensor>
DeviceTensor UpSampling2D<DeviceTensor>::operator()(DeviceTensor input) {
	return upsampling_2d(input, scale_, height_, width_, height_* width_);
}
DEFINE_FORWARD_FUNC(UpSampling2D);

 

}  // namespace hypertea
