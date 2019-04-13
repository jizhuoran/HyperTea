#include <vector>

#include "hypertea/util/im2col.hpp"

namespace hypertea {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}



template <typename Dtype>
void im2col(const TensorCPU<Dtype>& data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorCPU<Dtype>& data_col) {

  auto data_im_data = data_im.immutable_data();
  auto data_col_data = data_col.mutable_data();

  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im_data += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col_data++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col_data++) = data_im_data[input_row * width + input_col];
              } else {
                *(data_col_data++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }

}

template void im2col(const TensorCPU<float>& data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorCPU<float>& data_col);



template <typename Dtype>
void col2im(const TensorCPU<Dtype>& data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    TensorCPU<Dtype>& data_im) {


  data_im.set(0);
  auto data_im_data = data_im.mutable_data();
  auto data_col_data = data_col.immutable_data();


  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im_data += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col_data += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im_data[input_row * width + input_col] += *data_col_data;
              }
              data_col_data++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template void col2im(const TensorCPU<float>& data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    TensorCPU<float>& data_im
);



#ifdef USE_OPENCL


template <typename Dtype>
void im2col(const TensorGPU<Dtype>& data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorGPU<Dtype>& data_col) {

  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;


  auto data_im_data = data_im.immutable_data();
  auto data_col_data = data_col.mutable_data();

  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "im2col_gpu_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_int), (void *)&num_kernels),
      std::make_pair(sizeof(cl_mem), (void *)&data_im_data),
      std::make_pair(sizeof(cl_int), (void *)&height),
      std::make_pair(sizeof(cl_int), (void *)&width),
      std::make_pair(sizeof(cl_int), (void *)&kernel_h),
      std::make_pair(sizeof(cl_int), (void *)&kernel_w),
      std::make_pair(sizeof(cl_int), (void *)&pad_h),
      std::make_pair(sizeof(cl_int), (void *)&pad_w),
      std::make_pair(sizeof(cl_int), (void *)&stride_h),
      std::make_pair(sizeof(cl_int), (void *)&stride_w),
      std::make_pair(sizeof(cl_int), (void *)&dilation_h),
      std::make_pair(sizeof(cl_int), (void *)&dilation_w),
      std::make_pair(sizeof(cl_int), (void *)&height_col),
      std::make_pair(sizeof(cl_int), (void *)&width_col),
      std::make_pair(sizeof(cl_mem), (void *)&data_col_data),
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(num_kernels)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );

}

template void im2col(const TensorGPU<float>& data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorGPU<float>& data_col);

template void im2col(const TensorGPU<half>& data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    TensorGPU<half>& data_col);


template <typename Dtype>
void col2im(const TensorGPU<Dtype>& data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    TensorGPU<Dtype>& data_im) {


  auto data_im_data = data_im.mutable_data();
  auto data_col_data = data_col.mutable_data();

  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;


  opencl_launch_wrapper(
    OpenCLHandler::Get().math_program,
    "col2im_gpu_kernel",
    std::vector<std::pair<size_t, const void *> > {
      std::make_pair(sizeof(cl_int), (void *)&num_kernels),
      std::make_pair(sizeof(cl_mem), (void *)&data_col_data),
      std::make_pair(sizeof(cl_int), (void *)&height),
      std::make_pair(sizeof(cl_int), (void *)&width),
      std::make_pair(sizeof(cl_int), (void *)&channels),
      std::make_pair(sizeof(cl_int), (void *)&kernel_h),
      std::make_pair(sizeof(cl_int), (void *)&kernel_w),
      std::make_pair(sizeof(cl_int), (void *)&pad_h),
      std::make_pair(sizeof(cl_int), (void *)&pad_w),
      std::make_pair(sizeof(cl_int), (void *)&stride_h),
      std::make_pair(sizeof(cl_int), (void *)&stride_w),
      std::make_pair(sizeof(cl_int), (void *)&dilation_h),
      std::make_pair(sizeof(cl_int), (void *)&dilation_w),
      std::make_pair(sizeof(cl_int), (void *)&height_col),
      std::make_pair(sizeof(cl_int), (void *)&width_col),
      std::make_pair(sizeof(cl_mem), (void *)&data_im_data),
    },
    std::vector<size_t> {HYPERTEA_GET_BLOCKS(num_kernels)},
    std::vector<size_t> {HYPERTEA_OPENCL_NUM_THREADS}
  );


}



template void col2im(const TensorGPU<float>& data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    TensorGPU<float>& data_im
);

template void col2im(const TensorGPU<half>& data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    TensorGPU<half>& data_im
);

#endif //USE_OPENCL

}  // namespace hypertea
