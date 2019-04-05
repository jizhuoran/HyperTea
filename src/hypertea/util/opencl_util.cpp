#include "hypertea/util/opencl_util.hpp"
#include "hypertea/common.hpp"

#include <fstream>

namespace hypertea {
 
#ifdef USE_OPENCL

void cl_mem_destory(void* ptr) {

	OPENCL_CHECK(clReleaseMemObject((cl_mem) ptr));

}



void opencl_launch_wrapper(
  const cl_program& program,
  const std::string& kernel_name,
  std::vector<std::pair<size_t, const void *> > const& arg_list,
  std::vector<size_t> const& global_size,
  std::vector<size_t> const& local_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event
) {

  cl_int ret;

  cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &ret);
  OPENCL_CHECK(ret);


  for (int i = 0; i < arg_list.size(); ++i) {
    OPENCL_CHECK(clSetKernelArg(kernel, i, arg_list[i].first, arg_list[i].second));
  }

  
  OPENCL_CHECK(
    clEnqueueNDRangeKernel(
      OpenCLHandler::Get().commandQueue, 
      kernel, 
      global_size.size(), 
      nullptr,
      global_size.data(), 
      local_size.data(), 
      num_events_in_wait_list, 
      event_wait_list, 
      event
    )
  );  
  
}





size_t reference_count(cl_mem mem_obj) {

  cl_uint refer_count;

  OPENCL_CHECK(
    clGetMemObjectInfo (
      mem_obj,
      CL_MEM_REFERENCE_COUNT,
      sizeof(cl_uint),
      &refer_count,
      nullptr
    )
  );

  return refer_count;
}




size_t cl_mem_count(cl_mem mem_obj) {

  size_t mem_count;

  OPENCL_CHECK(
    clGetMemObjectInfo (
      mem_obj,
      CL_MEM_SIZE,
      sizeof(size_t),
      &mem_count,
      nullptr
    )
  );

  return mem_count;
}




static OpenCLHandler *thread_instance_ = NULL;

OpenCLHandler& OpenCLHandler::Get() {

  if (thread_instance_ == NULL) {
      thread_instance_ = new OpenCLHandler();
  }
  return *thread_instance_;
}



OpenCLHandler::OpenCLHandler() {

	OPENCL_CHECK(clGetPlatformIDs(1, &platformId, &retNumPlatforms));
	OPENCL_CHECK(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices));

	cl_int ret;

	context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);
	OPENCL_CHECK(ret);

	commandQueue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &ret);
	OPENCL_CHECK(ret);

}

void OpenCLHandler::build_opencl_math_code(bool is_half) {
    build_opencl_program(opencl_math_code(is_half), math_program);
}


void OpenCLHandler::load_opencl_program(std::string save_binary_file, cl_program &program) {


  std::ifstream file;


  file.open(save_binary_file, std::ios::in | std::ios::binary);  
  file.seekg(0, std::ios::end);

  size_t kernel_size = file.tellg();
  file.seekg(0, std::ios::beg);

  LOG(INFO) << "The kernel we read is size of " << kernel_size << std::endl;

  char* buffer = new char[kernel_size];

  file.read(buffer, kernel_size);
  file.close();

  cl_int ret = -1;


  // int8_t *buffer;
  // FILE *f;

  // f = fopen(save_binary_file.c_str(), "rb");

  // buffer = (int8_t*)malloc(size + 1);
  // size_t readed_size = fread(buffer, 1, size, f);
  // fclose(f);

  // buffer[size] = 0;

  // const unsigned char** binary = const_cast<const unsigned char**>(reinterpret_cast<unsigned char**>(& buffer));

  // LOG(INFO) << "The size is " << size << "and the readed size is " << readed_size;


  program = clCreateProgramWithBinary(context, 1, &deviceID,
                                        &kernel_size, (const unsigned char **)&buffer, NULL, &ret);


  LOG(INFO) << "pass this line";


  OPENCL_CHECK(ret);

  ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);


  if (ret != CL_SUCCESS) {
    char *buff_erro;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
      LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__ << " with code " << errcode;
      exit(-1);
    }

    buff_erro = (char *)malloc(build_log_len);
    if (!buff_erro) {
      LOG(ERROR) << "malloc failed at line " << __LINE__;
      exit(-2);
    }

    errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
    if (errcode) {
        LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__ << " with code " << errcode;
        exit(-3);
    }
    
    LOG(ERROR) << "Build log: " << buff_erro;

    free(buff_erro);

    LOG(ERROR) << "clBuildProgram failed";

    exit(EXIT_FAILURE);
  }

  delete[] buffer;
}


void OpenCLHandler::build_opencl_program(std::string kernel_code, cl_program &program) {

  cl_int ret = -1;

  size_t kernel_size = kernel_code.size() + 1;

  const char* kernelSource = kernel_code.c_str();

  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernel_size, &ret); 
  OPENCL_CHECK(ret);

  ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);


  if (ret != CL_SUCCESS) {
    char *buff_erro;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
      LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__;
      exit(-1);
    }

    buff_erro = (char *)malloc(build_log_len);
    if (!buff_erro) {
        printf("malloc failed at line %d\n", __LINE__);
        exit(-2);
    }

    errcode = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
    if (errcode) {
        LOG(ERROR) << "clGetProgramBuildInfo failed at line " << __LINE__;
        exit(-3);
    }
    
    LOG(ERROR) << "Build log: " << buff_erro;

    free(buff_erro);

    LOG(ERROR) << "clBuildProgram failed";

    exit(EXIT_FAILURE);
  }
}


void OpenCLHandler::build_opencl_program(std::string kernel_code, cl_program &program, std::string save_binary_file) {


  build_opencl_program(kernel_code, program);


  cl_uint num_devices;
  clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, 0, &num_devices, NULL);

  assert(("We do not support multiple devices", num_devices == 1));

  size_t binary_kernel_size;
  clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_kernel_size, NULL);


  unsigned char* binary = new unsigned char[binary_kernel_size];

  clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL);

  LOG(INFO) << "The kernel we write is size of " << binary_kernel_size << std::endl;


  std::fstream file;
  file.open(save_binary_file, std::ios::out | std::ios::binary);
  file.write((const char*)binary, binary_kernel_size); // ideally, you should memcpy it to a char buffer.

  file.close();


  // FILE *f = fopen(save_binary_file.c_str(), "wb");

  // auto pos = fwrite((int8_t*)binary, sizeof(char), binary_kernel_size, f);

  // assert(("We fail to write the binary kernel", pos == binary_kernel_size));

  // LOG(INFO) << "we have written " << pos << "bytes";

  // cl_int ret = -1;

  // program = clCreateProgramWithBinary(context, 1, &deviceID,
                                        // &binary_kernel_size, (const unsigned char **)&binary, NULL, &ret);

  // OPENCL_CHECK(ret);

  delete [] binary;

}



void OpenCLHandler::DeviceQuery(){


  char device_string[1024];

  // CL_DEVICE_NAME
  clGetDeviceInfo(deviceID, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  LOG(INFO) << "  CL_DEVICE_NAME: " << device_string;

  // CL_DEVICE_VENDOR
  clGetDeviceInfo(deviceID, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
  LOG(INFO) << "  CL_DEVICE_VENDOR: " << device_string;

  // CL_DRIVER_VERSION
  clGetDeviceInfo(deviceID, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
  LOG(INFO) << "  CL_DRIVER_VERSION: " << device_string;

  // CL_DEVICE_INFO
  cl_device_type type;
  clGetDeviceInfo(deviceID, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  if( type & CL_DEVICE_TYPE_CPU )
    LOG(INFO) << "  CL_DEVICE_TYPE:"<< "CL_DEVICE_TYPE_CPU";
  if( type & CL_DEVICE_TYPE_GPU )
    LOG(INFO) << "  CL_DEVICE_TYPE:"<< "CL_DEVICE_TYPE_GPU";
  if( type & CL_DEVICE_TYPE_ACCELERATOR )
    LOG(INFO) << "  CL_DEVICE_TYPE:"<< "CL_DEVICE_TYPE_ACCELERATOR";
  if( type & CL_DEVICE_TYPE_DEFAULT )
    LOG(INFO) << "  CL_DEVICE_TYPE:"<< "CL_DEVICE_TYPE_DEFAULT";

  // CL_DEVICE_MAX_COMPUTE_UNITS
  cl_uint compute_units;
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_COMPUTE_UNITS: " << compute_units;

  // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  size_t workitem_dims;
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << workitem_dims;

  // CL_DEVICE_MAX_WORK_ITEM_SIZES
  size_t workitem_size[3];
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_WORK_ITEM_SIZES:" << workitem_size[0] << workitem_size[1] << workitem_size[2];

  // CL_DEVICE_MAX_WORK_GROUP_SIZE
  size_t workgroup_size;
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_WORK_GROUP_SIZE: " << workgroup_size;

  // CL_DEVICE_MAX_CLOCK_FREQUENCY
  cl_uint clock_frequency;
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_CLOCK_FREQUENCY:" << clock_frequency << " MHz";

  // CL_DEVICE_ADDRESS_BITS
  cl_uint addr_bits;
  clGetDeviceInfo(deviceID, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
  LOG(INFO) << "  CL_DEVICE_ADDRESS_BITS:" << addr_bits;

  // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  cl_ulong max_mem_alloc_size;
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_MEM_ALLOC_SIZE:" << (unsigned int)(max_mem_alloc_size / (1024 * 1024)) << "MByte";

  // CL_DEVICE_GLOBAL_MEM_SIZE
  cl_ulong mem_size;
  clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  LOG(INFO) << "  CL_DEVICE_GLOBAL_MEM_SIZE:" << (unsigned int)(mem_size / (1024 * 1024)) << "MByte";


  // CL_DEVICE_LOCAL_MEM_TYPE
  cl_device_local_mem_type local_mem_type;
  clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
  if (local_mem_type == 1) {
    LOG(INFO) << "  CL_DEVICE_LOCAL_MEM_TYPE: local";
  } else {
    LOG(INFO) << "  CL_DEVICE_LOCAL_MEM_TYPE: global";
  }

  // CL_DEVICE_LOCAL_MEM_SIZE
  clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  LOG(INFO) << "  CL_DEVICE_LOCAL_MEM_SIZE:" << (unsigned int)(mem_size / 1024) << "KByte\n";

  // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:" << (unsigned int)(mem_size / 1024) << "KByte\n";

  // CL_DEVICE_QUEUE_PROPERTIES
  cl_command_queue_properties queue_properties;
  clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
  if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
    LOG(INFO) << "  CL_DEVICE_QUEUE_PROPERTIES:" << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE";
  if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
    LOG(INFO) << "  CL_DEVICE_QUEUE_PROPERTIES:" << "CL_QUEUE_PROFILING_ENABLE";

  // CL_DEVICE_IMAGE_SUPPORT
  cl_bool image_support;
  clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
  LOG(INFO) << "  CL_DEVICE_IMAGE_SUPPORT:" << image_support;

  // CL_DEVICE_MAX_READ_IMAGE_ARGS
  cl_uint max_read_image_args;
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_READ_IMAGE_ARGS:" << max_read_image_args;

  // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
  cl_uint max_write_image_args;
  clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
  LOG(INFO) << "  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:" << max_write_image_args;

  // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
  size_t szMaxDims[5];
  LOG(INFO) << "\n  CL_DEVICE_IMAGE <dim>";
  clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
  LOG(INFO) << "2D_MAX_WIDTH" << szMaxDims[0];
  clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
  LOG(INFO) << "2D_MAX_HEIGHT" << szMaxDims[1];
  clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
  LOG(INFO) << "3D_MAX_WIDTH" << szMaxDims[2];
  clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
  LOG(INFO) << "3D_MAX_HEIGHT" << szMaxDims[3];
  clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
  LOG(INFO) << "3D_MAX_DEPTH" << szMaxDims[4];

  // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
  LOG(INFO) << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>";
  cl_uint vec_width [6];
  clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
  clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
  clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
  clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
  clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
  clGetDeviceInfo(deviceID, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
  LOG(INFO) << "CHAR" << vec_width[0] << "SHORT" << vec_width[1] << "INT" << vec_width[2] 
    << "FLOAT" << vec_width[3] << "DOUBLE" << vec_width[4];

   
}



std::string OpenCLHandler::opencl_math_code(bool is_half) {
	
	std::string opencl_kernel_header = is_half?

	R"(
		
		#pragma OPENCL EXTENSION cl_khr_fp16 : enable

		#define Dtype half
		#define Dtype1 half
		#define Dtype2 half2
		#define Dtype4 half4
		#define Dtype8 half8
		#define Dtype16 half16)

		#undef FLT_MIN
		#define FLT_MIN 0x1.0p-14h
		
	)"
	:
	R"(

		#define Dtype float
		#define Dtype1 float
		#define Dtype2 float2
		#define Dtype4 float4
		#define Dtype8 float8
		#define Dtype16 float16

	)";



	std::string opencl_kernel_code = R"(


	#define OPENCL_KERNEL_LOOP(i, n) for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); i < (n); i += get_num_groups(0)*get_local_size(0))



	__kernel void null_kernel_float(int alpha) {
	int a = get_local_id(0);
	}


	__kernel void ReLUForward(__global Dtype *in,
	__global Dtype *out,
	Dtype negative_slope,
  int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
	}
	}


	__kernel void PReLUForward(__global Dtype *in, __global Dtype *slope_data,
	__global Dtype *out,
	int N, int channels, int dim, int div_factor) {
	OPENCL_KERNEL_LOOP(index, N) {
	 int c = (index / dim) % channels / div_factor;
	 out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
	}
	}



	__kernel void ELUForward(__global Dtype *in,
	__global Dtype *out,
	Dtype alpha,
  int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	out[index] = in[index] > 0 ? in[index] : alpha * (exp(in[index]) - 1);
	}
	}

	__kernel void ScaleForward(__global Dtype *in,
	__global Dtype *out,
	int N, __global Dtype *scale, int scale_dim, int inner_dim) {
	OPENCL_KERNEL_LOOP(index, N) {
	const int scale_index = (index / inner_dim) % scale_dim;
	out[index] = in[index] * scale[scale_index];
	}
	}

  __kernel void ChanneledAddForward(__global Dtype *in,
  __global Dtype *out,
  int N, __global Dtype *bias, int scale_dim, int inner_dim) {
  OPENCL_KERNEL_LOOP(index, N) {
  const int scale_index = (index / inner_dim) % scale_dim;
  out[index] = in[index] + bias[scale_index];
  }
  }

  __kernel void ChanneledSubForward(__global Dtype *in,
  __global Dtype *out,
  int N, __global Dtype *bias, int scale_dim, int inner_dim) {
  OPENCL_KERNEL_LOOP(index, N) {
  const int scale_index = (index / inner_dim) % scale_dim;
  out[index] = in[index] - bias[scale_index];
  }
  }

	__kernel void ScaleBiasForward(__global Dtype *in,
	__global Dtype *out,
	int N, __global Dtype *scale, __global Dtype *bias, int scale_dim, int inner_dim) {
	OPENCL_KERNEL_LOOP(index, N) {
	const int scale_index = (index / inner_dim) % scale_dim;
	out[index] = in[index] * scale[scale_index] + bias[scale_index];
	}
	}


	__kernel void TanHForward(__global Dtype *in,
	__global Dtype *out,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	out[index] = tanh(in[index]);
	}
	}

	__kernel void mul_kernel(__global Dtype *a, __global Dtype *b,
	__global Dtype *y,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = a[index] * b[index];
	}
	}

	__kernel void div_kernel(__global Dtype *a, __global Dtype *b,
	__global Dtype *y,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = a[index] / b[index];
	}
	}


	__kernel void sqrt_kernel(__global Dtype *x,
	__global Dtype *y,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = sqrt(x[index]);
	}
	}


  __kernel void sqr_kernel(__global Dtype *x,
  __global Dtype *y,
  int N) {
  OPENCL_KERNEL_LOOP(index, N) {
   y[index] = x[index] * x[index];
  }
  }


	__kernel void log_kernel(__global Dtype *a,
	__global Dtype *y,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = log(a[index]);
	}
	}


	__kernel void exp_kernel(__global Dtype *a,
	__global Dtype *y,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = exp(a[index]);
	}
	}

	__kernel void SigmoidForward(__global Dtype *in,
	__global Dtype *out,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 out[index] = 0.5 * tanh(0.5 * in[index]) + 0.5;
	}
	}


	__kernel void abs_kernel(__global Dtype *a,
	__global Dtype *y,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = fabs(a[index]);
	}
	}


  __kernel void inv_kernel(__global Dtype *a,
  __global Dtype *y,
  int N) {
  OPENCL_KERNEL_LOOP(index, N) {
   y[index] = 1 / a[index];
  }
  }



	__kernel void powx_kernel(__global Dtype *a,
	__global Dtype *y,
	Dtype alpha, int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = pow(a[index], alpha);
	}
	}



	__kernel void sub_kernel(__global Dtype *a, __global Dtype *b,
	__global Dtype *y,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = a[index] - b[index];
	}
	}

	__kernel void add_kernel(__global Dtype *a, __global Dtype *b,
	__global Dtype *y,
	int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	 y[index] = a[index] + b[index];
	}
	}


	__kernel void kernel_channel_max(__global Dtype *data,
	__global Dtype *out,
	int num, int channels, int spatial_dim) {
	OPENCL_KERNEL_LOOP(index, num * spatial_dim) {
	 int n = index / spatial_dim;
	 int s = index % spatial_dim;
	 Dtype maxval = -FLT_MAX;
	 for (int c = 0; c < channels; ++c) {
	  maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
	 }
	 out[index] = maxval;
	}
	}

	__kernel void kernel_channel_subtract(__global Dtype *channel_max,
	__global Dtype *data,
	int count, int num, int channels, int spatial_dim) {
	OPENCL_KERNEL_LOOP(index, count) {
	 int n = index / channels / spatial_dim;
	 int s = index % spatial_dim;
	 data[index] -= channel_max[n * spatial_dim + s];
	}
	}


	__kernel void kernel_channel_sum(__global Dtype *data,
	__global Dtype *channel_sum,
	int num, int channels, int spatial_dim) {
	OPENCL_KERNEL_LOOP(index, num * spatial_dim) {
	 int n = index / spatial_dim;
	 int s = index % spatial_dim;
	 Dtype sum = 0;
	 for (int c = 0; c < channels; ++c) {
	  sum += data[(n * channels + c) * spatial_dim + s];
	 }
	 channel_sum[index] = sum;
	}
	}



	__kernel void kernel_channel_div(__global Dtype *channel_max,
	__global Dtype *data,
	int count, int num, int channels, int spatial_dim) {
	OPENCL_KERNEL_LOOP(index, count) {
	 int n = index / channels / spatial_dim;
	 int s = index % spatial_dim;
	 data[index] /= channel_max[n * spatial_dim + s];
	}
	}


	__kernel void SoftmaxLossForwardGPU(__global Dtype *prob_data, 
	__global Dtype *label,
	__global Dtype *loss, __global Dtype *counts,
	const int num, const int dim, const int spatial_dim,
	const int has_ignore_label_, const int ignore_label_,
	int nthreads) {
	OPENCL_KERNEL_LOOP(index, nthreads) {
	 const int n = index / spatial_dim;
	 const int s = index % spatial_dim;
	 const int label_value = convert_int(label[n * spatial_dim + s]);
	 if (has_ignore_label_ == 1 && label_value == ignore_label_) {

	  loss[index] = 0;
	  counts[index] = 0;
	 } else {
	  loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s], FLT_MIN));

	  counts[index] = 1;
	 }
	}
	}





	__kernel void MaxForward(__global Dtype *bottom_data_a, __global Dtype *bottom_data_b,
	__global Dtype *top_data, __global int *mask,
	int nthreads, int blob_idx) {
	OPENCL_KERNEL_LOOP(index, nthreads) {
	Dtype maxval = -FLT_MAX;
	int maxidx = -1;
	if (bottom_data_a[index] > bottom_data_b[index]) {
	if (blob_idx == 0) {
	maxval = bottom_data_a[index];
	top_data[index] = maxval;
	maxidx = blob_idx;
	mask[index] = maxidx;
	}
	} else {
	maxval = bottom_data_a[index];
	top_data[index] = maxval;
	maxidx = blob_idx + 1;
	mask[index] = maxidx;
	}
	}
	}


	__kernel void axpy_kernel(__global Dtype *X,
	__global Dtype *Y,
	Dtype alpha, int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	Y[index] = X[index] * alpha + Y[index];
	}
	}


	__kernel void scal_kernel(__global Dtype *X,
	Dtype alpha, int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	X[index] *= alpha;
	}
	}


	__kernel void add_scalar_kernel(__global Dtype *y,
	Dtype alpha, int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	y[index] += alpha;
	}
	}


  __kernel void outplace_scal_scalar_kernel(
    __global Dtype* x,
    __global Dtype* y,
    Dtype alpha, int N) {
      OPENCL_KERNEL_LOOP(index, N) {
        y[index] = x[index] * alpha;
      }
  }


  __kernel void outplace_add_scalar_kernel(
    __global Dtype* x,
    __global Dtype* y,
    Dtype alpha, int N) {
      OPENCL_KERNEL_LOOP(index, N) {
        y[index] = x[index] + alpha;
      }
  }


	__kernel void BiasForward(__global Dtype *in, __global Dtype *bias,
	__global Dtype *out,
	int bias_dim, int inner_dim, int N) {
	OPENCL_KERNEL_LOOP(index, N) {
	const int bias_index = (index / inner_dim) % bias_dim;
	out[index] = in[index] + bias[bias_index];
	}
	}


	int compute_uncropped_index(int index, const int ndims,
	__global int *src_strides, __global int *dest_strides,
	__global int *offsets) {
	int dest_index = index;
	int src_index = 0;
	for (int i = 0; i < ndims; ++i) {
	int coord = dest_index / dest_strides[i];
	dest_index -= coord * dest_strides[i];
	src_index += src_strides[i] * (coord + offsets[i]);
	}
	return src_index;
	}


	__kernel void crop_kernel_forward(__global Dtype *src,
	__global Dtype *dest,
	__global int *src_strides,
	__global int *dest_strides,
	__global int *offsets,
	int ndims, int nthreads) {
	OPENCL_KERNEL_LOOP(index, nthreads) {
	 int src_index = compute_uncropped_index(index, ndims, src_strides, dest_strides, offsets);
	 dest[index] = src[src_index];
	}
	}


	__kernel void Concat(__global Dtype *in_data,
	__global Dtype *out_data,
	const int concat_size,
	const int top_concat_axis, const int bottom_concat_axis,
	const int offset_concat_axis, int nthreads) {
	OPENCL_KERNEL_LOOP(index, nthreads) {
	 const int total_concat_size = concat_size * bottom_concat_axis;
	 const int concat_num = index / total_concat_size;
	 const int concat_index = index % total_concat_size;
	 const int top_index = concat_index + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
	 out_data[top_index] = in_data[index];
	}
	}





	__kernel void Slice(__global Dtype *in_data,
	__global Dtype *out_data,
	const int slice_size,
	const int bottom_slice_axis, const int top_slice_axis,
	const int offset_slice_axis, int nthreads) {
	OPENCL_KERNEL_LOOP(index, nthreads) {
	 const int total_slice_size = slice_size * top_slice_axis;
	 const int slice_num = index / total_slice_size;
	 const int slice_index = index % total_slice_size;
	 const int bottom_index = slice_index + 
	(slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
	 out_data[index] = in_data[bottom_index];
	}
	}


	__kernel void Tile(__global Dtype *bottom_data,
	__global Dtype *top_data,
	const int tile_size,
	const int num_tiles, const int bottom_tile_axis,
	int nthreads) {
	OPENCL_KERNEL_LOOP(index, nthreads) {
	 const int d = index % tile_size;
	 const int b = (index / tile_size / num_tiles) % bottom_tile_axis;
	 const int n = index / tile_size / num_tiles / bottom_tile_axis;
	 const int bottom_index = (n * bottom_tile_axis + b) * tile_size + d;
	 top_data[index] = bottom_data[bottom_index];
	}
	}


	__kernel void EmbedForward(__global Dtype *bottom_data, __global Dtype *weight,
	__global Dtype *top_data,
	const int M, const int N, const int K,
	int nthreads) {
	OPENCL_KERNEL_LOOP(top_index, nthreads) {
	 const int n = top_index / N;
	 const int d = top_index % N;
	 const int index = convert_int(bottom_data[n]);
	 const int weight_index = index * N + d;
	 top_data[top_index] = weight[weight_index];
	}
	}

	//TODO Whether convert_in is expensive?

	__kernel void BRForward(__global Dtype *in, __global Dtype *permut,
	__global Dtype *out,
	const int inner_dim, int nthreads) {
	OPENCL_KERNEL_LOOP(index, nthreads) {
	 int n = index / (inner_dim);
	 int in_n = convert_int(permut[n]);
	 out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
	}
	}




	__kernel void LRNFillScale(__global Dtype* in, __global Dtype* scale,
	    const int num, const int channels, const int height,
	    const int width, const int size, const Dtype alpha_over_size,
	    const Dtype k, const int nthreads) {
	  OPENCL_KERNEL_LOOP(index, nthreads) {
	    // find out the local offset
	    const int w = index % width;
	    const int h = (index / width) % height;
	    const int n = index / width / height;
	    const int offset = (n * channels * height + h) * width + w;
	    const int step = height * width;
	    __global const Dtype* const in_off = in + offset;
	    __global Dtype* const scale_off = scale + offset;
	    int head = 0;
	    const int pre_pad = (size - 1) / 2;
	    const int post_pad = size - pre_pad - 1;
	    Dtype accum_scale = 0;
	    // fill the scale at [n, :, h, w]
	    // accumulate values
	    while (head < post_pad && head < channels) {
	      accum_scale += in_off[head * step] * in_off[head * step];
	      ++head;
	    }
	    // both add and subtract
	    while (head < channels) {
	      accum_scale += in_off[head * step] * in_off[head * step];
	      if (head - size >= 0) {
	        accum_scale -= in_off[(head - size) * step]
	                       * in_off[(head - size) * step];
	      }
	      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
	      ++head;
	    }
	    // subtract only
	    while (head < channels + post_pad) {
	      if (head - size >= 0) {
	        accum_scale -= in_off[(head - size) * step]
	                       * in_off[(head - size) * step];
	      }
	      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
	      ++head;
	    }
	  }
	}


	__kernel void LRNComputeOutput(__global Dtype* const in, __global Dtype* const scale,
	__global Dtype* const out,
	const Dtype negative_beta, const int nthreads) {
	  OPENCL_KERNEL_LOOP(index, nthreads) {
	    out[index] = in[index] * pow(scale[index], negative_beta);
	  }
	}



	__kernel void MaxPoolForward(int nthreads, 
	    __global Dtype* bottom_data, int num, int channels, 
	    int height, int width, int pooled_height, 
	    int pooled_width, int kernel_h, int kernel_w, 
	    int stride_h, int stride_w, __global Dtype* top_data, int pad_h, int pad_w, 
	    __global int* mask, __global Dtype* top_mask) { 
	  OPENCL_KERNEL_LOOP(index, nthreads) { 
	    const int pw = index % pooled_width; 
	    const int ph = (index / pooled_width) % pooled_height; 
	    const int c = (index / pooled_width / pooled_height) % channels; 
	    const int n = index / pooled_width / pooled_height / channels; 
	    int hstart = ph * stride_h - pad_h; 
	    int wstart = pw * stride_w - pad_w; 
	    const int hend = min(hstart + kernel_h, height); 
	    const int wend = min(wstart + kernel_w, width); 
	    hstart = max(hstart, 0); 
	    wstart = max(wstart, 0); 
	    Dtype maxval = -FLT_MAX; 
	    int maxidx = -1; 
	    __global Dtype* bottom_slice = 
	        bottom_data + (n * channels + c) * height * width; 
	    for (int h = hstart; h < hend; ++h) { 
	      for (int w = wstart; w < wend; ++w) { 
	        if (bottom_slice[h * width + w] > maxval) { 
	          maxidx = h * width + w; 
	          maxval = bottom_slice[maxidx]; 
	        } 
	      } 
	    } 
	    top_data[index] = maxval; 
	    if (mask) { 
	      mask[index] = maxidx; 
	    } else { 
	      top_mask[index] = maxidx; 
	    } 
	  } 
	} 

	__kernel void AvePoolForward(const int nthreads, 
	    __global Dtype* bottom_data, const int num, const int channels, 
	    const int height, const int width, const int pooled_height, 
	    const int pooled_width, const int kernel_h, const int kernel_w, 
	    const int stride_h, const int stride_w, __global Dtype* top_data, const int pad_h, const int pad_w) {
	  OPENCL_KERNEL_LOOP(index, nthreads) { 
	    const int pw = index % pooled_width; 
	    const int ph = (index / pooled_width) % pooled_height; 
	    const int c = (index / pooled_width / pooled_height) % channels; 
	    const int n = index / pooled_width / pooled_height / channels; 
	    int hstart = ph * stride_h - pad_h; 
	    int wstart = pw * stride_w - pad_w; 
	    int hend = min(hstart + kernel_h, height + pad_h); 
	    int wend = min(wstart + kernel_w, width + pad_w); 
	    const int pool_size = (hend - hstart) * (wend - wstart); 
	    hstart = max(hstart, 0); 
	    wstart = max(wstart, 0); 
	    hend = min(hend, height); 
	    wend = min(wend, width); 
	    Dtype aveval = 0; 
	    __global Dtype* const bottom_slice = 
	        bottom_data + (n * channels + c) * height * width; 
	    for (int h = hstart; h < hend; ++h) { 
	      for (int w = wstart; w < wend; ++w) { 
	        aveval += bottom_slice[h * width + w]; 
	      } 
	    } 
	    top_data[index] = aveval / pool_size; 
	  } 
	} 


	__kernel void StoPoolForwardTest(const int nthreads, 
	    __global Dtype* bottom_data, 
	    const int num, const int channels, const int height, 
	    const int width, const int pooled_height, const int pooled_width, 
	    const int kernel_h, const int kernel_w, const int stride_h, 
	    const int stride_w, __global Dtype* top_data) { 
	    OPENCL_KERNEL_LOOP(index, nthreads) { 
	    const int pw = index % pooled_width; 
	    const int ph = (index / pooled_width) % pooled_height; 
	    const int c = (index / pooled_width / pooled_height) % channels; 
	    const int n = index / pooled_width / pooled_height / channels; 
	    const int hstart = ph * stride_h; 
	    const int hend = min(hstart + kernel_h, height); 
	    const int wstart = pw * stride_w; 
	    const int wend = min(wstart + kernel_w, width); 
	    // We set cumsum to be 0 to avoid divide-by-zero problems 
	    Dtype cumsum = 0.; 
	    Dtype cumvalues = 0.; 
	    __global Dtype* const bottom_slice = 
	        bottom_data + (n * channels + c) * height * width; 
	    // First pass: get sum 
	    for (int h = hstart; h < hend; ++h) { 
	      for (int w = wstart; w < wend; ++w) { 
	        cumsum += bottom_slice[h * width + w]; 
	        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w]; 
	      } 
	    } 
	    top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.; 
	  } 
	} 





  static inline void ReduceKernel(__local Dtype* lcl_mem,
                                  unsigned int sum_stride,
                                  unsigned int unit_id,
                                  unsigned int unit_len)
  {
      Dtype sum              = (Dtype)0.;
      unsigned int lcl_offset = unit_id * unit_len;

      for(unsigned int i = 0; i < unit_len; i += sum_stride) {
          sum += lcl_mem[lcl_offset + i];
      }
      lcl_mem[lcl_offset] = sum;
  }
          
  static inline void
  regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
  {
      data[localID] = *value;
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID < (128 >> 2))
          ReduceKernel(data, 1, localID, 4);
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID < (128 >> 4))
          ReduceKernel(data, 4, localID, 16);
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID == 0)
          ReduceKernel(data, 16, localID, 128);
      barrier(CLK_LOCAL_MEM_FENCE);
      *value = data[0] * scale;
  }


  #define BUFFER_SIZE 128
 
  __attribute__((reqd_work_group_size(BUFFER_SIZE, 1, 1))) 
  __kernel void average_channeled(
    const __global Dtype* __restrict in,
    __global Dtype* __restrict mean_out,
    __global Dtype* __restrict var_out,
    const int spatial_dim,
    const int cspatial_dim,
    const int nspatial_dim,
    Dtype alpha) {


    Dtype mean        = (Dtype)0.;
    Dtype variance    = (Dtype)0.;

    uint index = 0;
    uint lid   = get_local_id(0);
    uint chwid = get_global_id(1) * spatial_dim;
    uint nidx  = 0;
    uint hwidx = 0;
    
    Dtype read;
    for(unsigned int k = lid; k < nspatial_dim; k += BUFFER_SIZE) {
      
      nidx  = k / spatial_dim;
      hwidx = k - (nidx * spatial_dim);
      index = nidx * cspatial_dim + chwid + hwidx;

      read = in[index];
      mean += read;
      variance = mad((Dtype)read, (Dtype)read, variance);
    }
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    local Dtype lcl_data[BUFFER_SIZE];

    lcl_data[lid] = mean;
    regLDSreduce(&mean, lcl_data, lid, alpha);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (lid == 0) {
      mean_out[get_group_id(1)] = mean;
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    regLDSreduce(&variance, lcl_data, lid, alpha);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
      variance = mad(-mean, mean, variance);
      var_out[get_group_id(1)] = sqrt(variance + 1e-05);
    }
    barrier(CLK_LOCAL_MEM_FENCE);


} // end spatial norm









	// #ifndef WGS1
	  #define WGS1 64
	// #endif
	// #ifndef WGS2
	  #define WGS2 64
	// #endif

	__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
	void Xasum(const int n,
	           const __global Dtype* restrict xgm, const int x_inc,
	           __global Dtype* output, Dtype alpha) {
	      
	  __local Dtype lm[WGS1];
	  const int lid = get_local_id(0);
	  const int wgid = get_group_id(0);
	  const int num_groups = get_num_groups(0);

	  // Performs loading and the first steps of the reduction
	  Dtype acc = 0;

	  int id = wgid*WGS1 + lid;

	  while (id*x_inc < n) {
	    Dtype x = xgm[id*x_inc + get_group_id(1) * n];
	    acc += x * alpha;
	    id += WGS1*num_groups;
	  }
	  lm[lid] = acc * alpha;
	  barrier(CLK_LOCAL_MEM_FENCE);

	  // Performs reduction in local memory
	  for (int s=WGS1/2; s>0; s=s>>1) {
	    if (lid < s) {
	      lm[lid] += lm[lid + s];
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	  }

	  // Stores the per-workgroup result
	  if (lid == 0) {
	    output[wgid + get_group_id(1) * num_groups] = lm[0];
	  }
	}


	__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
	void XasumEpilogue(const __global Dtype* restrict input,
	                   __global Dtype* asum, Dtype beta) {
	      
	  __local Dtype lm[WGS2];
	  const int lid = get_local_id(0);

	  // Performs the first step of the reduction while loading the data
	  lm[lid] = (input[get_group_id(1) * WGS2 * 2 + lid] + input[get_group_id(1) * WGS2 * 2 + lid + WGS2]) * beta;
	  barrier(CLK_LOCAL_MEM_FENCE);

	  // Performs reduction in local memory
	  for (int s=WGS2/2; s>0; s=s>>1) {
	    if (lid < s) {
	      lm[lid] += lm[lid + s];
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	  }

	  // Computes the absolute value and stores the final result
	  if (lid == 0) {
	    asum[get_group_id(1)] = lm[0];
	  }
	}



  __kernel void Col2Im(__global Dtype* col,
                     const int col_h,
                     const int col_w,
                     const int wei_h,
                     const int wei_w,
                     const int pad_h,
                     const int pad_w,
                     const int stride_h,
                     const int stride_w,
                     const int dilation_h,
                     const int dilation_w,
                     const int height,
                     const int width,
                     __global Dtype* im,
                     const int im_offset) {
    __global Dtype* im_off = im + im_offset;
    int gid               = (int)get_global_id(0);

    int im_ch  = gid / (width * height);
    int im_pix = gid % (width * height);
    int im_h   = (im_pix / width) + pad_h;
    int im_w   = (im_pix % width) + pad_w;

    int start_h = (im_h < dilation_h * (wei_h - 1) + 1)
                      ? 0
                      : (im_h - (dilation_h * (wei_h - 1) + 1)) / stride_h + 1;
    int end_h   = min(col_h, im_h / stride_h + 1);
    int start_w = (im_w < dilation_w * (wei_w - 1) + 1)
                      ? 0
                      : (im_w - (dilation_w * (wei_w - 1) + 1)) / stride_w + 1;
    int end_w = min(col_w, im_w / stride_w + 1);

    int ch_offset = im_ch * col_w * col_h * wei_w * wei_h;
    col += ch_offset;

    Dtype tmp = (Dtype)0;
    for(int cy = start_h; cy < end_h; cy++)
    {
        for(int cx = start_w; cx < end_w; cx++)
        {
            if((im_h - cy * stride_h) % dilation_h == 0 && (im_w - cx * stride_w) % dilation_w == 0)
            {
                int col_off_y = cy + (((im_h - cy * stride_h) / dilation_h) * wei_w * col_h);
                int col_off_x = cx + (((im_w - cx * stride_w) / dilation_w) * col_w * col_h);

                tmp += (Dtype)(col[col_off_y * col_w + col_off_x]);
            }
        }
    }
    im_off[gid] = tmp;
  }






  )";



  	return opencl_kernel_header + opencl_kernel_code;
}

#endif

}  // namespace hypertea
