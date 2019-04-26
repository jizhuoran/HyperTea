#include "hypertea/util/opencl_util.hpp"
#include "hypertea/common.hpp"

#include <fstream>

namespace hypertea {
 
#ifdef USE_OPENCL

void cl_mem_destory(void* ptr) { OPENCL_CHECK(clReleaseMemObject((cl_mem) ptr)); }



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


  program = clCreateProgramWithBinary(context, 1, &deviceID,
                                        &kernel_size, (const unsigned char **)&buffer, NULL, &ret);


  LOG(INFO) << "pass this line";


  OPENCL_CHECK(ret);

  ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

  OPENCL_BUILD_CHECK(ret);


  delete[] buffer;
}


void OpenCLHandler::build_opencl_program(const std::string &kernel_code, cl_program &program) {

  cl_int ret = -1;

  size_t kernel_size = kernel_code.size() + 1;

  const char* kernelSource = kernel_code.c_str();

  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernel_size, &ret); 
  OPENCL_CHECK(ret);

  ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

  OPENCL_BUILD_CHECK(ret);

}


void OpenCLHandler::build_save_opencl_program(std::string kernel_code, cl_program &program, std::string save_binary_file) {


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


std::string unary_opencl_math_kernel(
  std::string name,
  std::string operation) {

  std::stringstream ss;

  ss << std::endl << std::endl
     << "__kernel void " << name << "_kernel(" << std::endl
     << "  const __global Dtype *x," << std::endl
     << "  __global Dtype *y," << std::endl
     << "  int N) {" << std::endl
     << "  OPENCL_KERNEL_LOOP(index, N) {"  << std::endl
     << "    y[index] = " << operation << std::endl
     << "  }" << std::endl
     << "}" << std::endl << std::endl << std::endl;

  return ss.str();

}

std::string unary_scale_opencl_math_kernel(
  std::string name,
  std::string operation) {

  std::stringstream ss;

  ss << std::endl << std::endl
     << "__kernel void " << name << "_kernel(" << std::endl
     << "  const __global Dtype *x," << std::endl
     << "  __global Dtype *y," << std::endl
     << "  const Dtype a," << std::endl
     << "  int N) {" << std::endl
     << "  OPENCL_KERNEL_LOOP(index, N) {"  << std::endl
     << "    y[index] = " << operation << std::endl
     << "  }" << std::endl
     << "}" << std::endl << std::endl << std::endl;

  return ss.str();

}


std::string binary_opencl_math_kernel(
  std::string name,
  std::string operation) {

  std::stringstream ss;

  ss << std::endl << std::endl
     << "__kernel void " << name << "_kernel(" << std::endl
     << "  const __global Dtype *a," << std::endl
     << "  const __global Dtype *b," << std::endl
     << "  __global Dtype *y," << std::endl
     << "  int N) {" << std::endl
     << "  OPENCL_KERNEL_LOOP(index, N) {"  << std::endl
     << "    y[index] = " << operation << std::endl
     << "  }" << std::endl
     << "}" << std::endl << std::endl << std::endl;

  return ss.str();

}


std::string channel_opencl_math_kernel(
  std::string name,
  std::string operation,
  bool has_bias) {

  std::stringstream ss;

  ss << std::endl << std::endl;
  ss << "__kernel void " << name << "_kernel(" << std::endl;
  ss << "  const __global Dtype *x," << std::endl;
  ss << "  __global Dtype *y," << std::endl;
  ss << "  int N," << std::endl;
  ss << "  const __global Dtype *weight," << std::endl;
  if (has_bias) {ss << "  const __global Dtype *bias," << std::endl;}
  ss << " int scale_dim," << std::endl;
  ss << "  int inner_dim) {" << std::endl;
  ss << "  OPENCL_KERNEL_LOOP(index, N) {"  << std::endl;
  ss << "    const int scale_index = (index / inner_dim) % scale_dim;" << std::endl;
  ss << "    " << operation << std::endl;
  ss << "  }" << std::endl;
  ss << "}" << std::endl << std::endl << std::endl;

  return ss.str();

}



std::string reduce_opencl_math_kernel(
  std::string name,
  std::string reduce_op,
  std::string read_op) {

  std::stringstream ss;

  ss << std::endl << std::endl;
  ss << " static inline void " << name << "_reduce_kernel( " << std::endl;
  ss << "   __local Dtype* lcl_mem, " << std::endl;
  ss << "   unsigned int stride, " << std::endl;
  ss << "   unsigned int unit_id,  " << std::endl;
  ss << "   unsigned int unit_len) { " << std::endl;
  ss << "     Dtype value = 0.0; " << std::endl;
  ss << "     unsigned int lcl_offset = unit_id * unit_len; " << std::endl;
  ss << "     for(unsigned int i = 0; i < unit_len; i += stride) { " << std::endl;
  ss << "         " << reduce_op << std::endl;
  ss << "     } " << std::endl;
  ss << "     lcl_mem[lcl_offset] = value; " << std::endl;
  ss << " } " << std::endl << std::endl << std::endl;


  ss << " static inline void " << name << "_LDS_reduce( " << std::endl;
  ss << "   Dtype* value, " << std::endl;
  ss << "   __local Dtype* data,  " << std::endl;
  ss << "   unsigned int localID) { " << std::endl;
  ss << "     data[localID] = *value; " << std::endl;
  ss << "     barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  ss << "     if(localID < (128 >> 2)) " << std::endl;
  ss << "          " << name << "_reduce_kernel(data, 1, localID, 4); " << std::endl;
  ss << "     barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  ss << "     if(localID < (128 >> 4)) " << std::endl;
  ss << "          " << name << "_reduce_kernel(data, 4, localID, 16); " << std::endl;
  ss << "     barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  ss << "     if(localID == 0) " << std::endl;
  ss << "          " << name << "_reduce_kernel(data, 16, localID, 128); " << std::endl;
  ss << "     barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  ss << "     *value = data[0]; " << std::endl;
  ss << " } " << std::endl << std::endl << std::endl;


  ss << " __attribute__((reqd_work_group_size(128, 1, 1)))  " << std::endl;
  ss << " __kernel void channeled_" << name << "_kernel( " << std::endl;
  ss << "   const __global Dtype* __restrict in, " << std::endl;
  ss << "   __global Dtype* __restrict out, " << std::endl;
  ss << "   const int spatial_dim) { " << std::endl;
  ss << "   Dtype red_value = 0.0; " << std::endl;
  ss << "   uint lid = get_local_id(0); " << std::endl;
  ss << "   uint batch_index = get_global_id(1); " << std::endl;
  ss << "   for(unsigned int k = lid; k < spatial_dim; k += 128) {       " << std::endl;
  ss << "     " << read_op << std::endl;
  ss << "   } " << std::endl;
  ss << "   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); " << std::endl;
  ss << "   local Dtype lcl_data[128]; " << std::endl;
  ss << "   " << name << "_LDS_reduce(&red_value, lcl_data, lid); " << std::endl;
  ss << "   barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  ss << "   if (lid == 0) { " << std::endl;
  ss << "     out[batch_index] = red_value; " << std::endl;
  ss << "   } " << std::endl;
  ss << " } " << std::endl;

  return ss.str();

}
















                    








            



             




            







std::string OpenCLHandler::opencl_math_code(bool is_half) {
	


	std::string opencl_kernel_header = is_half?

	R"(
		
		#pragma OPENCL EXTENSION cl_khr_fp16 : enable

		#define Dtype half
		#define Dtype2 half2
		#define Dtype4 half4
		#define Dtype8 half8

		#undef FLT_MIN
		#define FLT_MIN 0x1.0p-14h

    #undef FLT_MAX
    #define FLT_MAX 0x1.ffcp15h
		
	)"
	:
	R"(

		#define Dtype float
		#define Dtype2 float2
		#define Dtype4 float4

	)";

  

	std::string opencl_kernel_code =


	"#define OPENCL_KERNEL_LOOP(i, n) for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); i < (n); i += get_num_groups(0)*get_local_size(0))"

   + unary_opencl_math_kernel("sqrt", "sqrt(x[index]);")
   + unary_opencl_math_kernel("sqr", "x[index] * x[index];")
   + unary_opencl_math_kernel("log", "log(x[index]);")
   + unary_opencl_math_kernel("exp", "exp(x[index]);")
   + unary_opencl_math_kernel("abs", "fabs(x[index]);")
   + unary_opencl_math_kernel("sigmoid", "0.5 * tanh(0.5 * x[index]) + 0.5;")
   + unary_opencl_math_kernel("inv", "1/x[index];")
   + unary_opencl_math_kernel("tanh", "tanh(x[index]);")



   + unary_scale_opencl_math_kernel("relu", "x[index] > 0 ? x[index] : x[index] * a;")
   + unary_scale_opencl_math_kernel("elu", "x[index] > 0 ? x[index] : a * (exp(x[index]) - 1);")
   + unary_scale_opencl_math_kernel("powx", "pow(x[index], a);")
   + unary_scale_opencl_math_kernel("scal_scalar", "x[index] * a;")
   + unary_scale_opencl_math_kernel("add_scalar", "x[index] + a;")

   + binary_opencl_math_kernel("add", "a[index] + b[index];")
   + binary_opencl_math_kernel("sub", "a[index] - b[index];")
   + binary_opencl_math_kernel("mul", "a[index] * b[index];")
   + binary_opencl_math_kernel("div", "a[index] / b[index];")

   + channel_opencl_math_kernel("channel_add", "y[index] = x[index] + weight[scale_index];", false)
   + channel_opencl_math_kernel("channel_sub", "y[index] = x[index] - weight[scale_index];", false)
   + channel_opencl_math_kernel("channel_scal", "y[index] = x[index] * weight[scale_index];", false)
   + channel_opencl_math_kernel("channel_scaladd", "y[index] = x[index] * weight[scale_index] + bias[scale_index];", true)
   + channel_opencl_math_kernel("prelu", "y[index] = x[index] > 0? x[index] : x[index] * weight[scale_index];", true)

   + reduce_opencl_math_kernel("sum", "value += lcl_mem[lcl_offset + i];", "red_value += in[batch_index * spatial_dim + k];")

  + R"(
	__kernel void null_kernel_float(int alpha) {
    int a = get_local_id(0);
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
    Dtype alpha,
    Dtype eps) {


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
      var_out[get_group_id(1)] = sqrt(variance + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);


} // end spatial norm


  


  static inline void MaxReduceKernel(
    __local Dtype* lcl_mem,
    __local int* lcl_index,
    unsigned int sum_stride,
    unsigned int unit_id,
    unsigned int unit_len) {

      Dtype max_value = -FLT_MAX;
      int max_index = -1;
      unsigned int lcl_offset = unit_id * unit_len;

      for(unsigned int i = 0; i < unit_len; i += sum_stride) {

        if (lcl_mem[lcl_offset + i] > max_value) {
          max_value = lcl_mem[lcl_offset + i];
          max_index = lcl_index[lcl_offset + i];
        }
      }
      lcl_mem[lcl_offset] = max_value;
      lcl_index[lcl_offset] = max_index;
  }
          
  static inline void
  MaxregLDSreduce(
    Dtype* value, int* pos,
    __local Dtype* data, 
    __local int* index, 
    unsigned int localID)
  {
      data[localID] = *value;
      index[localID] = *pos;

      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID < (128 >> 2))
          MaxReduceKernel(data, index, 1, localID, 4);
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID < (128 >> 4))
          MaxReduceKernel(data, index, 4, localID, 16);
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID == 0)
          MaxReduceKernel(data, index, 16, localID, 128);
      barrier(CLK_LOCAL_MEM_FENCE);
      *value = data[0];
      *pos = index[0];
  }


  #define BUFFER_SIZE 128
 
  __attribute__((reqd_work_group_size(BUFFER_SIZE, 1, 1))) 
  __kernel void argmax_kernel(
    const __global Dtype* __restrict in,
    __global Dtype* __restrict out_value,
    __global int* __restrict out_index,
    const int spatial_dim) {


    Dtype max_value = -FLT_MAX;
    int max_index = -1;

    uint index = 0;
    uint lid   = get_local_id(0);
    uint batch_index = get_global_id(1);
    
    Dtype read;
    for(unsigned int k = lid; k < spatial_dim; k += BUFFER_SIZE) {
      
      index = batch_index * spatial_dim + k;

      read = in[index];

      if (read > max_value) {
        max_value = read;
        max_index = k;
      }
    }
     
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    local Dtype lcl_data[BUFFER_SIZE];
    local int lcl_index[BUFFER_SIZE];

    MaxregLDSreduce(&max_value, &max_index, lcl_data, lcl_index, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (lid == 0) {
      out_value[batch_index] = max_value;
      out_index[batch_index] = max_index;
    }

  } // end argmax_kernel



static inline void SumReduceKernel(
    __local Dtype* lcl_mem,
    unsigned int sum_stride,
    unsigned int unit_id,
    unsigned int unit_len) {

      Dtype sum = 0.0;
      unsigned int lcl_offset = unit_id * unit_len;

      for(unsigned int i = 0; i < unit_len; i += sum_stride) {
          sum += lcl_mem[lcl_offset + i];
      }

      lcl_mem[lcl_offset] = sum;
  }
          
  static inline void
  SumregLDSreduce(
    Dtype* value,
    __local Dtype* data, 
    unsigned int localID)
  {
      data[localID] = *value;

      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID < (128 >> 2))
          SumReduceKernel(data, 1, localID, 4);
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID < (128 >> 4))
          SumReduceKernel(data, 4, localID, 16);
      barrier(CLK_LOCAL_MEM_FENCE);
      if(localID == 0)
          SumReduceKernel(data, 16, localID, 128);
      barrier(CLK_LOCAL_MEM_FENCE);
      *value = data[0];
  }


 
  __attribute__((reqd_work_group_size(BUFFER_SIZE, 1, 1))) 
  __kernel void reduce_sum_kernel(
    const __global Dtype* __restrict in,
    __global Dtype* __restrict out,
    const int spatial_dim) {


    Dtype sum = 0.0;

    uint index = 0;
    uint lid   = get_local_id(0);
    uint batch_index = get_global_id(1);
    
    Dtype read;
    for(unsigned int k = lid; k < spatial_dim; k += BUFFER_SIZE) {      
      sum += in[batch_index * spatial_dim + k];
    }
     
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    local Dtype lcl_data[BUFFER_SIZE];

    SumregLDSreduce(&sum, lcl_data, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (lid == 0) {
      out[batch_index] = sum;
    }

  } // end argmax_kernel


  __kernel void im2col_gpu_kernel(
    const int n, 
    __global Dtype* data_im,
      const int height, const int width, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      const int height_col, const int width_col,
      __global Dtype* data_col) {
    OPENCL_KERNEL_LOOP(index, n) {
      const int h_index = index / width_col;
      const int h_col = h_index % height_col;
      const int w_col = index % width_col;
      const int c_im = h_index / height_col;
      const int c_col = c_im * kernel_h * kernel_w;
      const int h_offset = h_col * stride_h - pad_h;
      const int w_offset = w_col * stride_w - pad_w;
      __global Dtype* data_col_ptr = data_col;
      data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
      __global Dtype* data_im_ptr = data_im;
      data_im_ptr += (c_im * height + h_offset) * width + w_offset;
      for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
          int h_im = h_offset + i * dilation_h;
          int w_im = w_offset + j * dilation_w;
          *data_col_ptr =
              (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
              data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
          data_col_ptr += height_col * width_col;
        }
      }
    }
  }



  __kernel void col2im_gpu_kernel(
    const int n, 
    __global Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    __global Dtype* data_im) {
  OPENCL_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}





  __kernel void up_sampling_nearest_neighbor_2d_kernel(
        const __global Dtype* in,
        __global Dtype* out,
        const int num,
        const int spatial_dim,
        const int width,
        const int scale) {

    int index = get_global_id(0);


    if (index < spatial_dim) {

      int h = index / width; 
      int w = index % width; 

      for (int n = 0; n < num; n++) {
        
        Dtype val = in[n * spatial_dim + index];

        for (int i = 0; i < scale; ++i)
        {
          for (int j = 0; j < scale; ++j)
          {
            out[n * spatial_dim * scale * scale + (h * scale + i) * width * scale + (w * scale + j)] = val;
          }
        }
      }

    }
  }


  __kernel void transpose_hw_kernel(
        const __global Dtype* in,
        __global Dtype* out,
        const int num,
        const int spatial_dim,
        const int old_last_dim,
        const int new_last_dim) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < new_last_dim && y < old_last_dim) {

      for (int i = 0; i < num; ++i) {
        const __global Dtype* per_in = in + i * spatial_dim;
        __global Dtype* per_out = out + i * spatial_dim;

        per_out[x * old_last_dim + y] = per_in[y * new_last_dim + x];

      }
    
    }

    

  }

  )";



  	return opencl_kernel_header + opencl_kernel_code;
}

#endif

}  // namespace hypertea
