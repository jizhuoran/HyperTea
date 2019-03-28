#ifndef HYPERTEA_UTIL_OPENCL_UTIL_H_
#define HYPERTEA_UTIL_OPENCL_UTIL_H_

#ifdef USE_OPENCL

#include <iostream>
#include <vector>
#include <string.h>
#include <sstream>
#include <CL/cl.h>



#include "hypertea/util/device_alternate.hpp"

namespace hypertea {


void cl_mem_destory(void* ptr);

void opencl_launch_wrapper(
  const cl_program& program,
  const std::string& kernel_name,
  std::vector<std::pair<size_t, const void *> > const& arg_list,
  std::vector<size_t> const& global_size,
  std::vector<size_t> const& local_size,
  cl_uint num_events_in_wait_list = 0,
  const cl_event *event_wait_list = nullptr,
  cl_event *event = nullptr
);

size_t reference_count(cl_mem mem_obj);


class OpenCLHandler
{
public:

	~OpenCLHandler() {}
	

	static OpenCLHandler& Get();

	void DeviceQuery();
	void build_opencl_program(std::string kernel_code, cl_program &program);
	void build_opencl_program(std::string kernel_code, cl_program &program, std::string save_binary_file);

	void load_opencl_program(std::string save_binary_file, cl_program &program);

  	std::string opencl_math_code(bool is_half);


  	void build_opencl_math_code(bool is_half);



  	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	  
	cl_context context;
	cl_command_queue commandQueue;

	cl_program math_program;
	cl_program conv_program;
	cl_program bn_program;


private:

	OpenCLHandler();
	OpenCLHandler(const OpenCLHandler&);
  	OpenCLHandler& operator=(const OpenCLHandler&);

};

}  // namespace hypertea

#endif //USE_OPENCL


#endif   // HYPERTEA_UTIL_OPENCL_UTIL_H_
