// hypertea.hpp is the header file that you need to include in your code. It wraps
// all the internal hypertea header files into one for simpler inclusion.

#ifndef HYPERTEA_HYPERTEA_HPP_
#define HYPERTEA_HYPERTEA_HPP_

#include "hypertea/common.hpp"

#include "hypertea/operators/activation.hpp"
#include "hypertea/operators/sampling_op.hpp"
#include "hypertea/operators/conv_op.hpp"
#include "hypertea/operators/deconv_op.hpp"
#include "hypertea/operators/libdnn_conv_op.hpp"
#include "hypertea/operators/scale_op.hpp"
#include "hypertea/operators/batch_norm_op.hpp"
#include "hypertea/operators/MIOpen_batch_norm_op.hpp"
#include "hypertea/operators/rnn_op.hpp"
#include "hypertea/operators/linear_op.hpp"

namespace hypertea {

    template <typename DeviceTensor>
    void load_weight_to_tensor(std::string path, DeviceTensor& param) {
        
        size_t weight_size = param.size();

        void* all_weights = malloc(weight_size);

        FILE *f = fopen(path.c_str(), "rb");

        size_t read_size = fread(all_weights, 1, weight_size, f);
        
        if (read_size != weight_size) { 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }

        fclose(f);

        param.copy_from_ptr((void*)all_weights);

        free(all_weights);
        
    }


    template <typename Dtype>
    
    Dtype* load_weights(std::string path, size_t weight_size) {
        

        Dtype* all_weights = (Dtype*)malloc(weight_size * sizeof(Dtype));

        FILE *f = fopen(path.c_str(), "rb");

        size_t read_size = fread(all_weights, 1, weight_size, f);
        
        if (read_size != weight_size) { 
            LOG(ERROR) << "Weight File Size Mismatch" << read_size << " and " << weight_size << std::endl;
        }

        fclose(f);

        return all_weights;        
    }


#ifdef USE_OPENCL

    void compile_opencl_kernels(
        const std::string &conv_opencl_funcs,
        const std::string &bn_opencl_funcs,
        bool is_half = false,
        const std::string save_to = " ") {
        
        OpenCLHandler::Get().build_opencl_math_code(is_half, save_to);

        if(save_to.size() > 2) {

            if(conv_opencl_funcs.size() > 4) {
                OpenCLHandler::Get().build_save_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program, save_to + "prebuilt_conv_program");
            }

            if(bn_opencl_funcs.size() > 4) {

                OpenCLHandler::Get().build_save_opencl_program(bn_opencl_funcs, OpenCLHandler::Get().bn_program, save_to + "prebuilt_bn_program");
            }

        } else {

            if(conv_opencl_funcs.size() > 4) {
                OpenCLHandler::Get().build_opencl_program(conv_opencl_funcs, OpenCLHandler::Get().conv_program);
            }

            if(bn_opencl_funcs.size() > 4) {
                OpenCLHandler::Get().build_opencl_program(bn_opencl_funcs, OpenCLHandler::Get().bn_program);
            }


        }
        

#endif //USE_OPENCL

    }


    void load_opencl_kernels(
        const std::string math_opencl_path,
        const std::string conv_opencl_path,
        const std::string bn_opencl_path) {

        OpenCLHandler::Get().load_opencl_program(math_opencl_path, OpenCLHandler::Get().math_program);


        if(conv_opencl_path.size() > 4) {
            OpenCLHandler::Get().load_opencl_program(conv_opencl_path, OpenCLHandler::Get().conv_program);
        }

        if(bn_opencl_path.size() > 4) {
            OpenCLHandler::Get().load_opencl_program(bn_opencl_path, OpenCLHandler::Get().bn_program);
        }
    }

        
}



#endif  // HYPERTEA_HYPERTEA_HPP_
