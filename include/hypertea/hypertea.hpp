// hypertea.hpp is the header file that you need to include in your code. It wraps
// all the internal hypertea header files into one for simpler inclusion.

#ifndef HYPERTEA_HYPERTEA_HPP_
#define HYPERTEA_HYPERTEA_HPP_

#include "hypertea/common.hpp"

#include "hypertea/operators/activation.hpp"
#include "hypertea/operators/conv_op.hpp"
#include "hypertea/operators/deconv_op.hpp"
#include "hypertea/operators/scale_op.hpp"
#include "hypertea/operators/batch_norm_op.hpp"
#include "hypertea/operators/MIOpen_batch_norm_op.hpp"
#include "hypertea/operators/rnn_op.hpp"


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

}



#endif  // HYPERTEA_HYPERTEA_HPP_
