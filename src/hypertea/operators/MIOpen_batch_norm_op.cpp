#include <algorithm>
#include <vector>
#include <math.h>

#include "hypertea/operators/MIOpen_batch_norm_op.hpp"
#include "hypertea/util/math_functions.hpp"

namespace hypertea {

template <typename Dtype>
TensorGPU<Dtype> MIOpenBatchNormOp_GPU<Dtype>::Forward(TensorGPU<Dtype> input_tensor){

    const cl_mem input_data = input_tensor.immutable_data();
    TensorGPU<Dtype> output_tensor = TensorGPU<Dtype>(input_tensor.count());
    cl_mem output_data = output_tensor.mutable_data();



    if(single_)
    {
        

        cl_int ret;


        cl_kernel kernel = clCreateKernel(bn_program, "MIOpenBatchNormFwdTrainSpatial", &ret);
        OPENCL_CHECK(ret);


          // Set arguments for kernel
        OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_data));  
        OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_data));  
        OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&weight_));  
        OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bias_));   
        OPENCL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_float), (void *)&inhw_));   
        OPENCL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_float), (void *)&eps_));   
  

        size_t* global_size = vgd_.data();
        size_t* local_size  = vld_.data();

        OPENCL_CHECK(clEnqueueNDRangeKernel(OpenCLHandler::Get().commandQueue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL));  


        
    }
    else
    {


        std::cout << "We need this part" << std::endl;
        exit(1);
        
        // vld.push_back(xlocalsize);
        // vld.push_back(ylocalsize);
        // vld.push_back(zlocalsize);

        // vgd.push_back(xgridsize);
        // vgd.push_back(ygridsize);
        // vgd.push_back(zgridsize);

        // std::string kernel_name  = "MIOpenBatchNormFwdTrainSpatial";
        // std::string program_name = "MIOpenBatchNormFwdTrainSpatial.cl";
        // std::string parms =
        //     " -DMIO_BN_N=" + std::to_string(n) + "\n" + 
        //     " -DMIO_BN_C=" + std::to_string(c) + "\n" +
        //     " -DMIO_BN_HW=" + std::to_string(in_cstride) + "\n" + 
        //     " -DMIO_BN_NHW=" + std::to_string(in_nhw) + "\n" + 
        //     " -DMIO_BN_CHW=" + std::to_string(in_nstride) + "\n" +
        //     " -DMIO_BN_NCHW=" + std::to_string(in_nchw) + "\n" + 
        //     " -DMIO_BN_NGRPS=" + std::to_string(int(std::ceil(float(ygridsize) / ylocalsize))) + "\n" +
        //     " -DMIO_BN_LDS_SIZE=" + std::to_string(ldsnogcn) + "\n" + 
        //     " -DMIO_BN_LDSGCN_SIZE=" + std::to_string(ldsgcn) + "\n" + 
        //     " -DMIO_BN_VARIANT=" + std::to_string(variant) + "\n" +
        //     " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) + "\n" + 
        //     " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) + "\n" + 
        //     " -DMIO_BN_GRP2=" + std::to_string(zlocalsize) + "\n";




        // bnFwdTrainSelectMulti(handle,
        //                       bnScaleBiasMeanVarDesc.GetType(),
        //                       program_name,
        //                       algo_name,
        //                       kernel_name,
        //                       network_config,
        //                       parms,
        //                       vld,
        //                       vgd,
        //                       x,
        //                       y,
        //                       bnScale,
        //                       bnBias,
        //                       resultsave,
        //                       resultrunning,
        //                       expAvgFactor,
        //                       resultRunningMean,
        //                       resultRunningVariance,
        //                       epsilon,
        //                       resultSaveMean,
        //                       resultSaveInvVariance,
        //                       inhw_);
        
    }


    return output_tensor;
}



template <typename Dtype>
void MIOpenBatchNormOp_GPU<Dtype>::build_program() {
    unsigned int in_cstride = spatial_dim_;
    unsigned int in_nstride = channels_ * in_cstride;
    unsigned int in_nhw     = num_ * in_cstride;
    unsigned int in_nchw    = num_ * in_nstride;

    size_t xlocalsize = 1024;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;

    size_t xgridsize = channels_ * xlocalsize;
    size_t ygridsize = 1;
    size_t zgridsize = 1;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;


    bool resultsave = false;
    bool resultrunning = false;

    inhw_ /= in_nhw;

    unsigned int variant  = 1;
    unsigned int ldsgcn   = xlocalsize / 64;
    unsigned int ldsnogcn = xlocalsize;
    if(in_nhw < 33554432 && in_cstride > 1024)
    {
        //
    }
    else if(in_nhw < 33554432 && in_cstride > 512)
    {
        variant    = 3;
        xlocalsize = 64 * ((in_cstride + 63) / 64);
        ldsgcn     = xlocalsize / 64;
        ldsnogcn   = xlocalsize;
        xgridsize  = channels_ * xlocalsize;
    }
    else if(in_cstride <= 512)
    {
        variant = 0;
    }
    else
    {
        variant      = 2;
        xlocalsize   = 1;
        ylocalsize   = 1024;
        auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
        xgridsize    = channels_;
        ygridsize    = segment * ylocalsize;
        single_       = false;
        ldsgcn       = ylocalsize / 64;
        ldsnogcn     = ylocalsize;
    }


    vld_.push_back(xlocalsize);
    vld_.push_back(ylocalsize);
    vld_.push_back(zlocalsize);

    vgd_.push_back(xgridsize);
    vgd_.push_back(ygridsize);
    vgd_.push_back(zgridsize);

    std::string parms =
        "#define MIO_BN_N " + std::to_string(num_) + "\n" + 
        "#define MIO_BN_C " + std::to_string(channels_) + "\n" +
        "#define MIO_BN_HW " + std::to_string(in_cstride) + "\n" + 
        "#define MIO_BN_NHW " + std::to_string(in_nhw) + "\n" + 
        "#define MIO_BN_CHW " + std::to_string(in_nstride) + "\n" +
        "#define MIO_BN_NCHW " + std::to_string(in_nchw) + "\n" + 
        "#define MIO_BN_LDS_SIZE " + std::to_string(ldsnogcn) + "\n" + 
        "#define MIO_BN_LDSGCN_SIZE " + std::to_string(ldsgcn) + "\n" +
        "#define MIO_BN_VARIANT " + std::to_string(variant) + "\n" + 
        "#define MIO_BN_GRP0 " + std::to_string(xlocalsize) + "\n" + 
        "#define MIO_BN_GRP1 " + std::to_string(ylocalsize) + "\n" +
        "#define MIO_BN_GRP2 " + std::to_string(zlocalsize) + "\n";

    
    OpenCLHandler::Get().build_opencl_program(MIOpen_BN_code(parms), bn_program);

}



template <typename Dtype>
std::string MIOpenBatchNormOp_GPU<Dtype>::MIOpen_BN_code(std::string params) {
    return params + R"(

/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#define Dtype float
#define Dtype4 float4


#ifndef MIO_BN_LDSGCN_SIZE
#define MIO_BN_LDSGCN_SIZE 16
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 256
#endif

#ifndef MIO_BN_C
#define MIO_BN_C 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#ifndef MIO_BN_GRP0
#define MIO_BN_GRP0 1
#endif

#ifndef MIO_BN_GRP1
#define MIO_BN_GRP1 1
#endif

#ifndef MIO_BN_GRP2
#define MIO_BN_GRP2 1
#endif

#ifndef MIO_BN_NGRPS
#define MIO_BN_NGRPS 1
#endif

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 255
#endif

#ifndef MIO_BN_MAXN
#define MIO_BN_MAXN 65
#endif


static inline void ReduceKernel(__local Dtype* lcl_mem,
                                unsigned int sum_stride,
                                unsigned int unit_id,
                                unsigned int unit_len)
{
    Dtype sum              = (Dtype)0.;
    unsigned int lcl_offset = unit_id * unit_len;

    for(unsigned int i = 0; i < unit_len; i += sum_stride)
    {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

static inline void
regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}



#if(MIO_BN_VARIANT == 0)

#define MIO_BN_SEGTMP (MIO_BN_HW * (MIO_BN_GRP0 / MIO_BN_HW))
#define MIO_BN_SEGMENT ((MIO_BN_SEGTMP > MIO_BN_NHW) ? (MIO_BN_NHW) : (MIO_BN_SEGTMP))
#define MIO_BN_NLOOP ((MIO_BN_NHW + MIO_BN_SEGMENT - 1) / MIO_BN_SEGMENT)
#define MIO_BN_SEGIHW (MIO_BN_SEGMENT / MIO_BN_HW)
#define MIO_BN_NLOOPM (MIO_BN_NLOOP - 1)
#define MIO_BN_SNHW (MIO_BN_NLOOPM * MIO_BN_SEGIHW)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias,
                               Dtype INHW,
                               Dtype epsilon
                               )
{

    // SPATIAL
    Dtype mean        = (Dtype)0.;
    Dtype variance    = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype pvscale     = (Dtype)0.;
    Dtype pvbias      = (Dtype)0.;
    Dtype batchvalues[MIO_BN_NLOOP];

    __local Dtype lcl_bias;
    __local Dtype lcl_scale;

    unsigned int index  = 0;
    unsigned int lid    = get_local_id(0);
    unsigned int grpid  = get_group_id(0);
    unsigned int chwid  = grpid * MIO_BN_HW + (lid % MIO_BN_HW);
    unsigned int lidihw = lid / MIO_BN_HW;
    unsigned int nid    = 0;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid < MIO_BN_SEGMENT)
    {
        __attribute__((opencl_unroll_hint(2)))
        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid            = n * MIO_BN_SEGIHW + lidihw;
            index          = nid * MIO_BN_CHW + chwid;
            batchvalues[n] = (Dtype)(*(in + index));
            mean += batchvalues[n];
            variance = mad(batchvalues[n], batchvalues[n], variance);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        batchvalues[MIO_BN_NLOOPM] =
            (index < MIO_BN_NCHW) ? (Dtype)(*(in + index)) : (Dtype)0.;
        mean += batchvalues[MIO_BN_NLOOPM];
        variance = mad(batchvalues[MIO_BN_NLOOPM], batchvalues[MIO_BN_NLOOPM], variance);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local Dtype lcl_data[MIO_BN_LDS_SIZE];

    // Reduce mean
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, (Dtype)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce variance
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);


    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + (Dtype)epsilon);
    pvscale     = (Dtype)lcl_scale;
    pvbias      = (Dtype)lcl_bias;

    if(lid < MIO_BN_SEGMENT)
    {
        //==== CALC NORM =======================
        Dtype inhat = (Dtype)0.;

        __attribute__((opencl_unroll_hint(2)))
        for(unsigned int n = 0; n < MIO_BN_NLOOPM; n++)
        { // apply normalization
            inhat      = (batchvalues[n] - mean) * invVariance;
            nid        = n * MIO_BN_SEGIHW + lidihw;
            index      = nid * MIO_BN_CHW + chwid;
            out[index] = (Dtype)mad(pvscale, inhat, pvbias);
        } // end for

        // Tail of loop
        inhat = (batchvalues[MIO_BN_NLOOPM] - mean) * invVariance;
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        if(index < MIO_BN_NCHW)
            out[index] = (Dtype)mad(pvscale, inhat, pvbias);
    }

} // end spatial norm

#elif(MIO_BN_VARIANT == 1)

//===========

#if(MIO_BN_HW >= 4096)
#define MIO_MAX_READ 3
#else
#define MIO_MAX_READ 2
#endif
#define RD_BLK 1
#define GRPRD (MIO_BN_GRP0 * RD_BLK * 4)
#define MIO_BN_REM4 (MIO_BN_NHW - ((MIO_BN_NHW / GRPRD) * GRPRD))
#define MIO_BN_LESS4 (MIO_BN_NHW - MIO_BN_REM4)
#define MIO_BN_CHUNK4 (MIO_MAX_READ * GRPRD)
#define MIO_BN_REMOUT4 (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK4) * MIO_BN_CHUNK4))
#define MIO_BN_LESSOUT4 (MIO_BN_NHW - MIO_BN_REMOUT4)
#define MIO_BN_REM (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_GRP0) * MIO_BN_GRP0))
#define MIO_BN_LESS (MIO_BN_NHW - MIO_BN_REM)
#define MIO_BN_CHUNK (MIO_MAX_READ * MIO_BN_GRP0)
#define MIO_BN_REMOUT (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK) * MIO_BN_CHUNK))
#define MIO_BN_LESSOUT (MIO_BN_NHW - MIO_BN_REMOUT)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias,
                               Dtype INHW,
                               Dtype epsilon
                               )
{

    // SPATIAL

    Dtype mean        = (Dtype)0.;
    Dtype variance    = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype pvscale, pvbias;

    __local Dtype lcl_bias;
    __local Dtype lcl_scale;

    uint index = 0;
    uint lid   = get_local_id(0);
    uint grpid = get_group_id(0);
    uint chwid = grpid * MIO_BN_HW;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_HW >= 4096)
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < MIO_BN_LESS4;
                                               k += GRPRD)
    {
        nidx  = k / MIO_BN_HW;
        hwidx = k - (nidx * MIO_BN_HW);
        index = nidx * MIO_BN_CHW + chwid + hwidx;
        read4 = *((const global Dtype4*)(in + index));
        mean += (Dtype)read4.x;
        mean += (Dtype)read4.y;
        mean += (Dtype)read4.z;
        mean += (Dtype)read4.w;
        variance = mad((Dtype)read4.x, (Dtype)read4.x, variance);
        variance = mad((Dtype)read4.y, (Dtype)read4.y, variance);
        variance = mad((Dtype)read4.z, (Dtype)read4.z, variance);
        variance = mad((Dtype)read4.w, (Dtype)read4.w, variance);
    }

#if(MIO_BN_REM4)
    unsigned int remkey = (lid << 2) + MIO_BN_LESS4;
    nidx                = remkey / MIO_BN_HW;
    hwidx               = remkey - (nidx * MIO_BN_HW);
    index               = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < MIO_BN_NCHW)
    {
        read4 = *((const global Dtype4*)(in + index));
        mean += (Dtype)read4.x;
        mean += (Dtype)read4.y;
        mean += (Dtype)read4.z;
        mean += (Dtype)read4.w;
        variance = mad((Dtype)read4.x, (Dtype)read4.x, variance);
        variance = mad((Dtype)read4.y, (Dtype)read4.y, variance);
        variance = mad((Dtype)read4.z, (Dtype)read4.z, variance);
        variance = mad((Dtype)read4.w, (Dtype)read4.w, variance);
    }

#endif

#else
    __attribute__((opencl_unroll_hint(4))) for(unsigned int k = lid; k < MIO_BN_LESS;
                                               k += MIO_BN_GRP0)
    {
        nidx            = k / MIO_BN_HW;
        hwidx           = k - (nidx * MIO_BN_HW);
        index           = nidx * MIO_BN_CHW + chwid + hwidx;
        Dtype xin = (Dtype)(*(in + index));
        mean += xin;
        variance = mad(xin, xin, variance);
    }
#if(MIO_BN_REM)
    if(lid < MIO_BN_REM)
    {
        unsigned int remkey = lid + MIO_BN_LESS;
        nidx                = remkey / MIO_BN_HW;
        hwidx               = remkey - (nidx * MIO_BN_HW);
        index               = nidx * MIO_BN_CHW + chwid + hwidx;
        Dtype xin = (index < MIO_BN_NCHW) ? (Dtype)(*(in + index)) : (Dtype)0.;
        mean += xin;
        variance = mad(xin, xin, variance);
    }
#endif
#endif
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, (Dtype)INHW);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + epsilon);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

#if(MIO_BN_REM == 0)
    for(unsigned int k = lid; k < MIO_BN_LESS;
                                               k += MIO_BN_GRP0)
    {
        nidx  = k / MIO_BN_HW;
        hwidx = k - (nidx * MIO_BN_HW);
        index = nidx * MIO_BN_CHW + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
#else
    Dtype xhat[MIO_MAX_READ];
    for(unsigned int k = (MIO_MAX_READ * lid);
                                               k < MIO_BN_LESSOUT;
                                               k += MIO_BN_CHUNK)
    {
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
        {
            unsigned int l = k + j;
            nidx           = l / MIO_BN_HW;
            hwidx          = l - (nidx * MIO_BN_HW);
            index          = nidx * MIO_BN_CHW + chwid + hwidx;
            xhat[j]        = ((Dtype)(*(in + index)) - mean) * invVariance;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
        {
            unsigned int l = k + j;
            nidx           = l / MIO_BN_HW;
            hwidx          = l - (nidx * MIO_BN_HW);
            index          = nidx * MIO_BN_CHW + chwid + hwidx;
            *(out + index) = (Dtype)mad(pvscale, xhat[j], pvbias);
        }
    } // end for

#if(MIO_BN_REMOUT)
    unsigned int remkeyout = (MIO_MAX_READ * lid) + MIO_BN_LESSOUT;
    for(unsigned int j = 0; j < MIO_MAX_READ; j++)
    {
        unsigned int l  = remkeyout + j;
        nidx            = l / MIO_BN_HW;
        hwidx           = l - (nidx * MIO_BN_HW);
        index           = nidx * MIO_BN_CHW + chwid + hwidx;
        Dtype xin = (index < MIO_BN_NCHW) ? (Dtype)(*(in + index)) : (Dtype)0.;
        xhat[j]         = (xin - mean) * invVariance;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for(unsigned int j = 0; j < MIO_MAX_READ; j++)
    {
        unsigned int l = remkeyout + j;
        nidx           = l / MIO_BN_HW;
        hwidx          = l - (nidx * MIO_BN_HW);
        index          = nidx * MIO_BN_CHW + chwid + hwidx;
        if(index < MIO_BN_NCHW)
        {
            *(out + index) = (Dtype)mad(pvscale, xhat[j], pvbias);
        }
    }
#endif
#endif


} // end spatial norm

#elif(MIO_BN_VARIANT == 2) // MULTI-KERNEL reduction for > 33M elements

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatialNorm(const __global Dtype* __restrict in,
                                   __global Dtype* __restrict out,
                                   const __global Dtype* __restrict scale,
                                   const __global Dtype* __restrict bias)
{

    // SPATIAL
    Dtype mean        = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype inhat       = (Dtype)0.;
    Dtype pvt_scale   = (Dtype)0.;
    Dtype pvt_bias    = (Dtype)0.;
    __local Dtype lcl_mean, lcl_ivar, lcl_scale, lcl_bias;

    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx           = xgid * MIO_BN_HW;
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;

    // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    if(get_local_id(1) == 0)
    {
        lcl_scale = *(scale + xgid);
        lcl_bias  = *(bias + xgid);
        lcl_mean  = *(out + meanstashindex); // load stashed mean
        lcl_ivar  = *(out + varstashindex);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
        mean        = lcl_mean;
        invVariance = lcl_ivar;
        pvt_scale   = lcl_scale;
        pvt_bias    = lcl_bias;
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index        = n * MIO_BN_CHW + cidx + ygid;
            Dtype inhat = (*(in + index) - mean) * invVariance;
            // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
        } // end for(n)
    }     // end if(inImgIndex)
} // end spatial norm

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatialFinalMeanVariance(
    __global Dtype* __restrict meanvarbuff,
    Dtype INHW,
    Dtype epsilon
    )
{
    Dtype variance             = (Dtype)0.;
    Dtype invVariance          = (Dtype)0.;
    Dtype mean                 = (Dtype)0.;
    unsigned int lid            = get_local_id(1);
    unsigned int ygrp_id        = get_group_id(1);
    unsigned int xgid           = get_global_id(0);
    unsigned int ygrp_sz        = get_local_size(1);
    unsigned int yngrps         = get_num_groups(1);
    unsigned int cidx           = xgid * MIO_BN_HW;
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;
    unsigned int commitID       = 0;

    for(int gn = 0; gn < yngrps; gn++)
    {
        unsigned int offset    = gn * ygrp_sz + lid;
        unsigned int meanindex = cidx + ygrp_sz * offset;
        unsigned int varindex  = cidx + ygrp_sz * offset + 2;
        if(offset < yngrps)
        { // modify to span larger number of groups
            mean += *(meanvarbuff + meanindex);
            variance += *(meanvarbuff + varindex); // load per group variance
        }
    }


    __local Dtype lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_NGRPS > 64)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, (Dtype)INHW);
#elif(MIO_BN_NGRPS <= 64)
    regLDSreduce(&mean, lcl_data, lid, (Dtype)INHW);
    commitID = 0;
#else
    mean = (Dtype)0.;
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        mean += lcl_data[i];
    }

#endif

    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_NGRPS > 256)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
#elif(MIO_BN_NGRPS > 64)
    regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
#elif(MIO_BN_NGRPS > 16)
    regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
#else //(MIO_BN_NGRPS <= 16)
    variance = (Dtype)0.;
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        variance += lcl_data[i];
    }
#endif



    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + epsilon);
    if(lid == commitID)
    {
        meanvarbuff[meanstashindex] = mean;        // stash mean
        meanvarbuff[varstashindex]  = invVariance; // stash mean
    }


}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatialMeanVariance(const __global Dtype* __restrict in,
                                           __global Dtype* __restrict mvbuff)
{

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx      = xgid * MIO_BN_HW;
    unsigned int meanindex = cidx + ygrp_sz * ygrp_id;
    unsigned int varindex  = meanindex + 2;
    Dtype mean            = (Dtype)0.;
    Dtype variance        = (Dtype)0.;
    Dtype value           = (Dtype)0.;

    if(ygid < MIO_BN_HW)
    {
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            value = *(in + index);
            mean += value;
            variance = mad(value, value, variance);
        }
    }



    __local Dtype lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, ylid, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, ylid, 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid == 0)
    {
        mvbuff[meanindex] = mean;
        mvbuff[varindex]  = variance;
    }
} // end spatial mean kernel

#elif(MIO_BN_VARIANT == 3)

// This kernel implies the image is greater than a wavefront, but smaller than 257
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias,
                               Dtype INHW,
                               Dtype epsilon
                               )
{

    // SPATIAL
    Dtype mean        = (Dtype)0.;
    Dtype variance    = (Dtype)0.;
    Dtype invVariance = (Dtype)0.;
    Dtype inhat       = (Dtype)0.;
    Dtype pvscale, pvbias;

    __local Dtype lcl_bias;
    __local Dtype lcl_scale;

    unsigned int index;
    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int cidx  = grpid * MIO_BN_HW;

#if(MIO_BN_N < MIO_BN_MAXN)
    Dtype minibatch[MIO_BN_N];
#endif

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // MEAN
    if(lid < MIO_BN_HW)
    {
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index        = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            minibatch[n] = (Dtype)(*(in + index));
            mean += minibatch[n];
            variance = mad(minibatch[n], minibatch[n], variance);
#else
            Dtype xin = (Dtype)(*(in + index));
            mean += xin;
            variance = mad(xin, xin, variance);
#endif
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    __local Dtype lcl_data[MIO_BN_LDS_SIZE];

    // Reduce mean
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, (Dtype)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce variance
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);


    barrier(CLK_LOCAL_MEM_FENCE);
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + (Dtype)epsilon);

    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(lid < MIO_BN_HW)
    {
        pvscale = lcl_scale;
        pvbias  = lcl_bias;

        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index      = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            inhat      = (minibatch[n] - mean) * invVariance; // (in[index] - mean) * invVariance;
#else
            inhat = ((Dtype)(*(in + index)) - mean) * invVariance;
// printf("lid: %d, index: %d, n: %d, mean: %f, invVar: %f\n", lid, index, n, mean, invVariance);
#endif
            out[index] = (Dtype)mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if


} // end spatial norm

#endif
)";
}



INSTANTIATE_CLASS_GPU(MIOpenBatchNormOp_GPU);


}//namespace hypertea
