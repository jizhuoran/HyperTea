
std::string bn_opencl_funcs = R"(

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


#define Dtype half
#define Dtype4 half4


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
bn1_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
bn1_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 262144;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 262144;
                                               k += 4096) {
        nidx  = k / 262144;
        hwidx = k - (nidx * 262144);
        index = nidx * 8388608 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    bn1_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)3.814697265625e-06);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    bn1_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)3.814697265625e-06);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 262144; k += 1024) {
        nidx  = k / 262144;
        hwidx = k - (nidx * 262144);
        index = nidx * 8388608 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
bn2_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
bn2_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 65536;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 65536;
                                               k += 4096) {
        nidx  = k / 65536;
        hwidx = k - (nidx * 65536);
        index = nidx * 4194304 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    bn2_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)1.52587890625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    bn2_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)1.52587890625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 65536; k += 1024) {
        nidx  = k / 65536;
        hwidx = k - (nidx * 65536);
        index = nidx * 4194304 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
bn3_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
bn3_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    bn3_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    bn3_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res1_bn1_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res1_bn1_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res1_bn1_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res1_bn1_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res1_bn2_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res1_bn2_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res1_bn2_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res1_bn2_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res2_bn1_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res2_bn1_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res2_bn1_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res2_bn1_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res2_bn2_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res2_bn2_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res2_bn2_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res2_bn2_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res3_bn1_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res3_bn1_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res3_bn1_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res3_bn1_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res3_bn2_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res3_bn2_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res3_bn2_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res3_bn2_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res4_bn1_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res4_bn1_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res4_bn1_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res4_bn1_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res4_bn2_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res4_bn2_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res4_bn2_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res4_bn2_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res5_bn1_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res5_bn1_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res5_bn1_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res5_bn1_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
res5_bn2_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
res5_bn2_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 16384;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 16384;
                                               k += 4096) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res5_bn2_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)6.103515625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    res5_bn2_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)6.103515625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 16384; k += 1024) {
        nidx  = k / 16384;
        hwidx = k - (nidx * 16384);
        index = nidx * 2097152 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
de_bn1_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
de_bn1_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 65536;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 65536;
                                               k += 4096) {
        nidx  = k / 65536;
        hwidx = k - (nidx * 65536);
        index = nidx * 4194304 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    de_bn1_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)1.52587890625e-05);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    de_bn1_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)1.52587890625e-05);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 65536; k += 1024) {
        nidx  = k / 65536;
        hwidx = k - (nidx * 65536);
        index = nidx * 4194304 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
static inline void
de_bn2_forward_regLDSreduce(Dtype* value, __local Dtype* data, unsigned int localID, Dtype scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (1024 >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, 1024);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}
        

__attribute__((reqd_work_group_size(1024, 1, 1))) __kernel void
de_bn2_forward_MIOpenBatchNormFwdTrainSpatial(const __global Dtype* __restrict in,
                               __global Dtype* __restrict out,
                               __constant Dtype* __restrict scale,
                               __constant Dtype* __restrict bias
                               ) {

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
    uint chwid = grpid * 262144;
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0) {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    Dtype4 read4;
    for(unsigned int k = lid << 2; k < 262144;
                                               k += 4096) {
        nidx  = k / 262144;
        hwidx = k - (nidx * 262144);
        index = nidx * 8388608 + chwid + hwidx;
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
     

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------

    local Dtype lcl_data[1024];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    de_bn2_forward_regLDSreduce(&mean, lcl_data, lid, (Dtype)3.814697265625e-06);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (1024 >> 1); red > 256; red >>= 1) {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    de_bn2_forward_regLDSreduce(&variance, lcl_data, lid, (Dtype)3.814697265625e-06);
    barrier(CLK_LOCAL_MEM_FENCE);

    // REDUCTION COMPLETE ---------------------------
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + 1e-05);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

    
    for(unsigned int k = lid; k < 262144; k += 1024) {
        nidx  = k / 262144;
        hwidx = k - (nidx * 262144);
        index = nidx * 8388608 + chwid + hwidx;
        out[index] =
            (Dtype)mad(pvscale, ((Dtype)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
        


} // end spatial norm
        
)";
        