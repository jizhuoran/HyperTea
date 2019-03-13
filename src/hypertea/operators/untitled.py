MIO_BN_LDS_SIZE = self.ldsgcn
MIO_BN_NGRPS = int(math.ceil(float(self.ygridsize) / self.ylocalsize))






reduce_mean = f'''
    for(unsigned int red = ({self.ylocalsize} >> 1); red > 256; red >>= 1) {{
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    regLDSreduce(&mean, lcl_data, lid, (Dtype)INHW);
''' if MIO_BN_NGRPS > 64 else '''
    regLDSreduce(&mean, lcl_data, lid, (Dtype)INHW);
    commitID = 0;

'''


if MIO_BN_NGRPS > 256:
    reduce_var = f'''
        for(unsigned int red = ({self.ylocalsize} >> 1); red > 256; red >>= 1) {{
            if(lid < red)
                lcl_data[lid] += lcl_data[lid + red];
            barrier(CLK_LOCAL_MEM_FENCE);
        }}
        regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
    '''
elif MIO_BN_NGRPS > 16
    reduce_var = '''
        regLDSreduce(&variance, lcl_data, lid, (Dtype)INHW);
    '''
else:
    reduce_var = '''
        variance = (Dtype)0.;
        for(int i = 0; i < {MIO_BN_NGRPS}; i++) {{
            variance += lcl_data[i];
        }}
'''




f'''
__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
MIOpenBatchNormFwdTrainSpatialNorm(const __global Dtype* __restrict in,
                                   __global Dtype* __restrict out,
                                   const __global Dtype* __restrict scale,
                                   const __global Dtype* __restrict bias)
{{

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
    unsigned int cidx           = xgid * {self.in_cstride};
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;

    // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    if(get_local_id(1) == 0) {{
        lcl_scale = *(scale + xgid);
        lcl_bias  = *(bias + xgid);
        lcl_mean  = *(out + meanstashindex); // load stashed mean
        lcl_ivar  = *(out + varstashindex);
    }}
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < {self.in_cstride}) {{
        mean        = lcl_mean;
        invVariance = lcl_ivar;
        pvt_scale   = lcl_scale;
        pvt_bias    = lcl_bias;
        for(unsigned int n = 0; n < {self.batch_size}; n++) {{ // apply normalization
            index        = n * {self.in_nstride} + cidx + ygid;
            Dtype inhat = (*(in + index) - mean) * invVariance;
            // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
        }} // end for(n)
    }}     // end if(inImgIndex)
}} // end spatial norm

__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
MIOpenBatchNormFwdTrainSpatialFinalMeanVariance(
    __global Dtype* __restrict meanvarbuff,
    Dtype INHW,
    Dtype epsilon
    ) {{
    Dtype variance             = (Dtype)0.;
    Dtype invVariance          = (Dtype)0.;
    Dtype mean                 = (Dtype)0.;
    unsigned int lid            = get_local_id(1);
    unsigned int ygrp_id        = get_group_id(1);
    unsigned int xgid           = get_global_id(0);
    unsigned int ygrp_sz        = get_local_size(1);
    unsigned int yngrps         = get_num_groups(1);
    unsigned int cidx           = xgid * {self.in_cstride};
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;
    unsigned int commitID       = 0;

    for(int gn = 0; gn < yngrps; gn++) {{
        unsigned int offset    = gn * ygrp_sz + lid;
        unsigned int meanindex = cidx + ygrp_sz * offset;
        unsigned int varindex  = cidx + ygrp_sz * offset + 2;
        if(offset < yngrps) {{ 
            // modify to span larger number of groups
            mean += *(meanvarbuff + meanindex);
            variance += *(meanvarbuff + varindex); // load per group variance
        }}
    }}


    __local Dtype lcl_data[{MIO_BN_LDS_SIZE}];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    {reduce_mean}

    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    {reduce_var}



    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + epsilon);
    if(lid == commitID) {{
        meanvarbuff[meanstashindex] = mean;        // stash mean
        meanvarbuff[varstashindex]  = invVariance; // stash mean
    }}


}}

__attribute__((reqd_work_group_size({self.xlocalsize}, {self.ylocalsize}, {self.zlocalsize}))) __kernel void
MIOpenBatchNormFwdTrainSpatialMeanVariance(const __global Dtype* __restrict in,
                                           __global Dtype* __restrict mvbuff) {{

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx      = xgid * {self.in_cstride};
    unsigned int meanindex = cidx + ygrp_sz * ygrp_id;
    unsigned int varindex  = meanindex + 2;
    Dtype mean            = (Dtype)0.;
    Dtype variance        = (Dtype)0.;
    Dtype value           = (Dtype)0.;

    if(ygid < {self.in_cstride}) {{
        for(unsigned int n = 0; n < {self.batch_size}; n++) {{
            index = n * {self.in_nstride} + cidx + ygid;
            value = *(in + index);
            mean += value;
            variance = mad(value, value, variance);
        }}
    }}



    __local Dtype lcl_data[{MIO_BN_LDS_SIZE}];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = ({self.ylocalsize} >> 1); red > 256; red >>= 1) {{
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    regLDSreduce(&mean, lcl_data, ylid, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = ({self.ylocalsize} >> 1); red > 256; red >>= 1) {{
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    regLDSreduce(&variance, lcl_data, ylid, 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid == 0) {{
        mvbuff[meanindex] = mean;
        mvbuff[varindex]  = variance;
    }}
}} // end spatial mean kernel


'''