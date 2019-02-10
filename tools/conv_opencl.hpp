std::string conv_opencl_funcs = R"(#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 786432
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 8388608
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 512
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 512
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 512
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 512
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 262144
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 262144
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 9
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 9
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 4
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 4
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 3
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 32
#ifdef MG
#undef MG
#endif
#define MG 32
#ifdef M
#undef M
#endif
#define M 32
#ifdef N
#undef N
#endif
#define N 262144
#ifdef KG
#undef KG
#endif
#define KG 243
#ifdef K
#undef K
#endif
#define K 243
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void conv1_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out, __global const Dtype* __restrict bias) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
__global const Dtype* Dptr = bias;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + biasval;
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 8388608
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 4194304
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 512
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 256
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 512
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 256
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 262144
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 65536
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 4
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 4
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 2
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 2
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 32
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 64
#ifdef MG
#undef MG
#endif
#define MG 64
#ifdef M
#undef M
#endif
#define M 64
#ifdef N
#undef N
#endif
#define N 65536
#ifdef KG
#undef KG
#endif
#define KG 512
#ifdef K
#undef K
#endif
#define K 512
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void conv2_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out, __global const Dtype* __restrict bias) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
__global const Dtype* Dptr = bias;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + biasval;
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 4194304
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 256
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 256
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 65536
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 4
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 4
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 2
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 2
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 64
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1024
#ifdef K
#undef K
#endif
#define K 1024
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void conv3_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out, __global const Dtype* __restrict bias) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
__global const Dtype* Dptr = bias;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + biasval;
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res1_conv1_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res1_conv2_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res2_conv1_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res2_conv2_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res3_conv1_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res3_conv2_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res4_conv1_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res4_conv2_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res5_conv1_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 2097152
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 128
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 128
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 16384
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 3
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 3
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 1
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 1
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 128
#ifdef MG
#undef MG
#endif
#define MG 128
#ifdef M
#undef M
#endif
#define M 128
#ifdef N
#undef N
#endif
#define N 16384
#ifdef KG
#undef KG
#endif
#define KG 1152
#ifdef K
#undef K
#endif
#define K 1152
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void res5_conv2_forward(__global const Dtype* __restrict im_in, __global const Dtype* __restrict wg, __global Dtype* __restrict im_out) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[TSM][TSK + v_pad_A];
volatile __local Dtype Bsub[TSK][TSN + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
#pragma unroll 4
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
bool in_range = true;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_A_off
#undef v_A_off
#endif
#define v_A_off 131072
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 2097152
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 4194304
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 128
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 256
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 128
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 256
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 16384
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 65536
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 4
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 4
#ifdef v_ks
#undef v_ks
#endif
#define v_ks 16
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 2
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 2
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 2
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 2
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 128
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 64
#ifdef MG
#undef MG
#endif
#define MG 64
#ifdef M
#undef M
#endif
#define M 64
#ifdef N
#undef N
#endif
#define N 65536
#ifdef KG
#undef KG
#endif
#define KG 2048
#ifdef K
#undef K
#endif
#define K 2048
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void deconv5_1_forward(__global const Dtype* __restrict im_out, __global const Dtype* __restrict wg, __global Dtype* __restrict im_in, __global const Dtype* __restrict bias) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[16][8 + v_pad_A];
volatile __local Dtype Bsub[8][128 + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_out + v_B_off * batch;
__global Dtype* Cptr = im_in + v_C_off * batch;
__global const Dtype* Dptr = bias;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
int kidx = (v_ks - 1 - tiledIndex % v_ks) + (offM + row) * v_ks;
int midx = tiledIndex / v_ks;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[kidx + (v_fout / v_g * v_ks) * midx];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
bool in_range = true;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) - v_p_0;
imageIndex = imageIndex / v_imso_0;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im / v_s_0;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0 * v_s_0 && d_iter_im % v_s_0 == 0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im / v_s_1;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1 * v_s_1 && d_iter_im % v_s_1 == 0;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + biasval;
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_A_off
#undef v_A_off
#endif
#define v_A_off 32768
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 4194304
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 8388608
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 256
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 512
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 256
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 512
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 65536
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 262144
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 4
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 4
#ifdef v_ks
#undef v_ks
#endif
#define v_ks 16
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 2
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 2
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 2
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 2
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 64
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 32
#ifdef MG
#undef MG
#endif
#define MG 32
#ifdef M
#undef M
#endif
#define M 32
#ifdef N
#undef N
#endif
#define N 262144
#ifdef KG
#undef KG
#endif
#define KG 1024
#ifdef K
#undef K
#endif
#define K 1024
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void deconv5_2_forward(__global const Dtype* __restrict im_out, __global const Dtype* __restrict wg, __global Dtype* __restrict im_in, __global const Dtype* __restrict bias) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[16][8 + v_pad_A];
volatile __local Dtype Bsub[8][128 + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_out + v_B_off * batch;
__global Dtype* Cptr = im_in + v_C_off * batch;
__global const Dtype* Dptr = bias;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
int kidx = (v_ks - 1 - tiledIndex % v_ks) + (offM + row) * v_ks;
int midx = tiledIndex / v_ks;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[kidx + (v_fout / v_g * v_ks) * midx];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
bool in_range = true;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) - v_p_0;
imageIndex = imageIndex / v_imso_0;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im / v_s_0;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0 * v_s_0 && d_iter_im % v_s_0 == 0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im / v_s_1;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1 * v_s_1 && d_iter_im % v_s_1 == 0;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + biasval;
}
}
}
}
}
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#ifdef v_g
#undef v_g
#endif
#define v_g 1
#ifdef v_A_off
#undef v_A_off
#endif
#define v_A_off 7776
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 8388608
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 786432
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 512
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 512
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 512
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 512
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 262144
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 262144
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 9
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 9
#ifdef v_ks
#undef v_ks
#endif
#define v_ks 81
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 4
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 4
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 1
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 1
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 1
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 1
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 32
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 3
#ifdef MG
#undef MG
#endif
#define MG 3
#ifdef M
#undef M
#endif
#define M 3
#ifdef N
#undef N
#endif
#define N 262144
#ifdef KG
#undef KG
#endif
#define KG 2592
#ifdef K
#undef K
#endif
#define K 2592
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 1
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 1
#ifdef TSM
#undef TSM
#endif
#define TSM 16
#ifdef TSN
#undef TSN
#endif
#define TSN 128
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 8
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 8
#ifdef VWN
#undef VWN
#endif
#define VWN 4
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 4
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 16
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK*TSN)/(RTSM*RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1)/(TSK*2) + 1)*2)
__kernel
__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((vec_type_hint(Dtype4)))
void deconv5_3_forward(__global const Dtype* __restrict im_out, __global const Dtype* __restrict wg, __global Dtype* __restrict im_in, __global const Dtype* __restrict bias) {
const int tidn = get_local_id(0);
const int tidm = get_local_id(1);
const int offN = TSN*get_group_id(0);
const int offM = TSM*get_group_id(1);
volatile __local Dtype Asub[16][8 + v_pad_A];
volatile __local Dtype Bsub[8][128 + v_pad_B];
int batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_out + v_B_off * batch;
__global Dtype* Cptr = im_in + v_C_off * batch;
__global const Dtype* Dptr = bias;
{
Dtype4 Creg[WPTM][WPTN/VWN];
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
#pragma unroll 1
for (int t = 0; t < v_num_tiles; ++t) {
{
for (int la = 0; la < LPTA; ++la) {
int tid = tidm * RTSN + tidn;
int id = la * RTSN * RTSM + tid;
int row = id / TSK;
int col = id % TSK;
int tiledIndex = TSK * t + col;
int kidx = (v_ks - 1 - tiledIndex % v_ks) + (offM + row) * v_ks;
int midx = tiledIndex / v_ks;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[kidx + (v_fout / v_g * v_ks) * midx];
} else {
Asub[row][col] = 0.0;
}
}
}
{
#pragma unroll 4
for (int lb = 0; lb < LPTB; ++lb) {
int tid = tidm * RTSN + tidn;
int id = lb * RTSN * RTSM + tid;
int col = id % TSN;
int row = id / TSN;
int tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int d_iter_0;
int d_temp_0;
int d_iter_1;
int d_temp_1;
bool in_range = true;
int imageIndex = offN + col;
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
d_temp_1 = (imageIndex % v_imso_1) - v_p_1;
imageIndex = imageIndex / v_imso_1;
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
d_temp_0 = (imageIndex % v_imso_0) - v_p_0;
imageIndex = imageIndex / v_imso_0;
int d_iter_im;
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im / v_s_0;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0 * v_s_0 && d_iter_im % v_s_0 == 0;
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im / v_s_1;
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1 * v_s_1 && d_iter_im % v_s_1 == 0;
if (in_range) {
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
barrier(CLK_LOCAL_MEM_FENCE);
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
#pragma unroll 1
for (int kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int ku=0; ku<TSK_UNROLL; ++ku) {
int k = kt + ku;
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
int col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
#pragma unroll
for (int wm=0; wm<WPTM/VWM; ++wm) {
int row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 4][k];
VEC_4_2(Areg) = Asub[row + 8][k];
VEC_4_3(Areg) = Asub[row + 12][k];
#pragma unroll
for (int wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

barrier(CLK_LOCAL_MEM_FENCE);
}
}
#pragma unroll
for (int wm=0; wm<WPTM; ++wm) {
int globalRow = offM + tidm + wm * RTSM;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int wn=0; wn<WPTN; ++wn) {
int globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + biasval;
}
}
}
}
}
)";
