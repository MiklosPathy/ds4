namespace ds41_engram_e4m3 {
// Codes are row-major; exponent grid uses ceil(K/32) blocks per output block.
// Integer reconstruction prevents compiler reassociation of scale into the dot.
__device__ __forceinline__ float exact_value(unsigned c,int scale) {
    unsigned m=c&7u,e=(c>>3)&15u;
    unsigned sign=(c&128u)<<24;
    if (!(c&127u)) return __uint_as_float(sign);
    int p;unsigned fraction;
    if(e) {p=(int)e-7+scale;fraction=m<<20;}
    else {unsigned high=31u-__clz(m);p=scale-9+(int)high;fraction=(m-(1u<<high))<<(23u-high);}
    return __uint_as_float(sign|((unsigned)(p+127)<<23)|fraction);
}
__global__ static void e4m3_scalar(
        float *out,
        const uint8_t *w, const int32_t *exps,
        const float *x,
        uint32_t in_dim,
        uint64_t out_dim) {
    extern __shared__ float shx[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t rows_per_block = blockDim.x >> 5u;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    const uint64_t row = (uint64_t)blockIdx.x * rows_per_block + wave;
    if (row >= out_dim) return;
    const uint8_t *wr = w + row * (uint64_t)in_dim;
    const int32_t *er = exps + (row >> 5u) * ((in_dim + 31u) >> 5u);
    float acc = 0.0f;
    uint32_t i = lane;
    for (; i + 224u < in_dim; i += 256u) {
        acc += exact_value(wr[i],er[(i)>>5u]) * shx[i];
        acc += exact_value(wr[i + 32u],er[(i + 32u)>>5u]) * shx[i + 32u];
        acc += exact_value(wr[i + 64u],er[(i + 64u)>>5u]) * shx[i + 64u];
        acc += exact_value(wr[i + 96u],er[(i + 96u)>>5u]) * shx[i + 96u];
        acc += exact_value(wr[i + 128u],er[(i + 128u)>>5u]) * shx[i + 128u];
        acc += exact_value(wr[i + 160u],er[(i + 160u)>>5u]) * shx[i + 160u];
        acc += exact_value(wr[i + 192u],er[(i + 192u)>>5u]) * shx[i + 192u];
        acc += exact_value(wr[i + 224u],er[(i + 224u)>>5u]) * shx[i + 224u];
    }
    for (; i < in_dim; i += 32u) {
        acc += exact_value(wr[i],er[(i)>>5u]) * shx[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}
template<unsigned TT,unsigned WAVES=8,unsigned SEG=256>
__global__ static void e4m3_tiny(float *out,const uint8_t *w,const int32_t *exps,const float *x) {
    constexpr unsigned K=6144,N=25600;
    static_assert(K % SEG == 0, "Engram LDS segment must divide K");
    const unsigned tid=threadIdx.x,lane=tid&31u,wave=tid>>5u;
    const unsigned row=blockIdx.x*WAVES+wave,token=blockIdx.y*TT;
    const uint8_t *wr=w+(uint64_t)row*K;
    const int32_t *er=exps+(row>>5u)*(K>>5u);
    __shared__ float tile[TT][SEG];
    float acc[TT]={};
    for(unsigned base=0;base<K;base+=SEG) {
        for(unsigned j=tid;j<TT*SEG;j+=WAVES*32u) {
            const unsigned t=j/SEG,k=j%SEG;
            tile[t][k]=x[(uint64_t)(token+t)*K+base+k];
        }
        __syncthreads();
        /* Match the scalar production kernel's eight explicit FMAs per
         * 256-wide group. Keeping that loop shape preserves bitwise parity
         * under the production fast-math flags. */
        #pragma unroll 1
        for(unsigned group=0;group<SEG;group+=256u) {
            const unsigned k=group+lane;
#pragma unroll
            for(unsigned t=0;t<TT;t++) {
                acc[t]+=exact_value(wr[base+k],er[(base+k)>>5u])*tile[t][k];
                acc[t]+=exact_value(wr[base+k+32u],er[(base+k+32u)>>5u])*tile[t][k+32u];
                acc[t]+=exact_value(wr[base+k+64u],er[(base+k+64u)>>5u])*tile[t][k+64u];
                acc[t]+=exact_value(wr[base+k+96u],er[(base+k+96u)>>5u])*tile[t][k+96u];
                acc[t]+=exact_value(wr[base+k+128u],er[(base+k+128u)>>5u])*tile[t][k+128u];
                acc[t]+=exact_value(wr[base+k+160u],er[(base+k+160u)>>5u])*tile[t][k+160u];
                acc[t]+=exact_value(wr[base+k+192u],er[(base+k+192u)>>5u])*tile[t][k+192u];
                acc[t]+=exact_value(wr[base+k+224u],er[(base+k+224u)>>5u])*tile[t][k+224u];
            }
        }
        // Every reader finishes before any thread overwrites the next segment.
        __syncthreads();
    }
#pragma unroll
    for(unsigned t=0;t<TT;t++) {
        const float total=warp_sum_f32(acc[t]);
        if(lane==0)out[(uint64_t)(token+t)*N+row]=total;
    }
}

} // namespace ds41_engram_e4m3
