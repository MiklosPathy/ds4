/* DeepSeek V4.1 baseline primitives. Float-addressable storage preserves the released BF16/FP8/FP4 graph boundaries. */
#ifdef __HIP_PLATFORM_AMD__

/* AMDGPU ignores float_control(precise), and HIP's math wrappers select native trig while preprocessing -ffast-math. Use explicit OCML operations and division instructions so the existing translation unit keeps its tuned arithmetic. */
__device__ static float v41_add(float x, float y) { return __ocml_add_rte_f32(x, y); }
__device__ static float v41_sub(float x, float y) { return __ocml_sub_rte_f32(x, y); }
__device__ static float v41_mul(float x, float y) { return __ocml_mul_rte_f32(x, y); }
__device__ static float v41_div(float x, float y) {
    /* ROCm 10 declares but does not define __ocml_div_rte_f32. A local reciprocal(off) pragma still leaves afn, which AMDGPU lowers to an approximate reciprocal. Encode the backend's full F32 division refinement, including temporary denorm preservation, as one indivisible block. */
    float q, d, n, r, e;
    uint32_t mode;
#define DS4_V41_DIV_ASM(VCC) asm volatile( \
        "v_div_scale_f32 %1, " VCC ", %7, %7, %6\n\t" \
        "v_div_scale_f32 %2, " VCC ", %6, %7, %6\n\t" \
        "v_rcp_f32 %3, %1\n\t" \
        "s_getreg_b32 %5, hwreg(HW_REG_MODE, 4, 2)\n\t" \
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3\n\t" \
        "v_fma_f32 %4, -%1, %3, 1.0\n\t" \
        "v_fma_f32 %3, %4, %3, %3\n\t" \
        "v_mul_f32 %0, %2, %3\n\t" \
        "v_fma_f32 %4, -%1, %0, %2\n\t" \
        "v_fma_f32 %0, %4, %3, %0\n\t" \
        "v_fma_f32 %4, -%1, %0, %2\n\t" \
        "s_setreg_b32 hwreg(HW_REG_MODE, 4, 2), %5\n\t" \
        "v_div_fmas_f32 %0, %4, %3, %0\n\t" \
        "v_div_fixup_f32 %0, %0, %7, %6\n\t" \
        : "=&v"(q), "=&v"(d), "=&v"(n), "=&v"(r), "=&v"(e), "=&s"(mode) \
        : "v"(x), "v"(y) : "vcc", "memory")
    /* The HIP compiler need not define a wave-size macro. Its target builtin folds before assembly, selecting the valid VCC operand for the actual wave mode. */
#if defined(__AMDGCN__)
    if (__builtin_amdgcn_wavefrontsize() == 64) DS4_V41_DIV_ASM("vcc");
    else
#endif
        DS4_V41_DIV_ASM("vcc_lo");
#undef DS4_V41_DIV_ASM
    return q;
}

__device__ static float v41_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__device__ static float v41_pow2_ceil(float x) {
    const uint32_t bits = __float_as_uint(x);
    return __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

__device__ static float v41_sum32(float x) {
    for (int delta = 16; delta; delta >>= 1)
        x = v41_add(x, __shfl_down(x, delta, 32));
    return __shfl(x, 0, 32);
}

__global__ static void v41_bf16_kernel(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = v41_bf16(x[i]);
}

__global__ static void v41_quantize_kernel(float *x, uint32_t format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    const uint32_t lane = threadIdx.x;
    const uint64_t i = (uint64_t)blockIdx.x * block + lane;
    const float value = lane < block ? v41_bf16(x[i]) : 0.0f;
    float amax = fabsf(value);
    for (int delta = 16; delta; delta >>= 1)
        amax = fmaxf(amax, __shfl_down(amax, delta, 32));
    amax = __shfl(amax, 0, 32);
    float result;
    if (format == DS4_V41_FP8_E8M0) {
        const float scale = v41_pow2_ceil(v41_mul(fmaxf(amax, 1.0e-4f), 1.0f / 448.0f));
        result = v41_mul(dsv4_e4m3fn_dequant_dev(v41_div(fabsf(value), scale)), scale);
    } else {
        const float scale = format == DS4_V41_FP4_E4M3 ?
            dsv4_e4m3fn_dequant_dev(v41_div(fmaxf(amax, 0.01171875f), 6.0f)) :
            v41_pow2_ceil(v41_mul(fmaxf(amax, 7.052966104933725e-38f), 1.0f / 6.0f));
        result = v41_mul(dsv4_e2m1fn_dequant_dev(v41_div(fabsf(value), scale)), scale);
    }
    if (lane < block) {
        /* Preserve signed zero even though the surrounding translation unit permits -fno-signed-zeros. */
        const uint32_t bits = (__float_as_uint(v41_bf16(result)) & 0x7fffffffu) | (__float_as_uint(value) & 0x80000000u);
        x[i] = __uint_as_float(bits);
    }
}

extern "C" int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                                      ds4_v41_activation_format format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    if (!width || !rows || format < DS4_V41_BF16 || format > DS4_V41_FP4_E4M3 ||
        (format != DS4_V41_BF16 && width % block) ||
        !cuda_tensor_has_elems2(x, width, rows, sizeof(float))) return 0;
    if (format == DS4_V41_BF16) {
        const uint64_t count = (uint64_t)width * rows;
        if ((count + 255u) / 256u > UINT32_MAX) return 0;
        v41_bf16_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>((float *)x->ptr, count);
    } else {
        const uint64_t blocks = (uint64_t)(width / block) * rows;
        if (blocks > UINT32_MAX) return 0;
        v41_quantize_kernel<<<(unsigned)blocks, 32>>>((float *)x->ptr, format);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 activation quantization");
}

struct v41_rope_args {
    uint32_t width, heads, start, stride, inverse;
    float frequencies[32];
};

__global__ static void v41_rope_kernel(float *x, v41_rope_args args) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x / args.heads;
    const float theta = v41_mul((float)(args.start + row * args.stride), args.frequencies[lane]);
    const float c = __ocml_cos_f32(theta), s = args.inverse ? -__ocml_sin_f32(theta) : __ocml_sin_f32(theta);
    const uint64_t i = (uint64_t)blockIdx.x * args.width + args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = v41_bf16(v41_sub(v41_mul(re, c), v41_mul(im, s)));
    x[i + 1u] = v41_bf16(v41_add(v41_mul(re, s), v41_mul(im, c)));
}

static float v41_rope_frequencies[2][32];
static pthread_once_t v41_rope_once = PTHREAD_ONCE_INIT;

/* Keep the reference's host pow/reciprocal and YaRN operation order; a frequency ULP grows into a phase error at long positions. */
#ifndef __HIP_DEVICE_COMPILE__
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
#endif
static void v41_init_rope_frequencies(void) {
    for (int kind = 0; kind < 2; kind++) {
        const float base = kind ? 160000.0f : 10000.0f;
        const float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) / (2.0 * log(base)));
        const float high = (float)ceil(64.0 * log(65536.0 / (2.0 * M_PI)) / (2.0 * log(base)));
        for (int i = 0; i < 32; i++) {
            const float denominator = powf(base, (float)i / 32.0f);
            float f = 1.0f / denominator;
            if (kind) {
                const float ramp = fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                const float smooth = 1.0f - ramp;
                const float interpolate = (f / 16.0f) * (1.0f - smooth);
                const float extrapolate = f * smooth;
                f = interpolate + extrapolate;
            }
            v41_rope_frequencies[kind][i] = f;
        }
    }
}
#ifndef __HIP_DEVICE_COMPILE__
#pragma float_control(pop)
#endif

extern "C" int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                         uint32_t rows, uint32_t start, uint32_t stride,
                                         bool compressed, bool inverse) {
    uint64_t elems = 0;
    if (width < 64u || !heads || !rows || rows > 1048576u || !stride ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > UINT32_MAX ||
        !cuda_u64_mul3_checked(width, heads, rows, &elems) || !cuda_tensor_has_f32(x, elems)) return 0;
    if (pthread_once(&v41_rope_once, v41_init_rope_frequencies)) return 0;
    v41_rope_args args = {width, heads, start, stride, inverse, {0}};
    memcpy(args.frequencies, v41_rope_frequencies[compressed ? 1 : 0], sizeof(args.frequencies));
    v41_rope_kernel<<<heads * rows, 32>>>((float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "V4.1 unit-magnitude RoPE");
}

extern "C" int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                  uint32_t rows, uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_rope_stride(x, width, heads, rows, start, 1, compressed, inverse);
}

__global__ static void v41_engram_kernel(float *residual, const float *kv, const float *qw,
                                        const float *kw, const uint8_t *mask, uint32_t width, float eps) {
    const uint32_t token = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (mask && !mask[token]) return;
    const uint64_t offset = ((uint64_t)token * 4u + head) * width;
    const uint64_t key = ((uint64_t)token * 5u + head) * width;
    const uint64_t value = ((uint64_t)token * 5u + 4u) * width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint32_t i = lane; i < width; i += 32u) {
        const float h = residual[offset + i], k = v41_bf16(kv[key + i]);
        const uint64_t wi = (uint64_t)head * width + i;
        h2 = v41_add(h2, v41_mul(h, h));
        k2 = v41_add(k2, v41_mul(k, k));
        dot = v41_add(dot, v41_mul(v41_mul(h, v41_mul(qw[wi], kw[wi])), k));
    }
    h2 = v41_sum32(h2);
    k2 = v41_sum32(k2);
    dot = v41_mul(v41_sum32(dot), __ocml_rsqrt_f32(v41_add(v41_div(h2, (float)width), eps)));
    dot = v41_mul(dot, __ocml_rsqrt_f32(v41_add(v41_div(k2, (float)width), eps)));
    dot = v41_mul(dot, __ocml_rsqrt_f32((float)width));
    const float gate = v41_div(1.0f, v41_add(1.0f,
        __ocml_exp_f32(-copysignf(__ocml_sqrt_f32(fmaxf(fabsf(dot), 1.0e-6f)), dot))));
    for (uint32_t i = lane; i < width; i += 32u)
        residual[offset + i] = v41_bf16(v41_add(residual[offset + i], v41_mul(gate, v41_bf16(kv[value + i]))));
}

extern "C" int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual, const ds4_gpu_tensor *kv,
                                        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
                                        const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows, float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || !isfinite(eps) || eps <= 0 || count > UINT64_MAX / 5u ||
        !cuda_tensor_has_f32(residual, count * 4u) || !cuda_tensor_has_f32(kv, count * 5u) ||
        !cuda_tensor_has_f32(q_weight, (uint64_t)width * 4u) ||
        !cuda_tensor_has_f32(k_weight, (uint64_t)width * 4u) ||
        (mask && !cuda_tensor_has_bytes(mask, rows))) return 0;
    v41_engram_kernel<<<dim3(rows, 4), 32>>>((float *)residual->ptr, (const float *)kv->ptr,
        (const float *)q_weight->ptr, (const float *)k_weight->ptr,
        mask ? (const uint8_t *)mask->ptr : NULL, width, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 Engram gate");
}

__global__ static void v41_pool_kernel(float *out, const float *kv, const float *scores,
                                      const float *previous_kv, const float *previous_scores,
                                      uint32_t width, uint32_t tail) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const int64_t a = (int64_t)blockIdx.y * 2 - tail;
    const uint64_t b = (uint64_t)(a + 1) * width + col;
    const float ka = a < 0 ? previous_kv[col] : kv[(uint64_t)a * width + col];
    const float sa = a < 0 ? previous_scores[col] : scores[(uint64_t)a * width + col];
    const float sb = scores[b], peak = fmaxf(sa, sb);
    const float ea = __ocml_exp_f32(v41_sub(sa, peak)), eb = __ocml_exp_f32(v41_sub(sb, peak));
    out[(uint64_t)blockIdx.y * width + col] = v41_bf16(v41_div(
        v41_add(v41_mul(ka, ea), v41_mul(kv[b], eb)), v41_add(ea, eb)));
}

extern "C" int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv,
                                   const ds4_gpu_tensor *scores, ds4_gpu_tensor *previous_kv,
                                   ds4_gpu_tensor *previous_scores, uint32_t width, uint32_t rows, uint32_t start) {
    const uint64_t count = (uint64_t)width * rows;
    const uint32_t pairs = (uint32_t)(((uint64_t)rows + (start & 1u)) / 2u);
    if (!width || !rows || rows > UINT32_MAX - start ||
        !cuda_tensor_has_f32(kv, count) || !cuda_tensor_has_f32(scores, count) ||
        !cuda_tensor_has_f32(previous_kv, width) || !cuda_tensor_has_f32(previous_scores, width) ||
        (pairs && !cuda_tensor_has_f32(out, (uint64_t)width * pairs))) return 0;
    if (pairs) {
        v41_pool_kernel<<<dim3((unsigned)(((uint64_t)width + 255u) / 256u), pairs), 256>>>(
            (float *)out->ptr, (const float *)kv->ptr, (const float *)scores->ptr,
            (const float *)previous_kv->ptr, (const float *)previous_scores->ptr, width, start & 1u);
        if (!cuda_ok(cudaGetLastError(), "V4.1 KV pair pooling")) return 0;
    }
    /* Retain the last even input even at even frontiers, so snapshots are independent of chunk partitioning. */
    const uint32_t last_even = (start + rows - 1u) & ~1u;
    if (last_even >= start) {
        const uint64_t bytes = (uint64_t)width * sizeof(float), offset = (last_even - start) * bytes;
        if (!ds4_gpu_tensor_copy(previous_kv, 0, kv, offset, bytes) ||
            !ds4_gpu_tensor_copy(previous_scores, 0, scores, offset, bytes)) return 0;
    }
    return 1;
}

template <bool FILTER>
__global__ static void v41_candidates_kernel(float *out, const float *scores, const float *mask,
                                            uint32_t width, uint32_t start, uint32_t ratio) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint32_t visible = min(width, (start + row + 1u) / ratio);
    if (FILTER) {
        if (col >= width) return;
        const uint64_t i = (uint64_t)row * width + col;
        out[i] = col < visible && mask[(uint64_t)row * blocks + col / 8u] == 0.0f ? scores[i] : -INFINITY;
    } else {
        if (col >= blocks) return;
        float best = -INFINITY;
        for (uint32_t i = col * 8u; i < min(visible, (col + 1u) * 8u); i++)
            best = fmaxf(best, scores[(uint64_t)row * width + i]);
        if (visible && col == (visible - 1u) / 8u) best = INFINITY;
        out[(uint64_t)row * blocks + col] = best;
    }
}

static int v41_candidates(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
                           const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows,
                           uint32_t start, uint32_t ratio) {
    if (!width || width > UINT32_MAX - 7u || !rows || !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u, out_width = mask ? width : blocks;
    if (!cuda_tensor_has_elems2(scores, width, rows, 4u) ||
        !cuda_tensor_has_elems2(out, out_width, rows, 4u) ||
        (mask && !cuda_tensor_has_elems2(mask, blocks, rows, 4u))) return 0;
    const dim3 grid((unsigned)(((uint64_t)out_width + 255u) / 256u), rows);
    if (mask) v41_candidates_kernel<true><<<grid, 256>>>((float *)out->ptr, (const float *)scores->ptr,
        (const float *)mask->ptr, width, start, ratio);
    else v41_candidates_kernel<false><<<grid, 256>>>((float *)out->ptr, (const float *)scores->ptr,
        NULL, width, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 candidate selection");
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *blocks, const ds4_gpu_tensor *scores,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return v41_candidates(blocks, scores, NULL, width, rows, start, ratio);
}

extern "C" int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores, const ds4_gpu_tensor *block_mask,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return block_mask && v41_candidates(scores, scores, block_mask, width, rows, start, ratio);
}

__global__ static void v41_carry_bf16_kernel(uint16_t *packed, float *plain, uint32_t width,
                                           uint32_t words, bool pack) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const uint64_t p = (uint64_t)blockIdx.y * words * 2u + col;
    const uint64_t f = (uint64_t)blockIdx.y * width + col;
    if (pack) packed[p] = (uint16_t)(__float_as_uint(plain[f]) >> 16u);
    else plain[f] = __uint_as_float((uint32_t)packed[p] << 16u);
}

__global__ static void v41_carry_mask_kernel(uint32_t *packed, float *plain, uint32_t width,
                                           uint32_t words, bool pack) {
    const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= words) return;
    const uint64_t p = (uint64_t)blockIdx.y * words + word;
    uint32_t bits = pack ? 0u : packed[p];
    for (uint32_t bit = 0; bit < 32u && (uint64_t)word * 32u + bit < width; bit++) {
        const uint64_t f = (uint64_t)blockIdx.y * width + (uint64_t)word * 32u + bit;
        if (pack) bits |= plain[f] == 0.0f ? 1u << bit : 0u;
        else plain[f] = bits & (1u << bit) ? 0.0f : -INFINITY;
    }
    if (pack) packed[p] = bits;
}

extern "C" int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t row_offset,
                                        ds4_gpu_tensor *plain, uint32_t width, uint32_t rows,
                                        uint32_t format, bool pack) {
    if (!width || !rows || rows > UINT32_MAX - row_offset ||
        format > DS4_V41_CARRY_MASK || packed == plain) return 0;
    const uint32_t words = format == DS4_V41_CARRY_BF16 ?
        (uint32_t)(((uint64_t)width + 1u) / 2u) : (uint32_t)(((uint64_t)width + 31u) / 32u);
    if (!cuda_tensor_has_elems2(packed, (uint64_t)row_offset + rows, words, 4u) ||
        !cuda_tensor_has_elems2(plain, rows, width, 4u)) return 0;
    uint32_t *p = (uint32_t *)packed->ptr + (uint64_t)row_offset * words;
    if (format == DS4_V41_CARRY_BF16)
        v41_carry_bf16_kernel<<<dim3((unsigned)(((uint64_t)width + 255u) / 256u), rows), 256>>>(
            (uint16_t *)p, (float *)plain->ptr, width, words, pack);
    else v41_carry_mask_kernel<<<dim3((unsigned)(((uint64_t)words + 255u) / 256u), rows), 256>>>(
        p, (float *)plain->ptr, width, words, pack);
    return cuda_ok(cudaGetLastError(), "V4.1 compact prefill carry");
}

__global__ static void v41_gather_kernel(float *out, const float *source, const int32_t *ids,
                                        uint32_t source_rows) {
    const uint32_t row = blockIdx.x, col = threadIdx.x;
    const int32_t id = ids[row];
    /* Graph IDs come from top-k. Keep malformed IDs from turning a validation failure into an out-of-bounds load. */
    if ((uint32_t)id >= source_rows) {
        out[(uint64_t)row * 512u + col] = NAN;
        out[(uint64_t)row * 512u + col + 256u] = NAN;
        return;
    }
    out[(uint64_t)row * 512u + col] = source[(uint64_t)id * 512u + col];
    out[(uint64_t)row * 512u + col + 256u] = source[(uint64_t)id * 512u + col + 256u];
}

extern "C" int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                                       const ds4_gpu_tensor *ids, uint32_t source_rows, uint32_t selected_rows) {
    if (!source_rows || !selected_rows || selected_rows > 512u || selected_rows > source_rows ||
        !cuda_tensor_has_elems2(source, source_rows, 512u, 4u) ||
        !cuda_tensor_has_elems2(out, selected_rows, 512u, 4u) || !cuda_tensor_has_f32(ids, selected_rows)) return 0;
    v41_gather_kernel<<<selected_rows, 256>>>((float *)out->ptr, (const float *)source->ptr,
        (const int32_t *)ids->ptr, source_rows);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse KV gather");
}

/* One wave per score: retain the scalar F32 reduction tree and scale boundary. */
__global__ static void v41_indexer_scalar_warp8_kernel(
        float *out, const float *q, const float *w, const float *k, uint32_t width, uint32_t start,
        uint32_t ratio) {
    uint32_t key = blockIdx.x * 8u + threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.y;
    if (key >= width) return;
    if (key >= (start + row + 1) / ratio) {
        if (!lane) out[(uint64_t) row * width + key] = -INFINITY;
        return;
    }
    const float *p = k + (uint64_t) key * 128 + lane;
    float k0 = p[0], k1 = p[32], k2 = p[64], k3 = p[96], total = 0;
    for (uint32_t h = 0; h < 32; h++) {
        const float *x = q + ((uint64_t) row * 32 + h) * 128 + lane;
        float a = v41_mul(x[0], k0), b = v41_mul(x[32], k1), c = v41_mul(x[64], k2), d = v41_mul(x[96], k3);
        float dot = v41_add(v41_add(a, c), v41_add(b, d));
        for (int s = 16; s; s >>= 1) dot = v41_add(dot, __shfl_down(dot, s, 32));
        total = v41_add(total, v41_mul(fmaxf(v41_mul(dot, 1.f / 64.f), 0.f), w[row * 32 + h]));
    }
    if (!lane) out[(uint64_t) row * width + key] = total;
}
extern "C" int ds4_gpu_dsv41_indexer_scores_one(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *keys, uint32_t source_rows) {
    if (!ds4_rocm_is_gfx1151())
        return ds4_gpu_glm_indexer_score_one_tensor(scores, q, weights,
            keys, source_rows, 32u, 128u, 1.0f / 64.0f, false);
    if (!source_rows || source_rows > INT32_MAX ||
        !cuda_tensor_has_f32(scores, source_rows) ||
        !cuda_tensor_has_f32(q, 32u * 128u) ||
        !cuda_tensor_has_f32(weights, 32u) ||
        !cuda_tensor_has_elems2(keys, source_rows, 128u, 4u)) return 0;
    v41_indexer_scalar_warp8_kernel<<<(source_rows + 7u) / 8u, 256>>>(
        (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
        (const float *)keys->ptr, source_rows, source_rows - 1u, 1u);
    return cuda_ok(cudaGetLastError(), "V4.1 scalar warp indexer");
}

__global__ static void v41_indexer_kernel(float *scores, const float *q, const float *weights,
                                         const float *keys, uint32_t width, uint32_t start, uint32_t ratio) {
    const uint32_t key = blockIdx.x, token = blockIdx.y, lane = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
    if (key >= (start + token + 1u) / ratio) {
        if (!threadIdx.x) scores[(uint64_t)token * width + key] = -INFINITY;
        return;
    }
    __shared__ float head_values[4];
    float total = 0.0f;
    for (uint32_t head0 = 0; head0 < 32u; head0 += 4u) {
        const uint32_t head = head0 + wave;
        const float *query = q + ((uint64_t)token * 32u + head) * 128u;
        const float *kv = keys + (uint64_t)key * 128u;
        float dot = 0.0f;
        for (uint32_t col = lane; col < 128u; col += 32u) dot = v41_add(dot, v41_mul(query[col], kv[col]));
        dot = v41_sum32(dot);
        if (!lane) head_values[wave] = v41_mul(fmaxf(v41_mul(dot, 1.0f / 64.0f), 0.0f), weights[(uint64_t)token * 32u + head]);
        __syncthreads();
        if (!threadIdx.x) for (uint32_t h = 0; h < 4u; h++) total = v41_add(total, head_values[h]);
        __syncthreads();
    }
    if (!threadIdx.x) scores[(uint64_t)token * width + key] = total;
}

__global__ static void v41_indexer_batch_warp8_kernel(
        float *out, const float *q, const float *w, const float *k, uint32_t width, uint32_t start,
        uint32_t ratio) {
    uint32_t key = blockIdx.x * 8u + threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.y;
    if (key >= width) return;
    if (key >= (start + row + 1) / ratio) {
        if (!lane) out[(uint64_t) row * width + key] = -INFINITY;
        return;
    }
    const float *p = k + (uint64_t) key * 128 + lane;
    float k0 = p[0], k1 = p[32], k2 = p[64], k3 = p[96], total = 0;
    for (uint32_t h = 0; h < 32; h++) {
        const float *x = q + ((uint64_t) row * 32 + h) * 128 + lane;
        float a = v41_mul(x[0], k0), b = v41_mul(x[32], k1), c = v41_mul(x[64], k2), d = v41_mul(x[96], k3);
        float dot = v41_add(v41_add(v41_add(v41_add(0.0f, a), b), c), d);
        for (int s = 16; s; s >>= 1) dot = v41_add(dot, __shfl_down(dot, s, 32));
        float weighted;
        // OCML wrappers alone do not prevent backend contraction across this boundary.
        // The control stores this product in LDS before adding it to the head total.
        asm volatile("v_mul_f32 %0, %1, %2" : "=v"(weighted)
            : "v"(fmaxf(v41_mul(dot, 1.f / 64.f), 0.f)), "v"(w[row * 32 + h]));
        total = v41_add(total, weighted);
    }
    if (!lane) out[(uint64_t) row * width + key] = total;
}
__global__ static void v41_indexer_head_queries_kernel(float *qh,const float *q,unsigned rows){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*4096u)return;
 unsigned d=i%128,h=(i/128)%32,t=i/4096;qh[(h*rows+t)*128+d]=q[i];
}
__global__ static void v41_indexer_reduce_heads_kernel(float *out,const float *dots,const float *w,unsigned width,unsigned rows,unsigned h0,unsigned heads,unsigned start,unsigned ratio){
 unsigned k=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;if(k>=width)return;
 size_t i=(size_t)row*width+k;
 if(k>=(start+row+1)/ratio){out[i]=-INFINITY;return;}
 float v=h0?out[i]:0.f;
 for(unsigned h=0;h<heads;h++)v=v41_add(v,v41_mul(fmaxf(v41_mul(dots[((size_t)h*rows+row)*width+k],1.f/64),0.f),w[row*32+h0+h]));
 out[i]=v;
}
/* Keep the query tile wide for GEMM; partition heads to bound scratch instead
 * of shrinking N to eight columns at long context. All operands remain F32. */
static int v41_indexer_head_gemm(float *out, const float *q, const float *weights,
        const float *keys, uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    const uint64_t per_head = (uint64_t)rows * width * sizeof(float);
    const uint32_t heads = (uint32_t)min(UINT64_C(32), (UINT64_C(64) << 20) / per_head);
    const uint64_t q_bytes = (uint64_t)rows * 32u * 128u * sizeof(float);
    char *scratch = (char *)cuda_tmp_alloc(q_bytes + heads * per_head, "V4.1 indexer head GEMM");
    if (!scratch) return 0;
    float *qh = (float *)scratch, *dots = (float *)(scratch + q_bytes);
    v41_indexer_head_queries_kernel<<<(rows * 4096u + 255u) / 256u, 256u>>>(qh, q, rows);
    if (!cuda_ok(cudaGetLastError(), "V4.1 indexer query preparation")) return 0;
    const float one = 1.0f, zero = 0.0f;
    for (uint32_t h0 = 0; h0 < 32u; h0 += heads) {
        const uint32_t count = min(heads, 32u - h0);
        if (!cublas_ok(cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                width, rows, 128, &one, keys, 128, 0,
                qh + (uint64_t)h0 * rows * 128u, 128, (long long)rows * 128,
                &zero, dots, width, (long long)rows * width, count), "V4.1 indexer head GEMM")) return 0;
        v41_indexer_reduce_heads_kernel<<<dim3((width + 255u) / 256u, rows), 256u>>>(
            out, dots, weights, width, rows, h0, count, start, ratio);
        if (!cuda_ok(cudaGetLastError(), "V4.1 indexer head reduction")) return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
                                                  const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                                  uint32_t source_rows, uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !source_rows || !rows || rows > UINT32_MAX - start ||
        (start + rows) / ratio > source_rows || source_rows > INT32_MAX || rows > INT32_MAX ||
        !cuda_tensor_has_elems2(scores, source_rows, rows, 4u) ||
        !cuda_tensor_has_elems2(q, rows, 32u * 128u, 4u) ||
        !cuda_tensor_has_elems2(keys, source_rows, 128u, 4u) ||
        !cuda_tensor_has_elems2(weights, rows, 32u, 4u)) return 0;
    if (ds4_rocm_is_gfx1151()) {
        /* Small query tiles favor wave scoring. GEMM uses a bounded head tile
         * only for the wide, nearly fully visible batches qualified here. */
        if (!g_quality_mode && g_cublas_ready && rows >= 31u && rows <= 32u &&
            source_rows >= 16384u && source_rows <= 65536u &&
            (start + 1u) / ratio >= source_rows - source_rows / 8u)
            return v41_indexer_head_gemm((float *)scores->ptr, (const float *)q->ptr,
                (const float *)weights->ptr, (const float *)keys->ptr, source_rows, rows, start, ratio);
        v41_indexer_batch_warp8_kernel<<<dim3((source_rows + 7u) / 8u, rows), 256u>>>(
            (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
            (const float *)keys->ptr, source_rows, start, ratio);
    } else {
        v41_indexer_kernel<<<dim3(source_rows, rows), 128>>>((float *)scores->ptr, (const float *)q->ptr,
            (const float *)weights->ptr, (const float *)keys->ptr, source_rows, start, ratio);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 causal FP4 index scores");
}

template <uint32_t SORT_N>
__global__ static void v41_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t score_stride, uint32_t start, uint32_t ratio,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const uint32_t n_comp = (start + t + 1u) / ratio;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * score_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void v41_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t score_stride, uint32_t start, uint32_t ratio,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const uint32_t n_comp = (start + t + 1u) / ratio;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * score_stride;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void v41_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t score_stride, uint32_t start, uint32_t ratio,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const uint32_t n_comp = (start + t + 1u) / ratio;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * score_stride;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

/* Preserve each row's original sort network, including score/index tie order.
 * Group rows with equal chunk counts; only launches and physical strides change. */
static int v41_indexer_topk_causal_batch(uint32_t *selected, const float *scores,
        uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    const uint32_t max_chunks = ((start + rows) / ratio + 4095u) / 4096u;
    const uint32_t scratch_sets = max_chunks + (max_chunks > 8u ? (max_chunks + 7u) / 8u : 0u);
    uint32_t *scratch = (uint32_t *)cuda_tmp_alloc((uint64_t)rows * scratch_sets * 512u * sizeof(uint32_t),
                                                  "V4.1 causal top-k rows");
    if (!scratch) return 0;
    for (uint32_t row = 0; row < rows;) {
        const uint32_t chunks = ((start + row + 1u) / ratio + 4095u) / 4096u;
        uint32_t end = row + 1u;
        while (end < rows && ((start + end + 1u) / ratio + 4095u) / 4096u == chunks) ++end;
        const uint32_t count = end - row, begin = start + row;
        const float *input = scores + (uint64_t)row * width;
        uint32_t *cur = scratch, sets = chunks, stride = chunks * 512u;
        v41_topk_chunk_pow2_kernel<4096><<<dim3(count, chunks), 1024>>>(cur, input,
            width, begin, ratio, count, 512u, stride);
        if (!cuda_ok(cudaGetLastError(), "V4.1 causal top-k chunks")) return 0;
        while (sets > 8u) {
            const uint32_t next_sets = (sets + 7u) / 8u, next_stride = next_sets * 512u;
            uint32_t *next = cur + (uint64_t)count * stride;
            v41_topk_tree_merge_pow2_kernel<4096><<<dim3(count, next_sets), 1024>>>(next,
                cur, input, width, begin, ratio, count, 512u, sets, 8u, stride, next_stride);
            if (!cuda_ok(cudaGetLastError(), "V4.1 causal top-k tree")) return 0;
            cur = next; sets = next_sets; stride = next_stride;
        }
        v41_topk_merge_pow2_kernel<4096><<<count, 1024>>>(selected + (uint64_t)row * 512u,
            cur, input, width, begin, ratio, count, 512u, sets * 512u, stride);
        if (!cuda_ok(cudaGetLastError(), "V4.1 causal top-k final")) return 0;
        row = end;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                               uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !rows || rows > UINT32_MAX - start ||
        width > INT32_MAX || rows > INT32_MAX || (start + rows) / ratio > width ||
        !cuda_tensor_has_elems2(scores, width, rows, 4u) ||
        !cuda_tensor_has_elems2(selected, 512u, rows, 4u)) return 0;
    if (ds4_rocm_is_gfx1151() && rows >= 2u && rows <= 32u && width <= 65536u &&
        (start + 1u) / ratio > 8192u)
        return v41_indexer_topk_causal_batch((uint32_t *)selected->ptr, (const float *)scores->ptr,
                                             width, rows, start, ratio);
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t visible = (start + row + 1u) / ratio;
        if (!visible) continue;
        const uint32_t top = visible < 512u ? visible : 512u;
        ds4_gpu_tensor in = {(float *)scores->ptr + (uint64_t)row * width, (uint64_t)visible * 4u, 0};
        ds4_gpu_tensor out = {(uint32_t *)selected->ptr + (uint64_t)row * 512u, 512u * 4u, 0};
        if (visible > 1u && visible < 512u && ds4_rocm_is_gfx1151()) {
            /* Preserve the scalar order and untouched tail slots, while
             * avoiding its serial insertion sort for short prefill rows. */
            indexer_topk_1024_kernel<<<1u, 1024u>>>((uint32_t *)out.ptr,
                (const float *)in.ptr, visible, 1u, top);
            if (!cuda_ok(cudaGetLastError(), "V4.1 short causal top-k")) return 0;
        } else if (!ds4_gpu_indexer_topk_tensor(&out, &in, visible, 1u, top)) return 0;
    }
    return 1;
}

/* Metal tensor packing is an optional acceleration. The graph selects the complete F32 baseline above when this capability is absent. */
extern "C" int ds4_gpu_dsv41_tensor_ops_available(void) { return 0; }

extern "C" uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows) {
    const uint64_t tiles = ((uint64_t)source_rows + 63u) / 64u;
    const uint64_t flags = (((uint64_t)rows + tiles) * 4u + 255u) & ~UINT64_C(255);
    return flags + (uint64_t)rows * 32u * 128u * 2u + tiles * 64u * 128u * 2u;
}

extern "C" int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *packed, const ds4_gpu_tensor *q,
                                         const ds4_gpu_tensor *keys, uint32_t source_rows, uint32_t rows) {
    (void)packed; (void)q; (void)keys; (void)source_rows; (void)rows;
    return 0;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
                                                   const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                                   const ds4_gpu_tensor *packed, uint32_t source_rows,
                                                   uint32_t rows, uint32_t start, uint32_t ratio,
                                                   uint32_t packed_rows, uint32_t offset) {
    (void)scores; (void)q; (void)weights; (void)keys; (void)packed;
    (void)source_rows; (void)rows; (void)start; (void)ratio; (void)packed_rows; (void)offset;
    return 0;
}

/* Each wave still owns one output row. Lane l sums the same contiguous
 * ceil(K/32) inputs in ascending order. Tokens have separate accumulators;
 * only the exactly decoded F16 weight is reused. Lane zero still adds the
 * 32 partials in order. No activation conversion or split-K reduction. */
template<unsigned TT, unsigned WAVES>
__global__ static void f16_ordered_token_reuse(
        float *out, const __half *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * WAVES + wave;
    const uint64_t token = (uint64_t)blockIdx.y * TT;
    __shared__ float partial[WAVES][TT][32];
    float sum[TT] = {};
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)lane * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    if (row < out_dim) {
        const __half *wr = w + row * in_dim;
        for (uint64_t i = k0; i < k1; i++) {
            const float weight = __half2float(wr[i]);
#pragma unroll
            for (unsigned t = 0; t < TT; t++) {
                if (token + t < n_tok)
                    sum[t] += weight * x[(token + t) * in_dim + i];
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < TT; t++) partial[wave][t][lane] = sum[t];
    __syncthreads();
    if (row < out_dim && lane == 0u) {
#pragma unroll
        for (unsigned t = 0; t < TT; t++) {
            if (token + t < n_tok) {
                float total = 0.0f;
                for (uint32_t i = 0; i < 32u; i++) total += partial[wave][t][i];
                out[(token + t) * out_dim + row] = total;
            }
        }
    }
}

/* Batch the V4.1 F32-input F16 projections without changing their arithmetic.
 * Each wave still owns one output row; every token keeps the production
 * shared-X kernel's strided FMA sequence and warp reduction. The only change
 * is reusing each decoded F16 weight across TT independent token accumulators. */
template<unsigned TT, unsigned WAVES, unsigned SEG>
__global__ static void v41_f16_sharedx_token_reuse(
        float *out, const __half *weight, const float *input,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    extern __shared__ float tile[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t row = blockIdx.x * WAVES + wave;
    const uint32_t token0 = blockIdx.y * TT;
    float acc[TT] = {};
    const __half *wr = row < out_dim ? weight + (uint64_t)row * in_dim : NULL;

    for (uint32_t base = 0; base < in_dim; base += SEG) {
        const uint32_t remaining = in_dim - base;
        const uint32_t valid = remaining < SEG ? remaining : SEG;
        for (uint32_t j = tid; j < TT * SEG; j += WAVES * 32u) {
            const uint32_t token = j / SEG;
            const uint32_t offset = j - token * SEG;
            tile[j] = token0 + token < rows && offset < valid ?
                input[(uint64_t)(token0 + token) * in_dim + base + offset] : 0.0f;
        }
        __syncthreads();
        if (row < out_dim) {
            for (uint32_t offset = lane; offset < valid; offset += 32u) {
                const float w = __half2float(wr[base + offset]);
#pragma unroll
                for (uint32_t token = 0; token < TT; token++) {
                    if (token0 + token < rows)
                        acc[token] += w * tile[token * SEG + offset];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t token = 0; token < TT; token++) {
        const float total = warp_sum_f32(acc[token]);
        if (row < out_dim && token0 + token < rows && lane == 0u)
            out[(uint64_t)(token0 + token) * out_dim + row] = total;
    }
}

template<unsigned TT, unsigned SEG>
static void v41_launch_f16_sharedx_token_reuse(
        float *out, const __half *weight, const float *input,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    const uint32_t waves = 16u;
    const dim3 grid((out_dim + waves - 1u) / waves,
                    (rows + TT - 1u) / TT, 1u);
    v41_f16_sharedx_token_reuse<TT, 16u, SEG>
        <<<grid, waves * 32u, TT * SEG * sizeof(float)>>>(
            out, weight, input, in_dim, out_dim, rows);
}

__global__ static void v41_hc_half_to_float(float *out, const __half *weight, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __half2float(weight[i]);
}
static hipError_t v41_hc_widen(float *out, const uint16_t *weight, uint64_t count) {
    v41_hc_half_to_float<<<(count + 255u) / 256u, 256u>>>(out, (const __half *)weight, count);
    return hipGetLastError();
}
static bool v41_hc_disjoint(const void *a, uint64_t na, const void *b, uint64_t nb) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return na <= UINTPTR_MAX - pa && nb <= UINTPTR_MAX - pb &&
        (pa + na <= pb || pb + nb <= pa);
}
#include "ds4_rocm_hc_sgemm.cuh"

extern "C" void ds4_gpu_dsv41_hc_plan_free(ds4_gpu_dsv41_hc_plan *plan) {
    v41_hc_plan_destroy(plan);
}
extern "C" int ds4_gpu_dsv41_hc_project(ds4_gpu_dsv41_hc_plan **plan,
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t rows, const ds4_gpu_tensor *input,
        ds4_gpu_tensor *full_heads_scratch) {
    if (rows != 2048u || !ds4_rocm_is_gfx1151() ||
        g_quality_mode || cuda_runtime_config()->graph_dump || !g_rocblas_ready ||
        g_rocblas_f16_solution_set != DS4_ROCBLAS_F16_SOLUTIONS_5_6_8D1AE90E) return 0;
    const uint64_t weight_bytes = UINT64_C(20480) * 24u * 2u;
    const uint64_t in_bytes = UINT64_C(20480) * 2048u * 4u;
    const uint64_t out_bytes = UINT64_C(24) * 2048u * 4u;
    if (!plan || !model_map || !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_bytes(input, in_bytes) || !cuda_tensor_has_bytes(out, out_bytes)) return -1;
    if (!cuda_tensor_has_bytes(full_heads_scratch, v41_hc_scratch_bytes)) return 0;
    if ((uintptr_t)full_heads_scratch->ptr % 256u ||
        !v41_hc_disjoint(out->ptr, out_bytes, input->ptr, in_bytes) ||
        !v41_hc_disjoint(out->ptr, out_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes) ||
        !v41_hc_disjoint(input->ptr, in_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes)) return -1;
    const uint16_t *weight = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "V4.1 HC F16");
    if (!weight ||
        !v41_hc_disjoint(weight, weight_bytes, out->ptr, out_bytes) ||
        !v41_hc_disjoint(weight, weight_bytes, input->ptr, in_bytes) ||
        !v41_hc_disjoint(weight, weight_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes)) return -1;
    return v41_hc_plan_run(plan, (float *)out->ptr, weight, (const float *)input->ptr, full_heads_scratch->ptr);
}

/* Default Engram lane l consumes l+32*i, i=0..191, then the existing
 * shuffle16/8/4/2/1. Each token retains an independent accumulation chain.
 * Only F32 input reuse across the eight output waves changes. */
template<unsigned TT,unsigned WAVES=8,unsigned SEG=256>
__global__ static void engram_lds_token_reuse(float *out,const __half *w,const float *x) {
    constexpr unsigned K=6144,N=25600;
    static_assert(K % SEG == 0, "Engram LDS segment must divide K");
    const unsigned tid=threadIdx.x,lane=tid&31u,wave=tid>>5u;
    const unsigned row=blockIdx.x*WAVES+wave,token=blockIdx.y*TT;
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
        for(unsigned group=0;group<SEG;group+=256u) {
            const unsigned k=group+lane;
#pragma unroll
            for(unsigned t=0;t<TT;t++) {
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k])*tile[t][k];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+32u])*tile[t][k+32u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+64u])*tile[t][k+64u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+96u])*tile[t][k+96u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+128u])*tile[t][k+128u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+160u])*tile[t][k+160u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+192u])*tile[t][k+192u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+224u])*tile[t][k+224u];
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

#include "ds4_rocm_engram_e4m3.cuh"

/* Engram table scales can exceed F16 range. Only use the matrix path when
 * every activation survives conversion exactly; retain F32 input otherwise. */
__global__ static void v41_engram_pack_checked_kernel(__half *out,
        const float *input, uint32_t count, uint32_t *loss) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = input[i];
        const __half packed = __float2half_rn(value);
        out[i] = packed;
        if (__half2float(packed) != value) atomicOr(loss, 1u);
    }
}

/* Return zero when the installed library cannot provide the qualified plan.
 * The existing plan cache owns descriptor cleanup across model changes. */
static cuda_hipblaslt_gemm_plan *v41_engram_lt_plan(uint32_t rows) {
    if (!g_hipblaslt_ready) return NULL;
    int version = 0;
    char revision[128] = {0};
    if (hipblasLtGetVersion(g_hipblaslt, &version) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtGetGitRevision(g_hipblaslt, revision) != HIPBLAS_STATUS_SUCCESS ||
        version != 100401 || strcmp(revision, "8d1ae90e") != 0) return NULL;
    return hipblaslt_gemm_plan_get(25600u, rows, 6144u,
        "V4.1 Engram F16/F32", HIP_R_32F, 2539);
}

/* Tiny HC projection: prefetch aligned pairs without changing the scalar
 * 640-element lane accumulation or the ordered sum of 32 lane partials. */
__global__ static void v41_hc_ordered_prefetch_kernel(float *out,
        const __half *weights, const float *input) {
    constexpr unsigned width = 20480u, outputs = 24u, chunk = 640u;
    const unsigned lane = threadIdx.x, row = blockIdx.x, token = blockIdx.y;
    const __half *w = weights + (uint64_t)row * width;
    const float *x = input + (uint64_t)token * width;
    __shared__ float partial[32];
    float sum = 0.0f;
    for (unsigned base = lane * chunk; base < (lane + 1u) * chunk; base += 16u) {
        float ww[16], xx[16];
#pragma unroll
        for (unsigned u = 0; u < 16u; u += 2u) {
            const float2 wp = __half22float2(*reinterpret_cast<const __half2 *>(w + base + u));
            const float2 xp = *reinterpret_cast<const float2 *>(x + base + u);
            ww[u] = wp.x; ww[u + 1u] = wp.y;
            xx[u] = xp.x; xx[u + 1u] = xp.y;
        }
#pragma unroll
        for (unsigned u = 0; u < 16u; u++) sum = __fmaf_rn(ww[u], xx[u], sum);
    }
    partial[lane] = sum;
    __syncthreads();
    if (lane == 0u) {
        float total = 0.0f;
        for (unsigned i = 0; i < 32u; i++) total += partial[i];
        out[(uint64_t)token * outputs + row] = total;
    }
}

extern "C" int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t width, uint32_t outputs,
                                             uint32_t rows, const ds4_gpu_tensor *in) {
    uint64_t weight_bytes = 0;
    if (!width || !outputs || !rows || rows > 8192u || !model_map ||
        !cuda_u64_mul3_checked(width, outputs, sizeof(uint16_t), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) || !cuda_tensor_has_elems2(out, outputs, rows, 4u)) return 0;
    if (width == 6144u && outputs == 25600u && rows <= 6u &&
        ds41_engram_e4m3::enabled() && !cuda_runtime_config()->graph_dump) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 exact Engram F16");
        if (!w) return 0;
        bool failed = false;
        const auto *packed = ds41_engram_e4m3::get(model_map, model_size, weight_offset, w, &failed);
        if (failed) return 0;
        if (packed) {
            const uint8_t *codes = (const uint8_t *)packed->device;
            const int32_t *scales = (const int32_t *)(codes + ds41_engram_e4m3::count);
            switch (rows) {
            case 1u: ds41_engram_e4m3::e4m3_scalar<<<800u, 1024u, 24576u>>>(
                (float *)out->ptr, codes, scales, (const float *)in->ptr, 6144u, 25600u); break;
            case 2u: ds41_engram_e4m3::e4m3_tiny<2,2,768><<<12800u, 64u>>>(
                (float *)out->ptr, codes, scales, (const float *)in->ptr); break;
            case 3u: ds41_engram_e4m3::e4m3_tiny<3,4,1024><<<6400u, 128u>>>(
                (float *)out->ptr, codes, scales, (const float *)in->ptr); break;
            case 4u: ds41_engram_e4m3::e4m3_tiny<4,8,1024><<<3200u, 256u>>>(
                (float *)out->ptr, codes, scales, (const float *)in->ptr); break;
            case 5u: ds41_engram_e4m3::e4m3_tiny<5,4,1024><<<6400u, 128u>>>(
                (float *)out->ptr, codes, scales, (const float *)in->ptr); break;
            case 6u: ds41_engram_e4m3::e4m3_tiny<6,4,512><<<6400u, 128u>>>(
                (float *)out->ptr, codes, scales, (const float *)in->ptr); break;
            }
            return cuda_ok(cudaGetLastError(), "V4.1 exact E4M3 Engram projection");
        }
    }
    if (width == 6144u && outputs == 25600u && rows >= 2u && rows <= 6u &&
        ds4_rocm_is_gfx1151() && !cuda_runtime_config()->graph_dump) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 exact Engram F16");
        if (!w) return 0;
        switch (rows) {
        case 2u: engram_lds_token_reuse<2,2,768><<<12800u, 64u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        case 3u: engram_lds_token_reuse<3,4,1024><<<6400u, 128u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        case 4u: engram_lds_token_reuse<4,8,1024><<<3200u, 256u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        case 5u: engram_lds_token_reuse<5,4,1024><<<6400u, 128u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        case 6u: engram_lds_token_reuse<6,4,512><<<6400u, 128u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        }
        return cuda_ok(cudaGetLastError(), "V4.1 exact tiny-row Engram projection");
    }
    if (width == 6144u && outputs == 25600u && rows >= 32u && rows <= 2048u &&
        ds4_rocm_is_gfx1151() && !g_quality_mode && !cuda_runtime_config()->graph_dump) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 exact Engram F16");
        if (!w) return 0;
        cuda_hipblaslt_gemm_plan *plan = v41_engram_lt_plan(rows);
        if (plan) {
            const uint32_t count = 6144u * rows;
            const uint64_t packed_bytes = (uint64_t)count * sizeof(__half);
            __half *packed = (__half *)cuda_tmp_alloc(packed_bytes + sizeof(uint32_t),
                "V4.1 Engram checked F16 activations");
            if (!packed) return 0;
            uint32_t *loss = (uint32_t *)((char *)packed + packed_bytes);
            if (!cuda_ok(hipMemsetAsync(loss, 0, sizeof(*loss), 0),
                         "V4.1 Engram conversion flag reset")) return 0;
            v41_engram_pack_checked_kernel<<<(count + 255u) / 256u, 256u>>>(
                packed, (const float *)in->ptr, count, loss);
            if (!cuda_ok(cudaGetLastError(), "V4.1 Engram checked conversion")) return 0;
            uint32_t converted_loss = 0;
            if (!cuda_ok(cudaMemcpy(&converted_loss, loss, sizeof(converted_loss),
                                   cudaMemcpyDeviceToHost), "V4.1 Engram conversion check")) return 0;
            if (!converted_loss) {
                const float alpha = 1.0f, beta = 0.0f;
                if (!hipblaslt_ok(hipblasLtMatmul(g_hipblaslt, plan->desc, &alpha,
                        w, plan->a_desc, packed, plan->b_desc, &beta,
                        out->ptr, plan->c_desc, out->ptr, plan->d_desc,
                        &plan->algo, NULL, 0, 0), "V4.1 Engram F16/F32")) return 0;
                return cuda_ok(cudaGetLastError(), "V4.1 Engram matrix projection");
            }
        }
        if (rows == 2048u) {
            engram_lds_token_reuse<16><<<dim3(3200u, 128u), 256u>>>(
                (float *)out->ptr, w, (const float *)in->ptr);
            return cuda_ok(cudaGetLastError(), "V4.1 exact Engram F32-input projection");
        }
        /* Partial batches retain the existing per-row F32-input fallback. */
    }
    if (width == 20480u && outputs == 24u && rows >= 8u && rows <= 2048u &&
        ds4_rocm_is_gfx1151()) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "f16");
        if (!w) return 0;
        /* Keep four-token tiles for smaller batches; the measured 384-row
         * workload and full 2048-row tiles favor eight-token reuse. */
        if (rows < 384u) {
            f16_ordered_token_reuse<4,8><<<dim3(3u, (rows + 3u) / 4u), 256u>>>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
        } else {
            f16_ordered_token_reuse<8,8><<<dim3(3u, (rows + 7u) / 8u), 256u>>>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
        }
        return cuda_ok(cudaGetLastError(), "V4.1 F32-input HC projection");
    }
    if (rows >= 32u && ds4_rocm_is_gfx1151()) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 F32-input F16 projection");
        if (!w) return 0;
        if (width == 5120u && outputs == 512u) {
            v41_launch_f16_sharedx_token_reuse<4u, 128u>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
            return cuda_ok(cudaGetLastError(), "V4.1 F16 512 token-reuse projection");
        }
        if (width == 5120u && outputs == 32u) {
            v41_launch_f16_sharedx_token_reuse<4u, 512u>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
            return cuda_ok(cudaGetLastError(), "V4.1 F16 32 token-reuse projection");
        }
    }
    if (width == 20480u && outputs == 24u && rows <= 6u &&
        ds4_rocm_is_gfx1151()) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 tiny HC F16");
        if (!w) return 0;
        /* Pair loads require aligned weights and F32 inputs. Tensor views
         * can be less aligned; retain the scalar kernel for those views. */
        if (((uintptr_t)w % alignof(__half2)) == 0u &&
            ((uintptr_t)in->ptr % alignof(float2)) == 0u) {
            v41_hc_ordered_prefetch_kernel<<<dim3(outputs, rows), 32u>>>(
                (float *)out->ptr, w, (const float *)in->ptr);
        } else {
            matmul_f16_ordered_chunks_kernel<<<dim3(outputs, rows), 32u>>>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
        }
        return cuda_ok(cudaGetLastError(), "V4.1 tiny HC F16 projection");
    }
    /* The general batched F16 API casts inputs to F16. Row views preserve decode arithmetic and retain F32 activations. */
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor x = {(float *)in->ptr + (uint64_t)row * width, (uint64_t)width * 4u, 0};
        ds4_gpu_tensor y = {(float *)out->ptr + (uint64_t)row * outputs, (uint64_t)outputs * 4u, 0};
        if (!ds4_gpu_matmul_f16_tensor(&y, model_map, model_size, weight_offset, width, outputs, &x, 1u)) return 0;
    }
    return 1;
}

/* Keep the scalar 256-lane reduction tree. Reuse weights across tiny rows
 * and replace its final five block barriers with the same warp additions. */
template <uint32_t TOKENS>
__global__ static void v41_router_f32_rows_kernel(float *out, const float *weights,
        const float *input, uint32_t outputs, uint32_t rows) {
    constexpr uint32_t width = 5120u;
    const uint32_t lane = threadIdx.x, row = blockIdx.x;
    float sum[TOKENS] = {};
#pragma unroll 1
    for (uint32_t base = 0; base < 20u; base += 4u) {
        float w[4], x[TOKENS][4];
#pragma unroll
        for (uint32_t u = 0; u < 4u; u++) {
            const uint32_t col = lane + (base + u) * 256u;
            w[u] = weights[(uint64_t)row * width + col];
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                x[t][u] = t < rows ? input[(uint64_t)t * width + col] : 0.0f;
        }
#pragma unroll
        for (uint32_t u = 0; u < 4u; u++) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                sum[t] = __fmaf_rn(w[u], x[t][u], sum[t]);
        }
    }
    __shared__ float partial[TOKENS][256];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; t++) partial[t][lane] = sum[t];
    __syncthreads();
    for (uint32_t stride = 128u; stride >= 32u; stride >>= 1u) {
        if (lane < stride) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                partial[t][lane] += partial[t][lane + stride];
        }
        __syncthreads();
    }
    if (lane < 32u) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; t++) {
            float value = partial[t][lane];
            for (uint32_t stride = 16u; stride; stride >>= 1u)
                value += __shfl_down(value, stride, 32);
            if (lane == 0u && t < rows) out[(uint64_t)t * outputs + row] = value;
        }
    }
}

template <uint32_t TOKENS, bool STRIPED>
__global__ static void v41_router_packed_rows_kernel(float *out, const void *weights,
        const float *input, uint32_t outputs, uint32_t rows) {
    constexpr uint32_t width = 5120u;
    const uint32_t lane = threadIdx.x, row = blockIdx.x;
    float sum[TOKENS] = {};
#pragma unroll 1
    for (uint32_t base = 0; base < 20u; base += 4u) {
        uint2 packed = {};
        if constexpr (STRIPED) packed = ((const uint2 *)weights)[((uint64_t)row*5u+base/4u)*256u+lane];
        float w[4], x[TOKENS][4];
#pragma unroll
        for (uint32_t u = 0; u < 4u; u++) {
            const uint32_t col = lane + (base + u) * 256u;
            if constexpr (STRIPED) {
                const uint32_t word=u<2u?packed.x:packed.y;
                w[u]=__uint_as_float((u&1u)?(word&0xffff0000u):(word<<16));
            } else w[u]=__uint_as_float((uint32_t)((const uint16_t *)weights)[(uint64_t)row*width+col]<<16);
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                x[t][u] = t < rows ? input[(uint64_t)t * width + col] : 0.0f;
        }
#pragma unroll
        for (uint32_t u = 0; u < 4u; u++) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                sum[t] = __fmaf_rn(w[u], x[t][u], sum[t]);
        }
    }
    __shared__ float partial[TOKENS][256];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; t++) partial[t][lane] = sum[t];
    __syncthreads();
    for (uint32_t stride = 128u; stride >= 32u; stride >>= 1u) {
        if (lane < stride) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                partial[t][lane] += partial[t][lane + stride];
        }
        __syncthreads();
    }
    if (lane < 32u) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; t++) {
            float value = partial[t][lane];
            for (uint32_t stride = 16u; stride; stride >>= 1u)
                value += __shfl_down(value, stride, 32);
            if (lane == 0u && t < rows) out[(uint64_t)t * outputs + row] = value;
        }
    }
}

extern "C" int ds4_gpu_dsv41_f32_projection_rows(ds4_gpu_tensor *out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t width, uint32_t outputs, uint32_t rows, const ds4_gpu_tensor *in) {
    uint64_t weight_bytes = 0;
    if (!width || !outputs || !rows || rows > 6u || !model_map ||
        !cuda_u64_mul3_checked(width, outputs, sizeof(float), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) ||
        !cuda_tensor_has_elems2(out, outputs, rows, 4u)) return 0;
    const float *w = (const float *)cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "V4.1 tiny F32 projection");
    if (!w) return 0;
    if (ds4_rocm_is_gfx1151() && width == 5120u &&
        (outputs == 128u || outputs == 384u)) {
        bool pack_failed=false;
        const void *packed=v41_router_packed_get(model_map,model_size,weight_offset,w,outputs,rows>3u,&pack_failed);
        if (pack_failed) return 0;
        if (packed) {
#define V41_ROUTER_PACKED(T,S) v41_router_packed_rows_kernel<T,S><<<outputs,256u>>>( \
            (float *)out->ptr,packed,(const float *)in->ptr,outputs,rows)
            if (rows==1u) { V41_ROUTER_PACKED(1u,false); }
            else if (rows==2u) { V41_ROUTER_PACKED(2u,false); }
            else if (rows==3u) { V41_ROUTER_PACKED(3u,false); }
            else { V41_ROUTER_PACKED(6u,true); }
#undef V41_ROUTER_PACKED
            return cuda_ok(cudaGetLastError(),"V4.1 lossless packed router");
        }
#define V41_ROUTER_ROWS(T) v41_router_f32_rows_kernel<T><<<outputs, 256u>>>( \
        (float *)out->ptr, w, (const float *)in->ptr, outputs, rows)
        if (rows == 1u) { V41_ROUTER_ROWS(1u); }
        else if (rows == 2u) { V41_ROUTER_ROWS(2u); }
        else if (rows == 3u) { V41_ROUTER_ROWS(3u); }
        else { V41_ROUTER_ROWS(6u); }
#undef V41_ROUTER_ROWS
        return cuda_ok(cudaGetLastError(), "V4.1 tiny F32 router projection");
    }
    /* Preserve the scalar F32 kernel, including its 256-lane reduction tree. */
    matmul_f32_kernel<<<dim3(outputs, rows), 256u>>>(
        (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
    return cuda_ok(cudaGetLastError(), "V4.1 tiny F32 projection");
}

extern "C" int ds4_gpu_hc_rms_scale_project_f16_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scale_scratch,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_rows, float eps) {
    if (!in_dim || !out_dim || !n_rows || !isfinite(eps) || eps <= 0.0f) return 0;
    return ds4_gpu_rms_norm_plain_rows_tensor(scale_scratch, x, in_dim, n_rows, eps) &&
        ds4_gpu_dsv41_projection_rows(out, model_map, model_size, weight_offset, in_dim, out_dim, n_rows, scale_scratch);
}

/* Four input values per lane, four Q8 blocks per wave. Scalar V4.1
 * projections retain F32 activations; the block-wise reduction differs from
 * the original lane accumulation and is qualified independently. */
template<bool ROUND_BF16 = false>
__global__ static void v41_q8_f32_blocks4_kernel(float *out,
        const unsigned char *weights, const float *input,
        uint32_t width, uint32_t outputs, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t token = blockIdx.y;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (row >= outputs) return;
    input += (uint64_t)token * width;
    out += (uint64_t)token * outputs;
    const uint32_t blocks = width / 32u;
    float acc = 0.0f;
    for (uint32_t b = lane / 8u; b < blocks; b += 4u) {
        const unsigned char *p = weights + row * row_bytes + (uint64_t)b * 34u;
        const float d = q8_0_scale_scalar(p);
        const uint32_t j = (lane & 7u) * 4u;
        float value = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++)
            value += (float)((const int8_t *)(p + 2u))[j + k] * input[b * 32u + j + k];
        acc += d * value;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = ROUND_BF16 ? v41_bf16(acc) : acc;
}

/* DSpark verifies two to six consecutive rows.  Reuse each packed Q8 block
 * across all rows while preserving the scalar blocks4 lane ownership,
 * per-lane dot boundary and wave32 reduction tree.  GROUPS supports the
 * interleaved eight-group attention-A layout; dense projections use one. */
template <uint32_t ROWS, uint32_t GROUPS = 1u, bool ROUND_BF16 = false>
__global__ __launch_bounds__(32) static void v41_q8_f32_tiny_rows_kernel(
        float *out, const unsigned char *weights, const float *input,
        uint32_t width, uint32_t rows_per_group, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x;
    const uint32_t group = row / rows_per_group;
    const uint32_t outputs = rows_per_group * GROUPS;
    const uint32_t blocks = width / 32u;
    float acc[ROWS] = {};
    for (uint32_t base = lane / 8u; base < blocks; base += 8u) {
#pragma unroll
        for (uint32_t u = 0; u < 2u; u++) {
            const uint32_t b = base + 4u * u;
            if (b >= blocks) continue;
            const unsigned char *p = weights + (uint64_t)row * row_bytes +
                                     (uint64_t)b * 34u;
            const float scale = q8_0_scale_scalar(p);
            const uint32_t j = (lane & 7u) * 4u;
#pragma unroll
            for (uint32_t t = 0; t < ROWS; t++) {
                const float *x = input + ((uint64_t)t * GROUPS + group) * width +
                                 (uint64_t)b * 32u + j;
                float value = 0.0f;
#pragma unroll
                for (uint32_t k = 0; k < 4u; k++)
                    value += (float)((const int8_t *)(p + 2u))[j + k] * x[k];
                acc[t] += scale * value;
            }
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < ROWS; t++) {
        const float total = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * outputs + row] = ROUND_BF16 ? v41_bf16(total) : total;
    }
}

template <uint32_t GROUPS, bool ROUND_BF16 = false>
static int v41_q8_f32_tiny_rows_launch(float *out, const unsigned char *weights,
        const float *input, uint32_t width, uint32_t rows_per_group,
        uint32_t rows, uint64_t row_bytes) {
    const uint32_t outputs = rows_per_group * GROUPS;
    switch (rows) {
        case 2u: v41_q8_f32_tiny_rows_kernel<2u, GROUPS, ROUND_BF16><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        case 3u: v41_q8_f32_tiny_rows_kernel<3u, GROUPS, ROUND_BF16><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        case 4u: v41_q8_f32_tiny_rows_kernel<4u, GROUPS, ROUND_BF16><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        case 5u: v41_q8_f32_tiny_rows_kernel<5u, GROUPS, ROUND_BF16><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        case 6u: v41_q8_f32_tiny_rows_kernel<6u, GROUPS, ROUND_BF16><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        default: return 0;
    }
    return cuda_ok(cudaGetLastError(), "V4.1 tiny-row Q8 projection");
}

template<bool ROUND>
__global__ static void v41_pair_q8_scalar(float *out0, float *out1,
        const unsigned char *weights0, const unsigned char *weights1, const float *input,
        uint32_t width, uint32_t outputs0, uint32_t outputs1, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t token = blockIdx.y;
    const uint64_t global_row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (global_row >= (uint64_t)outputs0 + outputs1) return;
    const bool second = global_row >= outputs0;
    const uint64_t row = second ? global_row - outputs0 : global_row;
    const uint32_t outputs = second ? outputs1 : outputs0;
    float *out = second ? out1 : out0;
    const unsigned char *weights = second ? weights1 : weights0;
    input += (uint64_t)token * width;
    out += (uint64_t)token * outputs;
    const uint32_t blocks = width / 32u;
    float acc = 0.0f;
    for (uint32_t b = lane / 8u; b < blocks; b += 4u) {
        const unsigned char *p = weights + row * row_bytes + (uint64_t)b * 34u;
        const float d = q8_0_scale_scalar(p);
        const uint32_t j = (lane & 7u) * 4u;
        float value = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++)
            value += (float)((const int8_t *)(p + 2u))[j + k] * input[b * 32u + j + k];
        acc += d * value;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = ROUND ? v41_bf16(acc) : acc;
}

template <uint32_t ROWS, bool ROUND>
__global__ __launch_bounds__(32) static void v41_pair_q8_tiny(
        float *out0, float *out1, const unsigned char *weights0,
        const unsigned char *weights1, const float *input,
        uint32_t width, uint32_t outputs0, uint32_t outputs1, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t global_row = blockIdx.x;
    if (global_row >= outputs0 + outputs1) return;
    const bool second = global_row >= outputs0;
    const uint32_t row = second ? global_row - outputs0 : global_row;
    const uint32_t rows_per_group = second ? outputs1 : outputs0;
    float *out = second ? out1 : out0;
    const unsigned char *weights = second ? weights1 : weights0;
    const uint32_t group = row / rows_per_group;
    const uint32_t outputs = rows_per_group * 1u;
    const uint32_t blocks = width / 32u;
    float acc[ROWS] = {};
    for (uint32_t base = lane / 8u; base < blocks; base += 8u) {
#pragma unroll
        for (uint32_t u = 0; u < 2u; u++) {
            const uint32_t b = base + 4u * u;
            if (b >= blocks) continue;
            const unsigned char *p = weights + (uint64_t)row * row_bytes +
                                     (uint64_t)b * 34u;
            const float scale = q8_0_scale_scalar(p);
            const uint32_t j = (lane & 7u) * 4u;
#pragma unroll
            for (uint32_t t = 0; t < ROWS; t++) {
                const float *x = input + ((uint64_t)t * 1u + group) * width +
                                 (uint64_t)b * 32u + j;
                float value = 0.0f;
#pragma unroll
                for (uint32_t k = 0; k < 4u; k++)
                    value += (float)((const int8_t *)(p + 2u))[j + k] * x[k];
                acc[t] += scale * value;
            }
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < ROWS; t++) {
        const float total = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * outputs + row] = ROUND ? v41_bf16(total) : total;
    }
}

static bool v41_q8_pair_overlap(const void *a, uint64_t an, const void *b, uint64_t bn) {
    const uintptr_t ap = (uintptr_t)a, bp = (uintptr_t)b;
    return ap <= bp ? bp - ap < an : ap - bp < bn;
}

extern "C" int ds4_gpu_dsv41_q8_attention_pair_bf16(
        ds4_gpu_tensor *q, ds4_gpu_tensor *kv, const void *model_map,
        uint64_t model_size, uint64_t q_offset, uint64_t kv_offset,
        uint32_t rows, const ds4_gpu_tensor *in) {
    const uint32_t width = 5120u, nq = 1280u, nk = 512u;
    const uint64_t row_bytes = UINT64_C(160) * 34u;
    if (!rows || rows > 6u || !model_map || !ds4_rocm_is_gfx1151() ||
        !cuda_model_range_fits(model_size, q_offset, nq * row_bytes) ||
        !cuda_model_range_fits(model_size, kv_offset, nk * row_bytes) ||
        !cuda_tensor_has_f32(in, (uint64_t)width * rows) ||
        !cuda_tensor_has_f32(q, (uint64_t)nq * rows) ||
        !cuda_tensor_has_f32(kv, (uint64_t)nk * rows)) return 0;
    if (v41_q8_pair_overlap(q->ptr, (uint64_t)nq * rows * 4u, kv->ptr, (uint64_t)nk * rows * 4u) ||
        v41_q8_pair_overlap(q->ptr, (uint64_t)nq * rows * 4u, in->ptr, (uint64_t)width * rows * 4u) ||
        v41_q8_pair_overlap(kv->ptr, (uint64_t)nk * rows * 4u, in->ptr, (uint64_t)width * rows * 4u)) return 0;
    const unsigned char *qw = (const unsigned char *)cuda_model_range_ptr(
        model_map, q_offset, nq * row_bytes, "V4.1 query-A pair");
    const unsigned char *kw = (const unsigned char *)cuda_model_range_ptr(
        model_map, kv_offset, nk * row_bytes, "V4.1 KV pair");
    if (!qw || !kw) return 0;
    if (rows == 1u) {
        v41_pair_q8_scalar<true><<<(nq + nk) / 8u, 256u>>>(
            (float *)q->ptr, (float *)kv->ptr, qw, kw, (const float *)in->ptr,
            width, nq, nk, row_bytes);
    } else {
#define V41_Q8_PAIR_ROW(N) case N: \
        v41_pair_q8_tiny<N, true><<<nq + nk, 32u>>>( \
            (float *)q->ptr, (float *)kv->ptr, qw, kw, (const float *)in->ptr, \
            width, nq, nk, row_bytes); break
        switch (rows) {
            V41_Q8_PAIR_ROW(2); V41_Q8_PAIR_ROW(3); V41_Q8_PAIR_ROW(4);
            V41_Q8_PAIR_ROW(5); V41_Q8_PAIR_ROW(6);
            default: return 0;
        }
#undef V41_Q8_PAIR_ROW
    }
    const char *trace = getenv("DS4_ROCM_V41_Q8_PAIR_TRACE");
    if (trace && !strcmp(trace, "1")) fprintf(stderr, "V41_Q8_PAIR rows=%u\n", rows);
    return cuda_ok(cudaGetLastError(), "V4.1 paired query-A/KV BF16");
}

template <bool ROUND_BF16>
static int v41_q8_projection_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                                uint64_t weight_offset, uint32_t width, uint32_t outputs,
                                                uint32_t rows, const ds4_gpu_tensor *in) {
    uint64_t weight_bytes = 0;
    if (!width || width % 32u || !outputs || !rows || rows > 8192u || !model_map ||
        !cuda_u64_mul3_checked(width / 32u, outputs, 34u, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) || !cuda_tensor_has_elems2(out, outputs, rows, 4u)) return 0;
    const unsigned char *weights = (const unsigned char *)cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "V4.1 Q8 projection");
    if (!weights) return 0;
    bool fused = false;
    if constexpr (ROUND_BF16) {
        const char *flag = getenv("DS4_ROCM_V41_Q8_SMALL_BF16_FUSION");
        if (rows <= 6u && ds4_rocm_is_gfx1151() && (!flag || !strcmp(flag, "1"))) {
            const char *trace = getenv("DS4_ROCM_V41_Q8_SMALL_BF16_TRACE");
            if (trace && !strcmp(trace, "1")) fprintf(stderr,
                "V41_Q8_SMALL_BF16 rows=%u width=%u outputs=%u\n", rows, width, outputs);
            if (rows == 1u) {
                v41_q8_f32_blocks4_kernel<true><<<(outputs + 7u) / 8u, 256u>>>(
                    (float *)out->ptr, weights, (const float *)in->ptr,
                    width, outputs, (uint64_t)(width / 32u) * 34u);
                return cuda_ok(cudaGetLastError(), "V4.1 scalar Q8 fused BF16");
            }
            return v41_q8_f32_tiny_rows_launch<1u, true>((float *)out->ptr, weights,
                (const float *)in->ptr, width, outputs, rows,
                (uint64_t)(width / 32u) * 34u);
        }
    }
    if (rows == 1u && ds4_rocm_is_gfx1151()) {
        v41_q8_f32_blocks4_kernel<<<(outputs + 7u) / 8u, 256u>>>(
            (float *)out->ptr, weights, (const float *)in->ptr,
            width, outputs, (uint64_t)(width / 32u) * 34u);
    } else if (rows <= 6u && ds4_rocm_is_gfx1151()) {
        const int ok = v41_q8_f32_tiny_rows_launch<1u>((float *)out->ptr, weights,
            (const float *)in->ptr, width, outputs, rows,
            (uint64_t)(width / 32u) * 34u);
        return ok && (!ROUND_BF16 || ds4_gpu_dsv41_quantize(out, outputs, rows, DS4_V41_BF16));
    } else if (!g_quality_mode && width == 1280u && (outputs == 32768u || outputs == 16384u ||
               outputs == 12288u || outputs == 8192u) && rows >= 32u && rows <= 2048u && ds4_rocm_is_gfx1151()) {
        fused = ROUND_BF16;
        /* Query-B, including contiguous two- and three-rank weight slices.
         * This numerical path rounds activations and decoded Q8 weights to
         * F16 before F32 accumulation; quality mode retains the F32 path. */
        matmul_q8_0_f32_batch_wmma_rowtile_kernel<256u, 16u, 16u, ROUND_BF16><<<dim3(outputs / 256u, (rows + 63u) / 64u), 512u>>>(
            (float *)out->ptr, weights, (const float *)in->ptr,
            rows, width, outputs, UINT64_C(40) * 34u);
    } else if (!g_quality_mode && rows >= 32u && rows <= 2048u && rows != 128u && ds4_rocm_is_gfx1151() &&
               ((width == 5120u && (outputs == 512u || outputs == 1280u || outputs == 2304u)) ||
                (width == 2304u && outputs == 5120u))) {
        fused = ROUND_BF16;
        /* Query-A, KV and shared-expert projections on prefill/replay tiles.
         * The shared-F32 path remains faster for the bounded 128-row tile.
         * A two-host, full-oracle sweep selected the 256-row/pad-8 tile for
         * the bulk of this topology.  The 32-row input-limited shape favors
         * 128 rows/pad-8, while Query-A crosses back to the 128-row tile at
         * the final 1984-row grid step. */
        if (width == 5120u && rows <= 32u) {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u, 8u, ROUND_BF16>
                <<<dim3((outputs + 127u) / 128u, (rows + 63u) / 64u), 256u>>>(
                    (float *)out->ptr, weights, (const float *)in->ptr,
                    rows, width, outputs, (uint64_t)(width / 32u) * 34u);
        } else if (width == 5120u && outputs == 512u && rows >= 1984u) {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u, 0u, ROUND_BF16>
                <<<dim3((outputs + 127u) / 128u, (rows + 63u) / 64u), 256u>>>(
                    (float *)out->ptr, weights, (const float *)in->ptr,
                    rows, width, outputs, (uint64_t)(width / 32u) * 34u);
        } else {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<256u, 16u, 8u, ROUND_BF16>
                <<<dim3((outputs + 255u) / 256u, (rows + 63u) / 64u), 512u>>>(
                    (float *)out->ptr, weights, (const float *)in->ptr,
                    rows, width, outputs, (uint64_t)(width / 32u) * 34u);
        }
    } else if (width == 1280u && outputs == 32768u && rows >= 32u && rows <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Reuse sixteen query-B activation rows with the same F32 lane
         * accumulation and wave reduction; keep the existing block tile. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, weights, (const float *)in->ptr,
            width / 32u, outputs, rows, (uint64_t)(width / 32u) * 34u, 8u, 16u, 8u);
    } else if (rows >= 32u && ds4_rocm_is_gfx1151()) {
        /* Reuse eight F32 activation rows without changing each lane's block
         * accumulation or wave reduction. No F16 cast or expanded weights. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, weights, (const float *)in->ptr,
            width / 32u, outputs, rows, (uint64_t)(width / 32u) * 34u, 8u, 8u, 8u);
    } else {
        matmul_q8_0_f32_batch_warp8_kernel<<<dim3((unsigned)(((uint64_t)outputs + 7u) / 8u), rows), 256>>>(
            (float *)out->ptr, weights, (const float *)in->ptr, width, outputs, rows, width / 32u);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 F32-input Q8 projection") &&
        (!ROUND_BF16 || fused || ds4_gpu_dsv41_quantize(out, outputs, rows, DS4_V41_BF16));
}
extern "C" int ds4_gpu_dsv41_q8_projection_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t width, uint32_t outputs,
        uint32_t rows, const ds4_gpu_tensor *in) {
    return v41_q8_projection_rows<false>(out, model_map, model_size,
        weight_offset, width, outputs, rows, in);
}

extern "C" int ds4_gpu_dsv41_q8_projection_rows_bf16(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t width, uint32_t outputs,
        uint32_t rows, const ds4_gpu_tensor *in) {
    return v41_q8_projection_rows<true>(out, model_map, model_size,
        weight_offset, width, outputs, rows, in);
}


/* V4.1 grouped output-A: retain physical token strides while using the
 * existing F16-operand/F32-accumulator WMMA body on bulk prefill rows. */
template <uint32_t M_TILE, uint32_t WARPS, uint32_t GROUPS = 8u>
__launch_bounds__(WARPS * 32u, 1)
__global__ static void v41_grouped_q8_f32_wmma_rowtile_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes) {
    const uint32_t group = (uint32_t)blockIdx.z;
    w += (uint64_t)group * out_dim * row_bytes;
    x += (uint64_t)group * in_dim;
    out += (uint64_t)group * out_dim;
    constexpr uint32_t N_TILE = 64u;
    constexpr uint32_t K_TILE = 32u;
    constexpr uint32_t M_PER_WARP = M_TILE / WARPS;
    constexpr uint32_t N_TILES_PER_WARP = N_TILE / 16u;

    const uint32_t block_m = (uint32_t)blockIdx.x * M_TILE;
    const uint32_t block_n = (uint32_t)blockIdx.y * N_TILE;
    if (block_m >= out_dim || block_n >= n_tokens) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t warp_m = block_m + warp_id * M_PER_WARP;
    const uint32_t my_row = warp_m + lane16;
    const uint32_t safe_row = my_row < out_dim ? my_row : (out_dim - 1u);
    const unsigned char *row_base = w + (uint64_t)safe_row * row_bytes;
    const uint32_t n_blocks = in_dim >> 5u;

    ds4_q8_float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ds4_q8_float8_t acc1 = acc0;
    ds4_q8_float8_t acc2 = acc0;
    ds4_q8_float8_t acc3 = acc0;

    __shared__ _Float16 lds_x[N_TILE * K_TILE];

    for (uint32_t bi = 0; bi < n_blocks; bi++) {
        for (uint32_t j = tid * 2u; j < N_TILE * K_TILE; j += blockDim.x * 2u) {
            const uint32_t nt = j >> 5u;
            const uint32_t kk = j & 31u;
            const uint32_t tok = block_n + nt;
            half2 xv = __floats2half2_rn(0.0f, 0.0f);
            if (tok < n_tokens) {
                const float2 f = *(const float2 *)(x + (uint64_t)tok * (GROUPS * 4096u) + bi * 32u + kk);
                xv = __floats2half2_rn(f.x, f.y);
            }
            *(half2 *)(lds_x + j) = xv;
        }
        __syncthreads();

        const unsigned char *bp = row_base + (uint64_t)bi * 34u;
        _Float16 sc;
        {
            uint16_t s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const int8_t *w0 = (const int8_t *)(bp + 2u);
        const int8_t *w1 = (const int8_t *)(bp + 18u);
        ds4_q8_half16_t a0;
        ds4_q8_half16_t a1;
#pragma unroll
        for (uint32_t i = 0; i < 16u; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }

#pragma unroll
        for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
            const uint32_t nt = ntile * 16u + lane16;
            const _Float16 *xb = lds_x + nt * K_TILE;
            const ds4_q8_half16_t b0 = *(const ds4_q8_half16_t *)(xb);
            const ds4_q8_half16_t b1 = *(const ds4_q8_half16_t *)(xb + 16u);
            if (ntile == 0u) {
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc0);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc0);
            } else if (ntile == 1u) {
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc1);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc1);
            } else if (ntile == 2u) {
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc2);
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc2);
            } else {
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc3);
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc3);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
        const uint32_t tok = block_n + ntile * 16u + lane16;
        if (tok >= n_tokens) continue;
        ds4_q8_float8_t acc = ntile == 0u ? acc0 : (ntile == 1u ? acc1 : (ntile == 2u ? acc2 : acc3));
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t row = warp_m + 2u * j + (lane >> 4u);
            if (row < out_dim) out[(uint64_t)tok * (GROUPS * 1024u) + row] = acc[j];
        }
    }
}

/* Scalar grouped projection with packed four-weight loads and F32 activations. */
__global__ static void v41_grouped_q8_f32_blocks4_kernel(
        float *out, const unsigned char *w, const float *x, int K, int M, int G) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= M * G) return;
    const uint32_t token = blockIdx.y;
    const float *in = x + ((size_t)token * G + row / M) * K;
    out += (size_t)token * M * G;
    float acc = 0;
    for (int b = lane / 8; b < K / 32; b += 4) {
        const unsigned char *p = w + ((size_t) row * (K / 32) + b) * 34;
        float d = __half2float(* (const __half *) p), v = 0;
        int j = (lane & 7) * 4;
#pragma unroll
        for (int k = 0; k < 4; k++) v += (float) ((const int8_t *) (p + 2)) [j + k] * in[b * 32 + j + k];
        acc += d * v;
    }
    acc = warp_sum_f32(acc);
    if (!lane) out[row] = acc;
}

extern "C" int ds4_gpu_dsv41_attention_output_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    const uint64_t a_bytes = UINT64_C(8192) * 128u * 34u, b_bytes = UINT64_C(5120) * 256u * 34u;
    if (!model_map || !n_tokens || !cuda_model_range_fits(model_size, out_a_offset, a_bytes) ||
        !cuda_model_range_fits(model_size, out_b_offset, b_bytes) ||
        !cuda_tensor_has_elems2(heads, n_tokens, 32768u, 4u) ||
        !cuda_tensor_has_elems2(low, n_tokens, 8192u, 4u) || !cuda_tensor_has_elems2(out, n_tokens, 5120u, 4u)) return 0;
    const unsigned char *a = (const unsigned char *)cuda_model_range_ptr(model_map, out_a_offset, a_bytes, "V4.1 attn_out_a");
    const unsigned char *b = (const unsigned char *)cuda_model_range_ptr(model_map, out_b_offset, b_bytes, "V4.1 attn_out_b");
    if (!a || !b) return 0;
    if (n_tokens >= 2u && n_tokens <= 6u && ds4_rocm_is_gfx1151()) {
        if (!v41_q8_f32_tiny_rows_launch<8u>((float *)low->ptr, a,
                (const float *)heads->ptr, 4096u, 1024u, n_tokens,
                UINT64_C(128) * 34u) ||
            !ds4_gpu_dsv41_quantize(low, 8192u, n_tokens, DS4_V41_BF16) ||
            !v41_q8_f32_tiny_rows_launch<1u>((float *)out->ptr, b,
                (const float *)low->ptr, 8192u, 5120u, n_tokens,
                UINT64_C(256) * 34u)) return 0;
        return 1;
    }
    if (n_tokens == 1u && ds4_rocm_is_gfx1151()) {
        v41_grouped_q8_f32_blocks4_kernel<<<1024u, 256u>>>(
            (float *)low->ptr, a, (const float *)heads->ptr, 4096, 1024, 8);
    } else if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Canonical eight groups of 4096 -> 1024, with physical F32 token
         * strides32768/8192. Keep the explicit BF16 low boundary below. */
        v41_grouped_q8_f32_wmma_rowtile_kernel<128u, 8u><<<dim3(8u, (n_tokens + 63u) / 64u, 8u), 256u>>>(
            (float *)low->ptr, a, (const float *)heads->ptr,
            n_tokens, 4096u, 1024u, UINT64_C(128) * 34u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_grouped_q8_a_sharedx((float *)low->ptr, a, (const float *)heads->ptr,
            n_tokens, 8u, 128u, 1024u, 128u * 34u, 8u, 8u, 8u);
    } else {
        grouped_q8_0_a_f32_batch_warp8_kernel<<<dim3(1024u, n_tokens), 256>>>((float *)low->ptr, a,
            (const float *)heads->ptr, 4096u, 1024u, 8u, n_tokens, 128u);
    }
    if (!cuda_ok(cudaGetLastError(), "V4.1 attention low projection") ||
        !ds4_gpu_dsv41_quantize(low, 8192u, n_tokens, DS4_V41_BF16)) return 0;
    if (n_tokens == 1u && ds4_rocm_is_gfx1151()) {
        /* The shared input is the same BF16-rounded output-A row above. */
        v41_q8_f32_blocks4_kernel<<<640u, 256u>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            8192u, 5120u, UINT64_C(256) * 34u);
    } else if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Keep the BF16 boundary above and use the existing generic bulk
         * matrix path: F16-rounded operands with F32 accumulation. */
        matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u><<<dim3(40u, (n_tokens + 63u) / 64u), 256u>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            n_tokens, 8192u, 5120u, UINT64_C(256) * 34u);
    } else if (n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Keep the existing BF16-rounded low rows and reuse sixteen tokens. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, b, (const float *)low->ptr,
            256u, 5120u, n_tokens, 256u * 34u, 8u, 16u, 8u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_q8_batch_sharedx((float *)out->ptr, b, (const float *)low->ptr,
            256u, 5120u, n_tokens, 256u * 34u, 8u, 8u, 8u);
    } else {
        matmul_q8_0_f32_batch_warp8_kernel<<<dim3(640u, n_tokens), 256>>>((float *)out->ptr, b,
            (const float *)low->ptr, 8192u, 5120u, n_tokens, 256u);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 attention output projection");
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_groups(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t group0, uint32_t groups) {
    /* Heads and low rows are packed for this rank's contiguous output
     * groups. Output-B keeps its original 8192-column physical row stride
     * while consuming this rank's groups * 1024 columns. */
    const uint64_t group_a_bytes = UINT64_C(1024) * 128u * 34u;
    const uint64_t b_bytes = UINT64_C(5120) * 256u * 34u;
    const uint32_t width = groups * 1024u;
    if (!groups || group0 >= 8u || groups > 8u - group0 || !model_map || !n_tokens ||
        !cuda_model_range_fits(model_size, out_a_offset, 8u * group_a_bytes) ||
        !cuda_model_range_fits(model_size, out_b_offset, b_bytes) ||
        !cuda_tensor_has_elems2(heads, n_tokens, groups * 4096u, 4u) ||
        !cuda_tensor_has_elems2(low, n_tokens, width, 4u) ||
        !cuda_tensor_has_elems2(out, n_tokens, 5120u, 4u)) return 0;
    const unsigned char *a = (const unsigned char *)cuda_model_range_ptr(model_map,
        out_a_offset + group0 * group_a_bytes, groups * group_a_bytes, "V4.1 TP attn_out_a");
    const unsigned char *b = (const unsigned char *)cuda_model_range_ptr(model_map,
        out_b_offset, b_bytes, "V4.1 TP attn_out_b");
    if (!a || !b) return 0;
    b += (uint64_t)group0 * 32u * 34u;
    const bool wmma = !g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151() &&
        (groups == 4u || groups == 3u || groups == 2u);
    if (n_tokens <= 6u && ds4_rocm_is_gfx1151()) {
        v41_grouped_q8_f32_blocks4_kernel<<<dim3(groups * 128u, n_tokens), 256u>>>(
            (float *)low->ptr, a, (const float *)heads->ptr, 4096, 1024, (int)groups);
    } else if (wmma) {
        const dim3 grid(8u, (n_tokens + 63u) / 64u, groups);
        if (groups == 4u)
            v41_grouped_q8_f32_wmma_rowtile_kernel<128u, 8u, 4u><<<grid, 256u>>>(
                (float *)low->ptr, a, (const float *)heads->ptr, n_tokens, 4096u, 1024u, UINT64_C(128) * 34u);
        else if (groups == 3u)
            v41_grouped_q8_f32_wmma_rowtile_kernel<128u, 8u, 3u><<<grid, 256u>>>(
                (float *)low->ptr, a, (const float *)heads->ptr, n_tokens, 4096u, 1024u, UINT64_C(128) * 34u);
        else
            v41_grouped_q8_f32_wmma_rowtile_kernel<128u, 8u, 2u><<<grid, 256u>>>(
                (float *)low->ptr, a, (const float *)heads->ptr, n_tokens, 4096u, 1024u, UINT64_C(128) * 34u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_grouped_q8_a_sharedx((float *)low->ptr, a, (const float *)heads->ptr,
            n_tokens, groups, 128u, 1024u, 128u * 34u, 8u, 8u, 8u);
    } else {
        grouped_q8_0_a_f32_batch_warp8_kernel<<<dim3(groups * 128u, n_tokens), 256>>>(
            (float *)low->ptr, a, (const float *)heads->ptr,
            4096u, 1024u, groups, n_tokens, 128u);
    }
    if (!cuda_ok(cudaGetLastError(), "V4.1 TP attention low projection") ||
        !ds4_gpu_dsv41_quantize(low, width, n_tokens, DS4_V41_BF16)) return 0;
    if (n_tokens <= 6u && ds4_rocm_is_gfx1151()) {
        v41_q8_f32_blocks4_kernel<<<dim3(640u, n_tokens), 256u>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            width, 5120u, UINT64_C(256) * 34u);
    } else if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u><<<dim3(40u, (n_tokens + 63u) / 64u), 256u>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            n_tokens, width, 5120u, UINT64_C(256) * 34u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_q8_batch_sharedx((float *)out->ptr, b, (const float *)low->ptr,
            groups * 32u, 5120u, n_tokens, 256u * 34u, 8u, n_tokens <= 2048u ? 16u : 8u, 8u);
    } else {
        /* This scalar kernel accepts separate input length and weight stride;
         * its column guard excludes the unowned groups of each physical row. */
        matmul_q8_0_f32_batch_warp8_kernel<<<dim3(640u, n_tokens), 256>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            width, 5120u, n_tokens, 256u);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 TP attention output projection");
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t tp_rank) {
    if (tp_rank > 1u) return 0;
    return ds4_gpu_dsv41_attention_output_tp_groups(out, low, model_map, model_size,
        out_a_offset, out_b_offset, heads, n_tokens, tp_rank * 4u, 4u);
}

__device__ static uint32_t shared_owner_rank(const int32_t *selected, uint32_t tie) {
    uint32_t left=0;
    for(uint32_t i=0;i<6;i++){
        const int32_t id=selected[i];
        if(id<0||id>=384)return 2;
        left += id<192;
    }
    return left==3 ? tie : left<3 ? 0u : 1u;
}

template<bool ROUND_BF16 = false>
__global__ static void owner_q8_scalar(float *out,
        const unsigned char *weights, const float *input,
        uint32_t width, uint32_t outputs, uint64_t row_bytes, const int32_t *selected, uint32_t rank, uint32_t tie) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t token = blockIdx.y;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (row >= outputs) return;
    input += (uint64_t)token * width;
    out += (uint64_t)token * outputs;
    if (shared_owner_rank(selected, tie) != rank) {
        if (lane == 0u) out[row] = 0.0f;
        return;
    }
    const uint32_t blocks = width / 32u;
    float acc = 0.0f;
    for (uint32_t b = lane / 8u; b < blocks; b += 4u) {
        const unsigned char *p = weights + row * row_bytes + (uint64_t)b * 34u;
        const float d = q8_0_scale_scalar(p);
        const uint32_t j = (lane & 7u) * 4u;
        float value = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++)
            value += (float)((const int8_t *)(p + 2u))[j + k] * input[b * 32u + j + k];
        acc += d * value;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = ROUND_BF16 ? v41_bf16(acc) : acc;
}


extern "C" int ds4_gpu_dsv41_shared_owner_scalar(ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
        const void *map, uint64_t size, uint64_t gate_offset,
        uint64_t up_offset, uint64_t down_offset, uint32_t rank, uint32_t tie) {
    if (!ds4_rocm_is_gfx1151()) return 0;
    constexpr uint64_t gate_bytes = UINT64_C(2304) * 5440u;
    constexpr uint64_t down_bytes = UINT64_C(5120) * 2448u;
    if (!g_deepseek41_model || !map || rank > 1u || tie > 1u ||
        !cuda_model_range_fits(size, gate_offset, gate_bytes) ||
        !cuda_model_range_fits(size, up_offset, gate_bytes) ||
        !cuda_model_range_fits(size, down_offset, down_bytes) ||
        !cuda_tensor_has_elems2(out, 1u, 5120u, 4u) ||
        !cuda_tensor_has_elems2(gate, 1u, 2304u, 4u) ||
        !cuda_tensor_has_elems2(up, 1u, 2304u, 4u) ||
        !cuda_tensor_has_elems2(mid, 1u, 2304u, 4u) ||
        !cuda_tensor_has_elems2(x, 1u, 5120u, 4u) ||
        !cuda_tensor_has_elems2(selected, 1u, 6u, 4u)) return -1;
    const auto *wg = (const unsigned char *)cuda_model_range_ptr(map, gate_offset, gate_bytes, "V4.1 balanced shared gate");
    const auto *wu = (const unsigned char *)cuda_model_range_ptr(map, up_offset, gate_bytes, "V4.1 balanced shared up");
    const auto *wd = (const unsigned char *)cuda_model_range_ptr(map, down_offset, down_bytes, "V4.1 balanced shared down");
    if (!wg || !wu || !wd) return -1;
    const auto *ids = (const int32_t *)selected->ptr;
    owner_q8_scalar<true><<<288u,256u>>>((float *)gate->ptr,wg,(const float *)x->ptr,5120u,2304u,5440u,ids,rank,tie);
    owner_q8_scalar<true><<<288u,256u>>>((float *)up->ptr,wu,(const float *)x->ptr,5120u,2304u,5440u,ids,rank,tie);
    if (!cuda_ok(cudaGetLastError(), "V4.1 balanced shared gate/up") ||
        !ds4_gpu_swiglu_tensor(mid,gate,up,2304u,10.0f,1.0f) ||
        !ds4_gpu_dsv41_quantize(mid,2304u,1u,DS4_V41_BF16)) return -1;
    owner_q8_scalar<true><<<640u,256u>>>((float *)out->ptr,wd,(const float *)mid->ptr,2304u,5120u,2448u,ids,rank,tie);
    if (!cuda_ok(cudaGetLastError(), "V4.1 balanced shared down")) return -1;
    const char *trace = getenv("DS4_ROCM_V41_SHARED_OWNER_TRACE");
    if (trace && strcmp(trace, "1") == 0)
        fprintf(stderr,"ds4: ROCm scalar shared owner rank=%u tie=%u\n",rank,tie);
    return 1;
}

/* Staged correctness reference. Validate global IDs before any pointer-table
 * lookup; a null table entry suppresses every unowned gate/up/down load.
 * The ordinary routed-MoE dispatch is untouched. */
extern "C" int ds4_gpu_dsv41_routed_moe_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *scratch,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x, uint32_t n_tokens, uint32_t tp_rank,
        bool allow_pointer_cache, bool dspark_step) {
    constexpr uint32_t experts = 384u, owned = 192u, used = 6u;
    constexpr uint64_t gate_row = 1320u, down_row = 756u;
    constexpr uint64_t gate_expert = gate_row * 2304u, down_expert = down_row * 5120u;
    routed_moe_launch_plan plan;
    if (!g_deepseek41_model || tp_rank > 1u || !n_tokens || n_tokens > 2048u ||
        !routed_moe_build_plan(out, gate, up, mid, scratch, model_map, model_size,
            gate_offset, up_offset, down_offset, 16u, 10u, gate_expert, down_expert,
            5120u, 2304u, 5120u, selected, weights, experts, used, x, n_tokens, &plan)) return 0;
    const uint64_t pairs = (uint64_t)n_tokens * used;
    std::vector<int32_t> ids((size_t)pairs);
    if (!ds4_gpu_tensor_read(selected, 0, ids.data(), pairs * sizeof(int32_t))) return 0;
    for (int32_t id : ids) if (id < 0 || (uint32_t)id >= experts) {
        fprintf(stderr, "ds4: V4.1 TP invalid global expert ID %d\n", id);
        return 0;
    }
    if (n_tokens >= 128u && !g_quality_mode && ds4_rocm_is_gfx1151()) {
        if (!ds4_gpu_dsv41_moe_tp_gate_up(gate, up, model_map, model_size,
                gate_offset, up_offset, selected, x, n_tokens, tp_rank)) return 0;
        const uint64_t count = pairs * 2304u;
        moe_swiglu_weighted_f32_kernel<<<(uint32_t)((count + 255u) / 256u), 256>>>(
            (float *)mid->ptr, (const float *)gate->ptr, (const float *)up->ptr,
            (const float *)weights->ptr, count, 2304u, 10.f);
        if (!cuda_ok(cudaGetLastError(), "V4.1 TP MMQ weighted activation") ||
            !ds4_gpu_synchronize()) return 0;
        return ds4_gpu_dsv41_moe_tp_down(out, scratch, mid, selected,
            model_map, model_size, down_offset, n_tokens, tp_rank);
    }
    const uint32_t first = tp_rank * owned;
    const char *g = cuda_model_range_ptr(model_map, gate_offset + first * gate_expert,
                                        owned * gate_expert, "V4.1 TP owned gate");
    const char *u = cuda_model_range_ptr(model_map, up_offset + first * gate_expert,
                                        owned * gate_expert, "V4.1 TP owned up");
    const char *d = cuda_model_range_ptr(model_map, down_offset + first * down_expert,
                                        owned * down_expert, "V4.1 TP owned down");
    if (!g || !u || !d) return 0;
    bool cache_failed = false;
    const bool dspark_async = v41_tp_dspark_cache_async_enabled(dspark_step, n_tokens);
    const char *const *slots = (allow_pointer_cache || dspark_async) ?
        v41_tp_pointer_cache_get(model_map, model_size, gate_offset, up_offset,
            down_offset, tp_rank, g, u, d, &cache_failed) : nullptr;
    if (cache_failed) return 0;
    ds4_gpu_tensor *table = nullptr;
    if (!slots) {
        const char *tables[3][experts] = {};
        for (uint32_t i = 0; i < owned; ++i) {
            tables[0][first+i] = g + i * gate_expert;
            tables[1][first+i] = u + i * gate_expert;
            tables[2][first+i] = d + i * down_expert;
        }
        table = ds4_gpu_tensor_alloc(sizeof(tables));
        if (!table) return 0;
        if (!ds4_gpu_tensor_write(table, 0, tables, sizeof(tables))) {
            ds4_gpu_tensor_free(table);
            return 0;
        }
        slots = (const char *const *)table->ptr;
    }
    const uint64_t mids = pairs * 2304u;
    int ok = ds4_gpu_tensor_fill_f32(gate, 0.f, mids) &&
        ds4_gpu_tensor_fill_f32(up, 0.f, mids) &&
        ds4_gpu_tensor_fill_f32(mid, 0.f, mids);
    cuda_block_q8_K *xq = (cuda_block_q8_K *)scratch->ptr;
    if (ok) {
        q8_K_quantize_kernel<<<dim3(20u, n_tokens), 256u>>>(
            xq, (const float *)x->ptr, 5120u, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "V4.1 TP input quantization");
    }
    if (ok) {
        if (n_tokens == 1u && ds4_rocm_is_gfx1151()) {
            moe_v41_gate_up_wave_ptrs_kernel<4><<<dim3(576u, used), 128>>>(
                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                slots, slots + experts, xq, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, 0, gate_row, 20u, 2304u, used,
                1u, 0x3fu, 10.f);
        } else {
            moe_gate_up_mid_qwarp32_ptrs_kernel<<<dim3(18u, (uint32_t)pairs), 256u>>>(
                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                slots, slots + experts, xq, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, gate_row, 20u, 2304u, used, 0x3fu, 10.f);
        }
        ok = cuda_ok(cudaGetLastError(), "V4.1 TP owned gate/up");
    }
    if (ok) {
        if (n_tokens == 1u && !g_quality_mode && ds4_rocm_is_gfx1151()) {
            // Gate/up has consumed xq; reuse its scratch for the six mid rows.
            cuda_block_q8_K *midq = (cuda_block_q8_K *)scratch->ptr;
            q8_K_quantize_kernel<<<dim3(9u, used), 256u>>>(
                midq, (const float *)mid->ptr, 2304u, used);
            ok = cuda_ok(cudaGetLastError(), "V4.1 TP mid quantization");
            if (ok) {
                moe_v41_down_wave_ptrs_kernel<4><<<1280u, 128>>>(
                    (float *)out->ptr, slots + 2u * experts, midq,
                    (const int32_t *)selected->ptr, 0, down_row, 9u, 5120u, used);
                ok = cuda_ok(cudaGetLastError(), "V4.1 TP quantized wave down");
            }
        } else {
            moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<dim3(640u, n_tokens), 256u>>>(
                (float *)out->ptr, slots + 2u * experts, (const float *)mid->ptr,
                (const int32_t *)selected->ptr, n_tokens, 2304u, 5120u, down_row, used);
            ok = cuda_ok(cudaGetLastError(), "V4.1 TP owned down");
        }
    }
    /* Only opted-in DSpark keeps immutable tables past this return. Ordered
     * consumers and TP arrival use the same stream; fallback/errors still drain. */
    const bool deferred = v41_tp_pointer_cache_may_defer(ok && !table,
        dspark_step, n_tokens);
    if (!deferred && !ds4_gpu_synchronize()) ok = 0;
    static bool traced[2][6] = {};
    if (deferred && !traced[tp_rank][n_tokens - 1u]) {
        const char *trace = getenv("DS4_ROCM_V41_TP_POINTER_CACHE_TRACE");
        if (trace && !strcmp(trace, "1")) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "TP DSpark cache async model=deepseek41 rank=%u rows=%u persistent=1 deferred=1\n",
                tp_rank, n_tokens);
            traced[tp_rank][n_tokens - 1u] = true;
        }
    }
    ds4_gpu_tensor_free(table);
    return ok;
}

/* Native MXFP4 x Q8_K matrix kernel. Four routed tokens use
 * the current sorted metadata; 16 feature rows share each WMMA fragment.
 * FP32 scale/accumulation order differs from the scalar half-block reader. */
__global__ static void moe_down_mxfp4_tp_wmma_tile4_kernel(
        float *partials, const char *down_table, const cuda_block_q8_K *midq,
        const uint32_t *sorted, const uint32_t *offsets, const uint32_t *counts,
        const uint32_t *total_tiles, const uint32_t *tile_experts,
        const uint32_t *tile_starts, uint64_t down_row_bytes) {
    using i4 = int __attribute__((ext_vector_type(4)));
    using i8 = int __attribute__((ext_vector_type(8)));
    const uint32_t tile = blockIdx.y;
    if (tile >= *total_tiles) return;
    const uint32_t expert = tile_experts[tile];
    const char *base = ((const char *const *)down_table)[expert];
    if (!base) return;
    const uint32_t start = tile_starts[tile];
    const uint32_t np = min(counts[expert] - start, 4u);
    uint32_t pair[4] = {};
    #pragma unroll
    for (uint32_t p = 0; p < 4u; ++p)
        if (p < np) pair[p] = sorted[offsets[expert] + start + p];
    __shared__ cuda_block_q8_K staged[4u * 9u];
    constexpr uint32_t words = 9u * sizeof(cuda_block_q8_K) / sizeof(uint32_t);
    for (uint32_t i = threadIdx.x; i < np * words; i += blockDim.x) {
        const uint32_t p = i / words, word = i - p * words;
        ((uint32_t *)staged)[i] = ((const uint32_t *)(midq + (uint64_t)pair[p] * 9u))[word];
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u, row = lane & 15u, half = lane >> 4u;
    const uint32_t first_feature = blockIdx.x * 64u + (threadIdx.x >> 5u) * 16u;
    const auto *w = (const cuda_block_mxfp4 *)(base + (uint64_t)(first_feature + row) * down_row_bytes);
    float acc[8] = {};
    for (uint32_t mb = 0; mb < 72u; ++mb) {
        i4 a = {}, b = {};
        const cuda_block_q8_K *x = row < np ? staged + row * 9u + (mb >> 3u) : nullptr;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t packed = *(const uint32_t *)(w[mb].qs + 4u * j);
            a[j] = (int)dev_mxfp4_unpack4(packed >> (half * 4u));
            b[j] = x ? *(const int32_t *)(x->qs + (mb & 7u) * 32u + half * 16u + 4u * j) : 0;
        }
        i4 ao, bo;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            ao[j] = __shfl_xor(a[j], 16, 32);
            bo[j] = __shfl_xor(b[j], 16, 32);
        }
        i4 a0 = half ? ao : a, a1 = half ? a : ao;
        i4 b0 = half ? bo : b, b1 = half ? b : bo;
        i8 z = {};
        z = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, z, true);
        z = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, z, true);
        const float weight_scale = dev_e8m0_to_f32(w[mb].e);
        const float activation_scale = x ? 0.5f * x->d : 0.f;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const float scale = activation_scale * __shfl(weight_scale, 2u * j + half, 32);
            acc[j] += scale * (float)z[j];
        }
    }
    if (row < np) {
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j)
            partials[(uint64_t)pair[row] * 5120u + first_feature + 2u * j + half] = acc[j];
    }
}

/* Bulk gate/up: existing sorted routing and Q8_K inputs.
 * Both native projections share activation fragments. Sequential K32 FP32
 * accumulation differs from the original half-block/warp reduction. */
__global__ static void moe_gate_up_mid_mxfp4_tp_wmma_tile16_kernel(
        float *gate_out, float *up_out, float *mid_out,
        const char *gate_table, const char *up_table,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted, const uint32_t *offsets, const uint32_t *counts,
        const uint32_t *total_tiles, const uint32_t *tile_experts,
        const uint32_t *tile_starts, const float *weights) {
    using i4 = int __attribute__((ext_vector_type(4)));
    using i8 = int __attribute__((ext_vector_type(8)));
    const uint32_t tile = blockIdx.y;
    if (tile >= *total_tiles) return;
    const uint32_t expert = tile_experts[tile];
    const char *gate_base = ((const char *const *)gate_table)[expert];
    const char *up_base = ((const char *const *)up_table)[expert];
    if (!gate_base || !up_base) return;
    const uint32_t start = tile_starts[tile];
    /* Existing metadata has four-pair starts. A start divisible by sixteen
     * owns the next sixteen pairs; intervening starts belong to that group.
     * Sparse experts use the scalar tile. Checks precede all barriers. */
    if ((start & 15u) || counts[expert] < 8u) return;
    const uint32_t np = min(counts[expert] - start, 16u);
    const uint32_t lane = threadIdx.x & 31u, row = lane & 15u, half = lane >> 4u;
    const uint32_t pair = row < np ? sorted[offsets[expert] + start + row] : 0u;
    __shared__ cuda_block_q8_K staged[16u];
    constexpr uint32_t words = sizeof(cuda_block_q8_K) / sizeof(uint32_t);
    const uint32_t first_feature = blockIdx.x * 64u + (threadIdx.x >> 5u) * 16u;
    const auto *gw = (const cuda_block_mxfp4 *)(gate_base + (uint64_t)(first_feature + row) * 2720u);
    const auto *uw = (const cuda_block_mxfp4 *)(up_base + (uint64_t)(first_feature + row) * 2720u);
    float gate[8] = {}, up[8] = {};
    for (uint32_t qb = 0; qb < 20u; ++qb) {
        for (uint32_t i = threadIdx.x; i < np * words; i += blockDim.x) {
            const uint32_t p = i / words, word = i - p * words;
            const uint32_t input_pair = sorted[offsets[expert] + start + p];
            const auto *src = (const uint32_t *)(xq + (uint64_t)(input_pair / 6u) * 20u + qb);
            ((uint32_t *)staged)[i] = src[word];
        }
        __syncthreads();
        for (uint32_t subblock = 0; subblock < 8u; ++subblock) {
            const uint32_t mb = qb * 8u + subblock;
            const cuda_block_q8_K *x = row < np ? staged + row : nullptr;
            i4 b = {};
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                b[j] = x ? *(const int32_t *)(x->qs + (mb & 7u) * 32u + half * 16u + 4u * j) : 0;
            i4 bo;
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) bo[j] = __shfl_xor(b[j], 16, 32);
            const i4 b0 = half ? bo : b, b1 = half ? b : bo;
            const float activation_scale = x ? 0.5f * x->d : 0.f;
            #pragma unroll
            for (uint32_t projection = 0; projection < 2u; ++projection) {
                const auto *w = projection ? uw : gw;
                i4 a = {};
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint32_t packed = *(const uint32_t *)(w[mb].qs + 4u * j);
                    a[j] = (int)dev_mxfp4_unpack4(packed >> (half * 4u));
                }
                i4 ao;
                #pragma unroll
                for (uint32_t j = 0; j < 4u; ++j) ao[j] = __shfl_xor(a[j], 16, 32);
                const i4 a0 = half ? ao : a, a1 = half ? a : ao;
                i8 z = {};
                z = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, z, true);
                z = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, z, true);
                const float weight_scale = dev_e8m0_to_f32(w[mb].e);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) {
                    const float scale = activation_scale * __shfl(weight_scale, 2u * j + half, 32);
                    if (projection) up[j] += scale * (float)z[j];
                    else gate[j] += scale * (float)z[j];
                }
            }
        }
        __syncthreads();
    }
    if (row < np) {
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            float g = gate[j], u = up[j];
            if (g > 10.f) g = 10.f;
            if (u > 10.f) u = 10.f;
            if (u < -10.f) u = -10.f;
            const uint64_t off = (uint64_t)pair * 2304u + first_feature + 2u * j + half;
            gate_out[off] = g;
            up_out[off] = u;
            mid_out[off] = (g / (1.f + expf(-g))) * u * weights[pair];
        }
    }
}

extern "C" int ds4_gpu_dsv41_routed_moe_typed_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *scratch,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x, uint32_t n_tokens, uint32_t tp_rank,
        uint32_t gate_type, uint32_t down_type,
        bool allow_pointer_cache, bool dspark_step) {
    if (gate_type == 16u && down_type == 10u)
        return ds4_gpu_dsv41_routed_moe_tp_tensor(out, gate, up, mid, scratch,
            model_map, model_size, gate_offset, up_offset, down_offset,
            selected, weights, x, n_tokens, tp_rank, allow_pointer_cache, dspark_step);
    constexpr uint32_t experts = 384u, owned = 192u, used = 6u;
    constexpr uint64_t gate_row = 2720u, down_row = 1224u;
    constexpr uint64_t gate_expert = gate_row * 2304u, down_expert = down_row * 5120u;
    routed_moe_launch_plan plan;
    if (gate_type != 39u || down_type != 39u || !g_deepseek41_model ||
        tp_rank > 1u || !n_tokens || n_tokens > 2048u ||
        !routed_moe_build_plan(out, gate, up, mid, scratch, model_map, model_size,
            gate_offset, up_offset, down_offset, 39u, 39u, gate_expert, down_expert,
            5120u, 2304u, 5120u, selected, weights, experts, used, x, n_tokens, &plan)) return 0;
    const uint64_t pairs = (uint64_t)n_tokens * used;
    std::vector<int32_t> ids((size_t)pairs);
    if (!ds4_gpu_tensor_read(selected, 0, ids.data(), pairs * sizeof(int32_t))) return 0;
    for (int32_t id : ids) if (id < 0 || (uint32_t)id >= experts) {
        fprintf(stderr, "ds4: V4.1 MXFP4 TP invalid global expert ID %d\n", id);
        return 0;
    }
    const uint32_t first = tp_rank * owned;
    const char *g = cuda_model_range_ptr(model_map, gate_offset + first * gate_expert,
                                        owned * gate_expert, "V4.1 MXFP4 TP owned gate");
    const char *u = cuda_model_range_ptr(model_map, up_offset + first * gate_expert,
                                        owned * gate_expert, "V4.1 MXFP4 TP owned up");
    const char *d = cuda_model_range_ptr(model_map, down_offset + first * down_expert,
                                        owned * down_expert, "V4.1 MXFP4 TP owned down");
    if (!g || !u || !d) return 0;
    const char *tables[3][experts] = {};
    for (uint32_t i = 0; i < owned; ++i) {
        tables[0][first+i] = g + i * gate_expert;
        tables[1][first+i] = u + i * gate_expert;
        tables[2][first+i] = d + i * down_expert;
    }
    ds4_gpu_tensor *table = ds4_gpu_tensor_alloc(sizeof(tables));
    if (!table) return 0;
    int ok = ds4_gpu_tensor_write(table, 0, tables, sizeof(tables));
    const char *slots = (const char *)table->ptr;
    cuda_block_q8_K *q = (cuda_block_q8_K *)scratch->ptr;
    if (ok) {
        q8_K_quantize_kernel<<<dim3(20u, n_tokens), 256u>>>(
            q, (const float *)x->ptr, 5120u, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "V4.1 MXFP4 TP input quantization");
    }
    const char *bulk_env = getenv("DS4_ROCM_V41_MXFP4_TP_BULK");
    const char *verify_env = getenv("DS4_ROCM_V41_MXFP4_TP_VERIFY_TILE");
    const bool verify_tile = n_tokens == 6u && ds4_rocm_is_gfx1151() &&
        (!verify_env || !strcmp(verify_env, "1") || !strcmp(verify_env, "2"));
    const uint32_t tile_tokens = verify_tile &&
        (!verify_env || !strcmp(verify_env, "2")) ? 2u : 4u;
    const bool bulk = verify_tile || (n_tokens >= 128u && ds4_rocm_is_gfx1151() &&
        (!bulk_env || !strcmp(bulk_env, "1")));
    ds4_gpu_tensor *metadata = nullptr;
    if (ok && bulk) {
        /* Stable global-ID buckets preserve the caller's routing arrays.
         * Remote expert tiles exit through null immutable table entries. */
        const uint32_t capacity = routed_moe_expert_tile_capacity((uint32_t)pairs, tile_tokens, experts);
        const uint64_t words = 4u * (experts + 1u) + pairs + 1u + 2u * capacity;
        metadata = ds4_gpu_tensor_alloc(words * sizeof(uint32_t));
        if (!metadata) ok = 0;
        if (ok) {
            uint32_t *counts = (uint32_t *)metadata->ptr;
            uint32_t *offsets = counts + experts + 1u;
            uint32_t *cursors = offsets + experts + 1u;
            uint32_t *tile_offsets = cursors + experts + 1u;
            uint32_t *sorted = tile_offsets + experts + 1u;
            uint32_t *total = sorted + pairs;
            uint32_t *tile_experts = total + 1u;
            uint32_t *starts = tile_experts + capacity;
            if (!verify_tile)
                ok = cuda_ok(cudaMemset(counts, 0, (experts + 1u) * sizeof(uint32_t)), "MXFP4 TP bucket clear");
            for (ds4_gpu_tensor *target : {gate, up, mid})
                if (ok) ok = cuda_ok(cudaMemset(target->ptr, 0, pairs * 2304u * sizeof(float)), "MXFP4 TP owned output clear");
            if (ok) {
                if (verify_tile) {
                    /* Three to six rows have at most 36 validated pairs, within the existing
                     * narrow builder's 48-entry shared routing array. */
                    moe_build_narrow_sorted_tiles_kernel<<<1u, 256u>>>(
                        counts, offsets, sorted, tile_offsets, total,
                        tile_experts, starts, (const int32_t *)selected->ptr,
                        (uint32_t)pairs, tile_tokens, experts);
                } else {
                    moe_count_sorted_pairs_kernel<<<(pairs + 255u) / 256u, 256u>>>(counts, (const int32_t *)selected->ptr, (uint32_t)pairs, experts);
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, experts);
                    moe_scatter_sorted_pairs_deterministic_kernel<<<experts, 1>>>(sorted, offsets, (const int32_t *)selected->ptr, (uint32_t)pairs, experts);
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, total, counts, 4u, experts);
                    moe_build_expert_tiles_kernel<<<2, 256>>>(tile_experts, starts, tile_offsets, counts, 4u, experts);
                }
                const char *gate_wmma = getenv("DS4_ROCM_V41_MXFP4_TP_GATE_WMMA");
                if (tile_tokens == 2u) {
                    moe_gate_up_mid_mxfp4_tp_tile4_qwarp32_kernel<2u><<<dim3(capacity, 288u), 256u, 2u * 20u * sizeof(cuda_block_q8_K)>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        slots, slots + experts * sizeof(char *), q, sorted, offsets, counts,
                        total, tile_experts, starts, (const float *)weights->ptr,
                        gate_expert, gate_row, 20u, 2304u, used, 0u, 1u, 10.f);
                } else if (n_tokens >= 128u && (!gate_wmma || !strcmp(gate_wmma, "1"))) {
                    /* Sparse experts keep the existing exact four-row path. */
                    moe_gate_up_mid_mxfp4_tp_tile4_qwarp32_kernel<4u><<<dim3(capacity, 288u), 256u, 4u * 20u * sizeof(cuda_block_q8_K)>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        slots, slots + experts * sizeof(char *), q, sorted, offsets, counts,
                        total, tile_experts, starts, (const float *)weights->ptr,
                        gate_expert, gate_row, 20u, 2304u, used, 8u, 1u, 10.f);
                    moe_gate_up_mid_mxfp4_tp_wmma_tile16_kernel<<<dim3(36u, capacity), 128u>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        slots, slots + experts * sizeof(char *), q, sorted, offsets, counts,
                        total, tile_experts, starts, (const float *)weights->ptr);
                } else {
                    moe_gate_up_mid_mxfp4_tp_tile4_qwarp32_kernel<4u><<<dim3(capacity, 288u), 256u, 4u * 20u * sizeof(cuda_block_q8_K)>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        slots, slots + experts * sizeof(char *), q, sorted, offsets, counts,
                        total, tile_experts, starts, (const float *)weights->ptr,
                        gate_expert, gate_row, 20u, 2304u, used, 0u, 1u, 10.f);
                }
                ok = cuda_ok(cudaGetLastError(), "V4.1 MXFP4 TP owned tile4 qwarp32 gate/up");
            }
        }
    } else if (ok) {
        moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<true><<<dim3(288u, (uint32_t)pairs), 256u>>>(
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
            slots, slots + experts * sizeof(char *), q,
            (const int32_t *)selected->ptr, (const float *)weights->ptr,
            gate_expert, gate_row, 20u, 2304u, used, 1u, 10.f);
        ok = cuda_ok(cudaGetLastError(), "V4.1 MXFP4 TP owned gate/up");
    }
    if (ok) {
        q8_K_quantize_kernel<<<dim3(9u, (uint32_t)pairs), 256u>>>(
            q, (const float *)mid->ptr, 2304u, (uint32_t)pairs);
        ok = cuda_ok(cudaGetLastError(), "V4.1 MXFP4 TP mid quantization");
    }
    if (ok) {
        const char *sorted_env = getenv("DS4_ROCM_V41_MXFP4_TP_DOWN_SORTED");
        const bool sorted_down = metadata && bulk && n_tokens >= 128u &&
            ds4_rocm_is_gfx1151() && (!sorted_env || strcmp(sorted_env, "0"));
        if (sorted_down) {
            /* The input scratch contains live quantized activations. Partial
             * outputs need distinct storage until the slot sum completes. */
            ds4_gpu_tensor *partials = ds4_gpu_tensor_alloc(pairs * 5120u * sizeof(float));
            if (!partials) ok = 0;
            if (ok) ok = cuda_ok(cudaMemset(partials->ptr, 0, pairs * 5120u * sizeof(float)),
                                 "MXFP4 TP sorted down owned output clear");
            if (ok) {
                const uint32_t capacity = routed_moe_expert_tile_capacity((uint32_t)pairs, 4u, experts);
                const uint32_t *counts = (const uint32_t *)metadata->ptr;
                const uint32_t *offsets = counts + experts + 1u;
                const uint32_t *sorted = counts + 4u * (experts + 1u);
                const uint32_t *total = sorted + pairs;
                const uint32_t *tile_experts = total + 1u;
                const uint32_t *starts = tile_experts + capacity;
                moe_down_mxfp4_tp_wmma_tile4_kernel<<<dim3(80u, capacity), 128u>>>(
                    (float *)partials->ptr, slots + 2u * experts * sizeof(char *), q,
                    sorted, offsets, counts, total, tile_experts, starts, down_row);
                ok = cuda_ok(cudaGetLastError(), "MXFP4 TP sorted down partials");
                if (ok) {
                    moe_sum_kernel<<<(n_tokens * 5120u + 255u) / 256u, 256u>>>(
                        (float *)out->ptr, (const float *)partials->ptr, 5120u, used, n_tokens);
                    ok = cuda_ok(cudaGetLastError(), "MXFP4 TP sorted down slot sum");
                }
            }
            if (!ds4_gpu_synchronize()) ok = 0;
            ds4_gpu_tensor_free(partials);
        } else {
            moe_down_mxfp4_sum6_qwarp32_kernel<true, true><<<dim3(640u, n_tokens), 256u>>>(
                (float *)out->ptr, slots + 2u * experts * sizeof(char *), q,
                (const int32_t *)selected->ptr, down_expert, down_row, 9u, 5120u, used);
        }
        if (ok) ok = cuda_ok(cudaGetLastError(), "V4.1 MXFP4 TP owned down");
    }
    if (!ds4_gpu_synchronize()) ok = 0;
    ds4_gpu_tensor_free(metadata);
    ds4_gpu_tensor_free(table);
    return ok;
}

/* Staged ownership adapter for the unchanged qualified Q2 down dispatcher.
 * Pair order is stable within each local expert, as in its GPU sort. */
extern "C" int ds4_gpu_dsv41_moe_tp_down(
        ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t n_tokens, uint32_t tp_rank) {
    constexpr uint64_t expert_bytes = UINT64_C(5120) * 756u;
    const uint64_t pairs = (uint64_t)n_tokens * 6u, mids = pairs * 2304u;
    if (!g_deepseek41_model || !ds4_rocm_is_gfx1151() || tp_rank > 1u ||
        n_tokens < 2u || n_tokens > 2048u || !out || !scratch || !mid || !selected ||
        !model_map || out->bytes < (uint64_t)n_tokens * 5120u * sizeof(float) ||
        scratch->bytes < pairs * 5120u * sizeof(float) ||
        mid->bytes < mids * sizeof(float) || selected->bytes < pairs * sizeof(int32_t) ||
        down_offset > model_size || 384u * expert_bytes > model_size - down_offset) return 0;
    std::vector<int32_t> ids((size_t)pairs);
    if (!ds4_gpu_tensor_read(selected, 0, ids.data(), pairs * sizeof(int32_t))) return 0;
    for (uint32_t row = 0; row < n_tokens; ++row) {
        for (uint32_t j = 0; j < 6u; ++j) {
            const int32_t id = ids[(size_t)row * 6u + j];
            if (id < 0 || id >= 384) return 0;
            for (uint32_t k = 0; k < j; ++k)
                if (id == ids[(size_t)row * 6u + k]) return 0;
        }
    }
    const uint32_t first = tp_rank * 192u;
    const char *d = cuda_model_range_ptr(model_map, down_offset + first * expert_bytes,
                                         192u * expert_bytes, "V4.1 TP owned bulk down");
    if (!d) return 0;
    /* Counts, offsets, spare hot-list storage, then original six-slot pair IDs. */
    constexpr uint32_t offsets_at = 192u, hot_at = 385u, pairs_at = 578u;
    std::vector<uint32_t> metadata(pairs_at + (size_t)pairs, 0u);
    uint32_t pos = 0;
    for (uint32_t e = 0; e < 192u; ++e) {
        metadata[offsets_at + e] = pos;
        for (uint32_t p = 0; p < pairs; ++p)
            if ((uint32_t)ids[p] == first + e) {
                metadata[pairs_at + pos++] = p;
                ++metadata[e];
            }
    }
    metadata[offsets_at + 192u] = pos;
    ds4_gpu_tensor *meta = ds4_gpu_tensor_alloc(metadata.size() * sizeof(uint32_t));
    ds4_gpu_tensor *mid_h = ds4_gpu_tensor_alloc(mids * sizeof(half));
    int ok = meta && mid_h;
    if (ok) ok = ds4_gpu_tensor_write(meta, 0, metadata.data(), metadata.size() * sizeof(uint32_t)) &&
        cuda_ok(cudaMemset(scratch->ptr, 0, pairs * 5120u * sizeof(half)),
                "V4.1 TP clear unowned down slots");
    if (ok) {
        f32_to_f16_kernel<<<(uint32_t)((mids + 255u) / 256u), 256>>>(
            (half *)mid_h->ptr, (const float *)mid->ptr, mids);
        ok = cuda_ok(cudaGetLastError(), "V4.1 TP down F16 mid");
    }
    if (ok) {
        uint32_t *m = (uint32_t *)meta->ptr;
        ok = routed_moe_q2_float_down_launch(out, scratch, mid,
            (const half *)mid_h->ptr, !g_quality_mode, d, m, m + offsets_at,
            m + pairs_at, m + hot_at, n_tokens, 192u, 6u, 2304u, 5120u,
            expert_bytes, 756u);
    }
    if (!ds4_gpu_synchronize()) ok = 0;
    ds4_gpu_tensor_free(mid_h);
    ds4_gpu_tensor_free(meta);
    return ok;
}

/* Isolated bulk operator: remap owned IDs to a contiguous192-expert table.
 * INT_MAX is a nonmatching sentinel in mm_ids_helper, so no unowned expert
 * contributes an assignment. Cleared output rows remain zero for those slots. */
extern "C" int ds4_gpu_dsv41_moe_tp_gate_up(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *x,
        uint32_t n_tokens, uint32_t tp_rank) {
    constexpr uint64_t expert_bytes = UINT64_C(2304) * 1320u;
    const uint64_t pairs = (uint64_t)n_tokens * 6u;
    const uint64_t output_bytes = pairs * 2304u * sizeof(float);
    if (!g_deepseek41_model || !ds4_rocm_is_gfx1151() || tp_rank > 1u ||
        n_tokens < 128u || n_tokens > 2048u || !gate || !up || !selected || !x ||
        !model_map || gate->bytes < output_bytes || up->bytes < output_bytes ||
        selected->bytes < pairs * sizeof(int32_t) ||
        x->bytes < (uint64_t)n_tokens * 5120u * sizeof(float) ||
        gate_offset > model_size || 384u * expert_bytes > model_size - gate_offset ||
        up_offset > model_size || 384u * expert_bytes > model_size - up_offset) return 0;
    std::vector<int32_t> ids((size_t)pairs);
    if (!ds4_gpu_tensor_read(selected, 0, ids.data(), pairs * sizeof(int32_t))) return 0;
    for (uint32_t row = 0; row < n_tokens; ++row) {
        for (uint32_t j = 0; j < 6u; ++j) {
            const int32_t id = ids[(size_t)row * 6u + j];
            if (id < 0 || id >= 384) return 0;
            for (uint32_t k = 0; k < j; ++k)
                if (id == ids[(size_t)row * 6u + k]) return 0;
        }
    }
    const uint32_t first = tp_rank * 192u;
    for (int32_t &id : ids)
        id = (uint32_t)id >= first && (uint32_t)id < first + 192u ? id - first : INT_MAX;
    const char *g = cuda_model_range_ptr(model_map, gate_offset + first * expert_bytes,
                                         192u * expert_bytes, "V4.1 TP MMQ owned gate");
    const char *u = cuda_model_range_ptr(model_map, up_offset + first * expert_bytes,
                                         192u * expert_bytes, "V4.1 TP MMQ owned up");
    if (!g || !u) return 0;
    ds4_gpu_tensor *local_ids = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    if (!local_ids) return 0;
    int ok = ds4_gpu_tensor_write(local_ids, 0, ids.data(), pairs * sizeof(int32_t)) &&
        ds4_gpu_tensor_fill_f32(gate, 0.f, pairs * 2304u) &&
        ds4_gpu_tensor_fill_f32(up, 0.f, pairs * 2304u) && ds4_mmq_init(0) == 0;
    if (ok) ok = ds4_mmq_iq2_xxs_moe_pair(g, u, (const float *)x->ptr,
        (const int32_t *)local_ids->ptr, (float *)gate->ptr, (float *)up->ptr,
        2304, 5120, (int)n_tokens, 192, 6, (cudaStream_t)0) == 0;
    if (!ds4_gpu_synchronize()) ok = 0;
    ds4_gpu_tensor_free(local_ids);
    return ok;
}

/* Average the HC copies entering each DSpark target layer directly into the
 * interleaved main-feature buffer. */
__global__ static void v41_hc_mean_kernel(const float *stream, float *out,
                                          uint32_t rows, uint32_t dim,
                                          uint32_t hc, uint32_t out_stride,
                                          uint32_t out_off) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = (uint32_t)(i / dim), col = (uint32_t)(i % dim);
    if (row >= rows) return;
    float acc = 0.0f;
    for (uint32_t c = 0; c < hc; c++)
        acc += stream[((uint64_t)row * hc + c) * dim + col];
    out[(uint64_t)row * out_stride + out_off + col] = acc / (float)hc;
}

extern "C" int ds4_gpu_dsv41_hc_mean(
        uint32_t rows, uint32_t dim, uint32_t hc,
        const ds4_gpu_tensor *stream, ds4_gpu_tensor *out,
        uint32_t out_stride, uint32_t out_off) {
    const uint64_t in_count = (uint64_t)rows * hc * dim;
    const uint64_t out_count = (uint64_t)rows * out_stride;
    if (!rows || !dim || !hc || out_stride < dim ||
        out_off > out_stride - dim || !stream || !out ||
        !stream->ptr || !out->ptr || in_count > stream->bytes / sizeof(float) ||
        out_count > out->bytes / sizeof(float)) return 0;
    v41_hc_mean_kernel<<<(unsigned)(((uint64_t)rows * dim + 255u) / 256u), 256>>>(
        (const float *)stream->ptr, (float *)out->ptr,
        rows, dim, hc, out_stride, out_off);
    return cuda_ok(cudaGetLastError(), "V4.1 stream mean");
}

__device__ static float v41_markov_value(const void *p, uint64_t i, int f16) {
    return f16 ? __half2float(((const half *)p)[i]) : ((const float *)p)[i];
}

__global__ static void v41_markov_transpose_f16_kernel(
        half *out, const half *in, uint32_t rows, uint32_t cols) {
    __shared__ half tile[32][33];
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    const uint32_t in_col = blockIdx.x * 32u + tx;
#pragma unroll
    for (uint32_t j = 0; j < 32u; j += 8u) {
        const uint32_t in_row = blockIdx.y * 32u + ty + j;
        if (in_col < cols && in_row < rows)
            tile[ty + j][tx] = in[(uint64_t)in_row * cols + in_col];
    }
    __syncthreads();
    const uint32_t out_col = blockIdx.y * 32u + tx;
#pragma unroll
    for (uint32_t j = 0; j < 32u; j += 8u) {
        const uint32_t out_row = blockIdx.x * 32u + ty + j;
        if (out_row < cols && out_col < rows)
            out[(uint64_t)out_row * rows + out_col] = tile[tx][ty + j];
    }
}

extern "C" int ds4_gpu_dsv41_markov_prepare_head(
        ds4_gpu_tensor *transposed, const void *model_map,
        uint64_t model_size, uint64_t head_offset,
        uint32_t vocab, uint32_t rank) {
    const uint64_t bytes = (uint64_t)vocab * rank * sizeof(half);
    if (!transposed || !transposed->ptr || transposed->bytes < bytes ||
        !model_map || !vocab || !rank ||
        !cuda_model_range_fits(model_size, head_offset, bytes)) return 0;
    const half *head = (const half *)cuda_model_range_ptr(
        model_map, head_offset, bytes, "V4.1 Markov head transpose");
    if (!head) return 0;
    v41_markov_transpose_f16_kernel<<<
        dim3((rank + 31u) / 32u, (vocab + 31u) / 32u), dim3(32u, 8u)>>>(
            (half *)transposed->ptr, head, vocab, rank);
    return cuda_ok(cudaGetLastError(), "V4.1 Markov head transpose");
}

__global__ static void v41_markov_part_kernel(
        const float *logits, const void *embed, const void *head,
        const half *head_t,
        const int *tokens, float *part_val, int *part_idx,
        uint32_t vocab, uint32_t rank, uint32_t step,
        uint32_t n_parts, int f16) {
    extern __shared__ float embedded[];
    const int prev = tokens[step];
    for (uint32_t r = threadIdx.x; r < rank; r += blockDim.x)
        embedded[r] = v41_markov_value(embed, (uint64_t)prev * rank + r, f16);
    __syncthreads();
    const float *lg = logits + (uint64_t)step * vocab;
    float best = -INFINITY;
    int best_id = -1;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
         v < vocab; v += n_parts * blockDim.x) {
        float score = lg[v];
        for (uint32_t r = 0; r < rank; r++)
            score = fmaf(head_t ? __half2float(head_t[(uint64_t)r * vocab + v]) :
                         v41_markov_value(head, (uint64_t)v * rank + r, f16),
                         embedded[r], score);
        if (score > best || (score == best && (best_id < 0 || (int)v < best_id))) {
            best = score;
            best_id = (int)v;
        }
    }
    __shared__ float values[256];
    __shared__ int ids[256];
    values[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t k = 1; k < blockDim.x; k++)
            if (ids[k] >= 0 &&
                (values[k] > best ||
                 (values[k] == best && (best_id < 0 || ids[k] < best_id)))) {
                best = values[k];
                best_id = ids[k];
            }
        part_val[blockIdx.x] = best;
        part_idx[blockIdx.x] = best_id;
    }
}

/* Provisional lossless F32 sidecar path; the published F16 kernels stay intact. */
extern "C" int ds4_gpu_dsv41_markov_f32_head_available(void) {
    return ds4_rocm_is_gfx1151();
}

__global__ static void v41_markov_transpose_f32_kernel(
        float *out, const float *in, uint32_t rows, uint32_t cols) {
    __shared__ float tile[32][33];
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    const uint32_t in_col = blockIdx.x * 32u + tx;
#pragma unroll
    for (uint32_t j = 0; j < 32u; j += 8u) {
        const uint32_t in_row = blockIdx.y * 32u + ty + j;
        if (in_col < cols && in_row < rows)
            tile[ty + j][tx] = in[(uint64_t)in_row * cols + in_col];
    }
    __syncthreads();
    const uint32_t out_col = blockIdx.y * 32u + tx;
#pragma unroll
    for (uint32_t j = 0; j < 32u; j += 8u) {
        const uint32_t out_row = blockIdx.x * 32u + ty + j;
        if (out_row < cols && out_col < rows)
            out[(uint64_t)out_row * rows + out_col] = tile[tx][ty + j];
    }
}

extern "C" int ds4_gpu_dsv41_markov_prepare_head_f32(
        ds4_gpu_tensor *transposed, const void *model_map,
        uint64_t model_size, uint64_t head_offset,
        uint32_t vocab, uint32_t rank) {
    if (vocab != 129280u || rank != 256u ||
        !ds4_gpu_dsv41_markov_f32_head_available()) return 0;
    const uint64_t bytes = (uint64_t)vocab * rank * sizeof(float);
    if (!transposed || !transposed->ptr || transposed->bytes < bytes ||
        (uintptr_t)transposed->ptr % alignof(float) ||
        !model_map || head_offset % alignof(float) ||
        !cuda_model_range_fits(model_size, head_offset, bytes)) return 0;
    const float *head = (const float *)cuda_model_range_ptr(
        model_map, head_offset, bytes, "V4.1 F32 Markov head transpose");
    if (!head) return 0;
    v41_markov_transpose_f32_kernel<<<
        dim3((rank + 31u) / 32u, (vocab + 31u) / 32u), dim3(32u, 8u)>>>(
            (float *)transposed->ptr, head, vocab, rank);
    return cuda_ok(cudaGetLastError(), "V4.1 F32 Markov head transpose");
}

__global__ static void v41_markov_part_f32_transposed_kernel(
        const float *logits, const float *embed, const float *head_t,
        const int *tokens, float *part_val, int *part_idx,
        uint32_t vocab, uint32_t rank, uint32_t step,
        uint32_t n_parts) {
    extern __shared__ float embedded[];
    const int prev = tokens[step];
    for (uint32_t r = threadIdx.x; r < rank; r += blockDim.x)
        embedded[r] = embed[(uint64_t)prev * rank + r];
    __syncthreads();
    const float *lg = logits + (uint64_t)step * vocab;
    float best = -INFINITY;
    int best_id = -1;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
         v < vocab; v += n_parts * blockDim.x) {
        float score = lg[v];
        for (uint32_t r = 0; r < rank; r++)
            score = fmaf(head_t[(uint64_t)r * vocab + v],
                         embedded[r], score);
        if (score > best || (score == best && (best_id < 0 || (int)v < best_id))) {
            best = score;
            best_id = (int)v;
        }
    }
    __shared__ float values[256];
    __shared__ int ids[256];
    values[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t k = 1; k < blockDim.x; k++)
            if (ids[k] >= 0 &&
                (values[k] > best ||
                 (values[k] == best && (best_id < 0 || ids[k] < best_id)))) {
                best = values[k];
                best_id = ids[k];
            }
        part_val[blockIdx.x] = best;
        part_idx[blockIdx.x] = best_id;
    }
}

__global__ static void v41_markov_final_kernel(
        const float *part_val, const int *part_idx, int *tokens,
        const float *x, const void *embed, const float *conf_proj,
        float *conf, uint32_t rank, uint32_t dim, uint32_t step,
        uint32_t n_parts, int f16) {
    __shared__ float values[256];
    __shared__ int ids[256];
    float best = -INFINITY;
    int best_id = -1;
    for (uint32_t p = threadIdx.x; p < n_parts; p += blockDim.x)
        if (part_idx[p] >= 0 &&
            (part_val[p] > best ||
             (part_val[p] == best && (best_id < 0 || part_idx[p] < best_id)))) {
            best = part_val[p];
            best_id = part_idx[p];
        }
    values[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t k = 1; k < blockDim.x; k++)
            if (ids[k] >= 0 &&
                (values[k] > best ||
                 (values[k] == best && (best_id < 0 || ids[k] < best_id)))) {
                best = values[k];
                best_id = ids[k];
            }
        tokens[step + 1u] = best_id < 0 ? 0 : best_id;
    }
    /* Other waves must not overwrite argmax values with confidence partials
     * while thread zero is still reading the complete block. */
    __syncthreads();
    const int prev = tokens[step];
    float sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x)
        sum = fmaf(conf_proj[d], x[(uint64_t)step * dim + d], sum);
    for (uint32_t r = threadIdx.x; r < rank; r += blockDim.x)
        sum = fmaf(conf_proj[dim + r],
            v41_markov_value(embed, (uint64_t)prev * rank + r, f16), sum);
    values[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (uint32_t k = 0; k < blockDim.x; k++) total += values[k];
        conf[step] = total;
    }
}

/* Preselected actual-weight variant9: F16 transposed head, 256 threads,
 * ordered rank FMAs and tree argmax. Confidence arithmetic is unchanged. */
__device__ static bool v41_markov_tree_better(float b,int bi,float a,int ai) {
 return bi>=0 && (b>a || (b==a && (ai<0 || bi<ai)));
}
template<int THREADS,int U,bool TREE,bool WRITE>
__global__ __launch_bounds__(THREADS) void v41_markov_f16_tree_part(
 const float* logits,const half* embed,const half* head_t,const int* tokens,
 float* pv,int* pi,unsigned vocab,unsigned rank,unsigned step,unsigned parts,float* scores) {
 extern __shared__ float embedded[];
 for(unsigned r=threadIdx.x;r<rank;r+=THREADS)
  embedded[r]=__half2float(embed[(uint64_t)tokens[step]*rank+r]);
 __syncthreads();
 float best=-INFINITY;int id=-1;
 for(unsigned v=blockIdx.x*THREADS+threadIdx.x;v<vocab;v+=parts*THREADS) {
  float score=logits[(uint64_t)step*vocab+v];
  for(unsigned base=0;base<rank;base+=U) {
   float w[U];
#pragma unroll
   for(unsigned j=0;j<U;j++)
    if(base+j<rank)w[j]=__half2float(head_t[(uint64_t)(base+j)*vocab+v]);
#pragma unroll
   for(unsigned j=0;j<U;j++)
    if(base+j<rank)score=fmaf(w[j],embedded[base+j],score);
  }
  if constexpr(WRITE)scores[(uint64_t)step*vocab+v]=score;
  if(v41_markov_tree_better(score,(int)v,best,id)){best=score;id=(int)v;}
 }
 __shared__ float values[THREADS];__shared__ int ids[THREADS];
 values[threadIdx.x]=best;ids[threadIdx.x]=id;__syncthreads();
 if constexpr(TREE) {
  for(unsigned d=THREADS/2;d;d>>=1) {
   if(threadIdx.x<d && v41_markov_tree_better(values[threadIdx.x+d],ids[threadIdx.x+d],
                             values[threadIdx.x],ids[threadIdx.x])) {
    values[threadIdx.x]=values[threadIdx.x+d];ids[threadIdx.x]=ids[threadIdx.x+d];
   }
   __syncthreads();
  }
  if(threadIdx.x==0){pv[blockIdx.x]=values[0];pi[blockIdx.x]=ids[0];}
 } else if(threadIdx.x==0) {
  for(unsigned k=1;k<THREADS;k++)if(v41_markov_tree_better(values[k],ids[k],best,id)){
   best=values[k];id=ids[k];
  }
  pv[blockIdx.x]=best;pi[blockIdx.x]=id;
 }
}

template<bool TREE>
__global__ void v41_markov_f16_tree_final(const float* pv,const int* pi,int* tokens,const float* x,
 const half* embed,const float* proj,float* conf,unsigned rank,unsigned dim,
 unsigned step,unsigned parts) {
 __shared__ float values[256];__shared__ int ids[256];
 float best=-INFINITY;int id=-1;
 for(unsigned p=threadIdx.x;p<parts;p+=256)
  if(v41_markov_tree_better(pv[p],pi[p],best,id)){best=pv[p];id=pi[p];}
 values[threadIdx.x]=best;ids[threadIdx.x]=id;__syncthreads();
 if constexpr(TREE) {
  for(unsigned d=128;d;d>>=1) {
   if(threadIdx.x<d && v41_markov_tree_better(values[threadIdx.x+d],ids[threadIdx.x+d],
                             values[threadIdx.x],ids[threadIdx.x])){
    values[threadIdx.x]=values[threadIdx.x+d];ids[threadIdx.x]=ids[threadIdx.x+d];
   }
   __syncthreads();
  }
  if(threadIdx.x==0)tokens[step+1]=ids[0]<0?0:ids[0];
 } else if(threadIdx.x==0) {
  for(unsigned k=1;k<256;k++)if(v41_markov_tree_better(values[k],ids[k],best,id)){
   best=values[k];id=ids[k];
  }
  tokens[step+1]=id<0?0:id;
 }
 // All argmax readers finish before this shared array is reused for confidence.
 __syncthreads();
 float sum=0;
 for(unsigned d=threadIdx.x;d<dim;d+=256)sum=fmaf(proj[d],x[(uint64_t)step*dim+d],sum);
 for(unsigned r=threadIdx.x;r<rank;r+=256)
  sum=fmaf(proj[dim+r],__half2float(embed[(uint64_t)tokens[step]*rank+r]),sum);
 values[threadIdx.x]=sum;__syncthreads();
 if(threadIdx.x==0){
  float total=0;for(unsigned k=0;k<256;k++)total+=values[k];conf[step]=total;
 }
}

extern "C" int ds4_gpu_dsv41_markov_chain(
        uint32_t block, uint32_t vocab, uint32_t rank, uint32_t dim,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t embed_offset, uint64_t head_offset, int f16,
        const ds4_gpu_tensor *transposed_head,
        const ds4_gpu_tensor *conf_proj, ds4_gpu_tensor *tokens,
        ds4_gpu_tensor *conf, ds4_gpu_tensor *parts, uint32_t n_parts) {
    const bool f32_transposed = transposed_head && !f16;
    if (f32_transposed && (vocab != 129280u || rank != 256u ||
        dim != 5120u || n_parts != 1024u ||
        !ds4_gpu_dsv41_markov_f32_head_available())) return 0;
    const uint64_t weight_bytes = (uint64_t)vocab * rank * (f16 ? 2u : 4u);
    if (!block || !vocab || !rank || !dim || !n_parts || n_parts > 4096u ||
        !model_map || !logits || !x || !conf_proj || !tokens || !conf || !parts ||
        !logits->ptr || !x->ptr || !conf_proj->ptr || !tokens->ptr ||
        !conf->ptr || !parts->ptr ||
        (transposed_head && (!transposed_head->ptr ||
            transposed_head->bytes < weight_bytes ||
            (f32_transposed && (uintptr_t)transposed_head->ptr % alignof(float)))) ||
        !cuda_model_range_fits(model_size, embed_offset, weight_bytes) ||
        !cuda_model_range_fits(model_size, head_offset, weight_bytes) ||
        (uint64_t)block * vocab > logits->bytes / sizeof(float) ||
        (uint64_t)block * dim > x->bytes / sizeof(float) ||
        (uint64_t)dim + rank > conf_proj->bytes / sizeof(float) ||
        (uint64_t)block + 1u > tokens->bytes / sizeof(int32_t) ||
        block > conf->bytes / sizeof(float) ||
        (uint64_t)n_parts * 2u > parts->bytes / sizeof(float)) return 0;
    const void *embed = cuda_model_range_ptr(model_map, embed_offset,
        weight_bytes, "V4.1 Markov embed");
    const void *head = cuda_model_range_ptr(model_map, head_offset,
        weight_bytes, "V4.1 Markov head");
    if (!embed || !head) return 0;
    float *part_val = (float *)parts->ptr;
    int *part_idx = (int *)(part_val + n_parts);
    const half *head_t = transposed_head && f16 ?
        (const half *)transposed_head->ptr : NULL;
    const uint32_t launch_parts = head_t && n_parts > (vocab + 255u) / 256u ?
        (vocab + 255u) / 256u : n_parts;
    const char *tree_env = getenv("DS4_ROCM_V41_MARKOV_TREE");
    const bool f16_tree = (!tree_env || strcmp(tree_env, "1") == 0) &&
        ds4_rocm_is_gfx1151() && f16 && head_t &&
        vocab == 129280u && rank == 256u && dim == 5120u;
    for (uint32_t step = 0; step < block; step++) {
        if (f16_tree) {
            v41_markov_f16_tree_part<256, 1, true, false>
                <<<launch_parts, 256, rank * sizeof(float)>>>(
                    (const float *)logits->ptr, (const half *)embed, head_t,
                    (const int *)tokens->ptr, part_val, part_idx,
                    vocab, rank, step, launch_parts, nullptr);
            v41_markov_f16_tree_final<true><<<1, 256>>>(
                part_val, part_idx, (int *)tokens->ptr, (const float *)x->ptr,
                (const half *)embed, (const float *)conf_proj->ptr,
                (float *)conf->ptr, rank, dim, step, launch_parts);
            continue;
        }
        if (f32_transposed) {
            v41_markov_part_f32_transposed_kernel<<<n_parts, 256, rank * sizeof(float)>>>(
                (const float *)logits->ptr, (const float *)embed,
                (const float *)transposed_head->ptr, (const int *)tokens->ptr,
                part_val, part_idx, vocab, rank, step, n_parts);
        } else {
            v41_markov_part_kernel<<<launch_parts, 256, rank * sizeof(float)>>>(
                (const float *)logits->ptr, embed, head, head_t,
                (const int *)tokens->ptr, part_val, part_idx, vocab, rank, step,
                launch_parts, f16);
        }
        v41_markov_final_kernel<<<1, 256>>>(part_val, part_idx,
            (int *)tokens->ptr, (const float *)x->ptr, embed,
            (const float *)conf_proj->ptr, (float *)conf->ptr,
            rank, dim, step, launch_parts, f16);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 Markov chain");
}

__global__ static void rms_norm_weight_bf16_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = v41_bf16(xr[i] * scale * w[i]);
    }
}

extern "C" int ds4_gpu_dsv41_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    uint64_t weight_bytes = 0;
    if (!model_map || !cuda_u64_mul_checked(n, sizeof(float), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(out, n, rows, sizeof(float)) ||
        !cuda_tensor_has_elems2(x, n, rows, sizeof(float))) return 0;
    if (n == 0u || rows == 0u) return 1;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_bf16_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 norm BF16 launch");
}

#endif
