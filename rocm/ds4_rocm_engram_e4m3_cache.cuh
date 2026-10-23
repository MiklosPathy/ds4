/* Exact, optional Engram copies. Original F16 mappings remain available. */
namespace ds41_engram_e4m3 {
static constexpr uint32_t width = 6144u, outputs = 25600u;
static constexpr uint64_t count = (uint64_t)width * outputs;
static constexpr uint64_t blocks = (width / 32u) * (outputs / 32u);
static constexpr uint64_t packed_bytes = count + blocks * sizeof(int32_t);
static constexpr uint64_t byte_limit = 2u * packed_bytes;
struct packed_range {
    const void *model, *original;
    uint64_t model_size, offset;
    void *device;
};
static std::vector<packed_range> ranges;
static uint64_t allocated_bytes;
static bool enabled(void) {
    const char *flag = getenv("DS4_ROCM_V41_ENGRAM_E4M3");
    return (!flag || !strcmp(flag, "1")) && g_deepseek41_model &&
        !g_quality_mode && ds4_rocm_is_gfx1151();
}

/* F16 values become normal finite F32 or signed zero. This encoder retains
 * every bit; it does not round an ineligible value into the representation. */
__device__ static bool encode(uint32_t bits, int scale, uint8_t *code) {
    const uint32_t magnitude = bits & 0x7fffffffu;
    const uint32_t sign = (bits >> 24u) & 128u;
    if (!magnitude) { *code = (uint8_t)sign; return true; }
    const uint32_t exponent = magnitude >> 23u;
    if (exponent == 0u || exponent == 255u) return false;
    const int e = (int)exponent - 127 - scale;
    const uint32_t fraction = magnitude & 0x7fffffu;
    if (e >= -6 && e <= 8 && !(fraction & 0xfffffu)) {
        const uint32_t c = (uint32_t)(e + 7) * 8u + (fraction >> 20u);
        if (c > 126u) return false;
        *code = (uint8_t)(sign | c); return true;
    }
    if (e < -9 || e >= -6) return false;
    const uint32_t significand = 0x800000u | fraction;
    const unsigned shift = (unsigned)(14 - e);
    if (significand & ((1u << shift) - 1u)) return false;
    const uint32_t c = significand >> shift;
    if (!c || c > 7u) return false;
    *code = (uint8_t)(sign | c); return true;
}

/* One CTA owns one32x32 block. The scale search matches the independent
 * fixture packer: ceil(log2(maxabs/448)), then +1 and -1 if needed. */
__global__ static void pack(uint8_t *codes, int32_t *scales, uint32_t *invalid,
        const __half *weights) {
    __shared__ uint32_t maximum, bad;
    const unsigned tid = threadIdx.x, block = blockIdx.x;
    const unsigned row0 = (block / (width / 32u)) * 32u;
    const unsigned col0 = (block % (width / 32u)) * 32u;
    uint32_t values[8];
    if (!tid) { maximum = 0u; bad = 0u; }
    __syncthreads();
    uint32_t local_max = 0u;
#pragma unroll
    for (unsigned j = 0; j < 8u; j++) {
        const unsigned i = tid + j * 128u;
        const uint64_t index = (uint64_t)(row0 + i / 32u) * width + col0 + i % 32u;
        values[j] = __float_as_uint(__half2float(weights[index]));
        const uint32_t magnitude = values[j] & 0x7fffffffu;
        if ((magnitude >> 23u) == 255u) atomicOr(&bad, 1u);
        local_max = local_max > magnitude ? local_max : magnitude;
    }
    atomicMax(&maximum, local_max);
    __syncthreads();
    if (bad) { if (!tid) atomicOr(invalid, 1u); return; }
    int initial = maximum ? (int)(maximum >> 23u) - 127 - 8 : 0;
    if (maximum && (maximum & 0x7fffffu) > 0x600000u) initial++;
    for (unsigned trial = 0; trial < 3u; trial++) {
        const int scale = initial + (trial == 1u ? 1 : trial == 2u ? -1 : 0);
        if (!tid) bad = 0u;
        __syncthreads();
        uint8_t encoded[8];
        bool ok = true;
#pragma unroll
        for (unsigned j = 0; j < 8u; j++) ok = encode(values[j], scale, &encoded[j]) && ok;
        if (!ok) atomicOr(&bad, 1u);
        __syncthreads();
        if (!bad) {
            if (!tid) scales[block] = scale;
#pragma unroll
            for (unsigned j = 0; j < 8u; j++) {
                const unsigned i = tid + j * 128u;
                codes[(uint64_t)(row0 + i / 32u) * width + col0 + i % 32u] = encoded[j];
            }
            return;
        }
        __syncthreads();
    }
    if (!tid) atomicOr(invalid, 1u);
}

static void clear(void) {
    for (const auto &r : ranges) if (r.device) (void)cudaFree(r.device);
    ranges.clear(); allocated_bytes = 0;
}
static const packed_range *get(const void *model, uint64_t model_size,
        uint64_t offset, const __half *weights, bool *failed) {
    *failed = false;
    if (!enabled() || !model || !weights || offset > model_size ||
        count * sizeof(__half) > model_size - offset) return nullptr;
    hipStreamCaptureStatus capture;
    cudaError_t error = hipStreamIsCapturing(nullptr, &capture);
    if (error != cudaSuccess) {
        cuda_ok(error, "Engram E4M3 capture status"); *failed = true; return nullptr;
    }
    // Neither allocation, validation synchronization nor insertion during capture.
    if (capture != hipStreamCaptureStatusNone) return nullptr;
    for (const auto &r : ranges)
        if (r.model == model && r.model_size == model_size && r.offset == offset &&
            r.original == weights) return r.device ? &r : nullptr;
    if (ranges.size() >= 8u || packed_bytes > byte_limit - allocated_bytes) return nullptr;
    void *device = nullptr;
    uint32_t *bad = nullptr, invalid = 1u;
    error = cudaMalloc(&device, (size_t)packed_bytes);
    if (error == cudaSuccess) error = cudaMalloc((void **)&bad, sizeof(*bad));
    if (error == hipErrorOutOfMemory) {
        (void)cudaGetLastError();
    } else if (error == cudaSuccess) {
        error = cudaMemset(bad, 0, sizeof(*bad));
        if (error == cudaSuccess) {
            pack<<<(uint32_t)blocks, 128u>>>((uint8_t *)device,
                (int32_t *)((char *)device + count), bad, weights);
            error = cudaGetLastError();
        }
        if (error == cudaSuccess) error = cudaMemcpy(&invalid, bad, sizeof(invalid), cudaMemcpyDeviceToHost);
    }
    if (bad) (void)cudaFree(bad);
    if (error != cudaSuccess || invalid) {
        if (device) (void)cudaFree(device);
        device = nullptr;
    }
    if (error != cudaSuccess && error != hipErrorOutOfMemory) {
        cuda_ok(error, "Engram E4M3 preparation"); *failed = true; return nullptr;
    }
    if (device) allocated_bytes += packed_bytes;
    ranges.push_back({model, weights, model_size, offset, device});
    const char *trace = getenv("DS4_ROCM_V41_ENGRAM_E4M3_TRACE");
    if (trace && !strcmp(trace, "1")) fprintf(stderr,
        DS4_GPU_LOG_PREFIX "Engram E4M3 usable=%u bytes=%llu total=%llu\n",
        (unsigned)(device != nullptr), (unsigned long long)(device ? packed_bytes : 0u),
        (unsigned long long)allocated_bytes);
    return device ? &ranges.back() : nullptr;
}
} // namespace ds41_engram_e4m3
