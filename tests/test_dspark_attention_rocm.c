/* Production-size typed draft attention projections with a sparse CPU oracle.
 * Include the graph to test its actual dispatch and workspace strides. */
#include "../ds4.c"
#include <assert.h>

enum { GUARD = 64 };
typedef struct {
    ds4_gpu_tensor *storage, *view;
    size_t bytes;
} guarded;

static guarded alloc_guarded(size_t bytes, const void *data) {
    guarded a = {.bytes = bytes};
    uint8_t guard[GUARD];
    memset(guard, 0xa5, sizeof(guard));
    a.storage = ds4_gpu_tensor_alloc(bytes + 2 * GUARD);
    assert(a.storage);
    a.view = ds4_gpu_tensor_view(a.storage, GUARD, bytes);
    assert(a.view);
    assert(ds4_gpu_tensor_write(a.storage, 0, guard, GUARD));
    assert(ds4_gpu_tensor_write(a.storage, GUARD + bytes, guard, GUARD));
    if (data) assert(ds4_gpu_tensor_write(a.view, 0, data, bytes));
    return a;
}

static void check_guard(guarded a) {
    uint8_t before[GUARD], after[GUARD];
    assert(ds4_gpu_tensor_read(a.storage, 0, before, GUARD));
    assert(ds4_gpu_tensor_read(a.storage, GUARD + a.bytes, after, GUARD));
    for (unsigned i = 0; i < GUARD; i++) assert(before[i] == 0xa5 && after[i] == 0xa5);
}

static float oracle_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    bits &= 0xffff0000;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static void put_weight(uint8_t *data, size_t index, bool f16) {
    if (f16) {
        uint16_t half = 0x3800; /* Exactly 0.5. */
        memcpy(data + index * 2, &half, 2);
    } else {
        float value = 0.5f;
        memcpy(data + index * 4, &value, 4);
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    const bool f16 = strcmp(argv[1], "f16") == 0;
    assert(f16 || strcmp(argv[1], "f32") == 0);
    const unsigned rows = (unsigned)atoi(argv[2]);
    assert(rows == 1 || rows == 5);
    g_ds4_shape.n_embd = 5120;
    g_ds4_shape.family = DS4_MODEL_FAMILY_DEEPSEEK41;
    const unsigned groups = 8, width = 4096, low = 1024, outputs = 5120;
    const unsigned item = f16 ? 2 : 4;
    const size_t a_bytes = (size_t)groups * low * width * item;
    const size_t b_bytes = (size_t)outputs * groups * low * item;
    uint8_t *weights = calloc(1, a_bytes + b_bytes);
    float *heads = malloc((size_t)rows * groups * width * sizeof(float));
    float *got_low = malloc((size_t)rows * groups * low * sizeof(float));
    float *got = malloc((size_t)rows * outputs * sizeof(float));
    assert(weights && heads && got_low && got);
    for (unsigned group = 0; group < groups; group++)
        for (unsigned j = 0; j < low; j++)
            put_weight(weights, ((size_t)group * low + j) * width + (j * 3 + group * 17) % width, f16);
    for (unsigned j = 0; j < outputs; j++)
        put_weight(weights + a_bytes, (size_t)j * groups * low + (j * 13 + 7) % (groups * low), f16);
    for (unsigned r = 0; r < rows; r++)
        for (unsigned g = 0; g < groups; g++)
            for (unsigned j = 0; j < width; j++)
                heads[((size_t)r * groups + g) * width + j] =
                    (float)((int)((r * 211 + g * 97 + j * 7) % 1009) - 504) / 256.0f;
    ds4_tensor a = {.type = f16 ? DS4_TENSOR_F16 : DS4_TENSOR_F32,
                    .ndim = 2, .dim = {width, groups * low}, .bytes = a_bytes};
    ds4_tensor b = {.type = a.type, .ndim = 2, .dim = {groups * low, outputs},
                    .abs_offset = a_bytes, .bytes = b_bytes};
    ds4_model model = {.map = weights, .size = a_bytes + b_bytes};
    ds4_layer_weights layer = {.attn_output_a = &a, .attn_output_b = &b};
    assert(ds4_gpu_init());
    guarded input = alloc_guarded((size_t)rows * groups * width * 4, heads);
    guarded narrow = alloc_guarded((size_t)rows * groups * low * 4, NULL);
    guarded output = alloc_guarded((size_t)rows * outputs * 4, NULL);
    ds41_gpu_graph graph = {.tp_world = 1};
    graph.batch.heads = input.view;
    graph.batch.low = narrow.view;
    graph.batch.block = output.view;
    /* Intentionally no target rows_view: draft storage has its own full strides. */
    assert(ds41_draft_attention_output(&graph, &model, &layer, rows));
    assert(ds4_gpu_synchronize());
    assert(ds4_gpu_tensor_read(narrow.view, 0, got_low, narrow.bytes));
    assert(ds4_gpu_tensor_read(output.view, 0, got, output.bytes));
    for (unsigned r = 0; r < rows; r++) {
        for (unsigned g = 0; g < groups; g++)
            for (unsigned j = 0; j < low; j++) {
                const float expected = oracle_bf16(0.5f * heads[
                    ((size_t)r * groups + g) * width + (j * 3 + g * 17) % width]);
                assert(got_low[((size_t)r * groups + g) * low + j] == expected);
            }
        for (unsigned j = 0; j < outputs; j++) {
            const unsigned col = (j * 13 + 7) % (groups * low);
            const unsigned g = col / low, k = col % low;
            const float expected = 0.5f * oracle_bf16(0.5f * heads[
                ((size_t)r * groups + g) * width + (k * 3 + g * 17) % width]);
            assert(got[(size_t)r * outputs + j] == expected);
        }
    }
    check_guard(input); check_guard(narrow); check_guard(output);
    printf("DSpark attention %s rows=%u: all low/output values and guards passed\n", argv[1], rows);
    return 0;
}
