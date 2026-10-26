/* --tensor-parallel3 kernel parity on synthetic V4.1 shapes.
 *
 * Attention output: the three rank partials over groups 0-2, 3-5 and 6-7
 * must reproduce the eight-group low projection bit for bit and the full
 * output projection within float reassociation error.
 * Routed Q4_K MoE: the three owned-expert partials must sum to the full
 * routed output across the decode, small-batch, tiled and atomic paths.
 * Shared-expert owner: exactly the rank with the fewest selected routed
 * experts (ties from the layer's rank) computes the shared expert, with the
 * same bits for two and three ranks; the two-rank choice is unchanged. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t rng_state = 12345u;
static uint32_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float frand(void) { return (float)(rng() % 20001u) / 10000.0f - 1.0f; }

static uint16_t f16_bits(float f) {
    _Float16 h = (_Float16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (t && data && !ds4_gpu_tensor_write(t, 0, data, bytes)) {
        ds4_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

static double rel_error(const float *got, const float *want, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) return INFINITY;
        const double d = (double)got[i] - want[i];
        num += d * d;
        den += (double)want[i] * want[i];
    }
    return sqrt(num / (den > 0 ? den : 1));
}

/* ---------------------------------------------------------------- attention */

static int attention(unsigned n) {
    enum { K = 4096, R = 1024, M = 5120 };
    const size_t ab = (size_t)8 * R * (K / 32) * 34, bb = (size_t)M * (8192 / 32) * 34;
    const size_t wb = ab + bb;
    unsigned char *model = NULL;
    CHECK(!posix_memalign((void **)&model, 4096, wb));
    for (size_t off = 0; off < wb; off += 34) {
        uint16_t d = f16_bits(0.001f + 0.002f * (float)(rng() % 16u));
        memcpy(model + off, &d, 2);
        for (unsigned j = 0; j < 32; j++) model[off + 2 + j] = (unsigned char)(rng() & 0xff);
    }
    const size_t nx = (size_t)n * 8 * K;
    float *x = malloc(nx * 4), *full_low = malloc((size_t)n * 8 * R * 4),
          *full_out = malloc((size_t)n * M * 4), *sum = calloc((size_t)n * M, 4),
          *part = malloc((size_t)n * M * 4), *low = malloc((size_t)n * 3 * R * 4),
          *xr = malloc((size_t)n * 3 * K * 4);
    CHECK(x && full_low && full_out && sum && part && low && xr);
    for (size_t i = 0; i < nx; i++) x[i] = frand();
    CHECK(ds4_gpu_set_model_map(model, wb));
    ds4_gpu_tensor *xt = upload(x, nx * 4), *lt = upload(NULL, (size_t)n * 8 * R * 4),
                   *yt = upload(NULL, (size_t)n * M * 4);
    CHECK(xt && lt && yt);
    CHECK(ds4_gpu_dsv41_attention_output_batch(yt, lt, model, wb, 0, ab, xt, n));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(lt, 0, full_low, (size_t)n * 8 * R * 4));
    CHECK(ds4_gpu_tensor_read(yt, 0, full_out, (size_t)n * M * 4));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(lt); ds4_gpu_tensor_free(yt);
    const unsigned group0[3] = {0, 3, 6}, groups[3] = {3, 3, 2};
    size_t low_diff = 0;
    for (unsigned r = 0; r < 3; r++) {
        const unsigned g0 = group0[r], gc = groups[r];
        for (unsigned t = 0; t < n; t++)
            memcpy(xr + (size_t)t * gc * K, x + ((size_t)t * 8 + g0) * K, (size_t)gc * K * 4);
        ds4_gpu_tensor *hx = upload(xr, (size_t)n * gc * K * 4),
                       *hl = upload(NULL, (size_t)n * gc * R * 4),
                       *hy = upload(NULL, (size_t)n * M * 4);
        CHECK(hx && hl && hy);
        CHECK(ds4_gpu_dsv41_attention_output_tp_groups(hy, hl, model, wb, 0, ab, hx, n, g0, gc));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(hl, 0, low, (size_t)n * gc * R * 4));
        CHECK(ds4_gpu_tensor_read(hy, 0, part, (size_t)n * M * 4));
        for (unsigned t = 0; t < n; t++)
            for (unsigned j = 0; j < gc * R; j++)
                if (memcmp(&low[(size_t)t * gc * R + j], &full_low[((size_t)t * 8 + g0) * R + j], 4))
                    low_diff++;
        for (size_t i = 0; i < (size_t)n * M; i++) sum[i] += part[i];
        /* Invalid ranges are rejected. */
        CHECK(!ds4_gpu_dsv41_attention_output_tp_groups(hy, hl, model, wb, 0, ab, hx, n, 7, 2));
        ds4_gpu_tensor_free(hx); ds4_gpu_tensor_free(hl); ds4_gpu_tensor_free(hy);
    }
    const double err = rel_error(sum, full_out, (size_t)n * M);
    fprintf(stderr, "attention rows=%u low_diff=%zu out_rel_err=%.3g\n", n, low_diff, err);
    CHECK(low_diff == 0);
    CHECK(err < 1e-4);
    ds4_gpu_cleanup();
    free(model); free(x); free(full_low); free(full_out); free(sum); free(part); free(low); free(xr);
    return 1;
}

/* --------------------------------------------------------------------- MoE */

static void fill_q4k(unsigned char *p, size_t blocks) {
    for (size_t b = 0; b < blocks; b++, p += 144) {
        uint16_t d = f16_bits(0.002f + 0.001f * (float)(rng() % 8u));
        uint16_t dmin = f16_bits(0.001f * (float)(rng() % 8u));
        memcpy(p, &d, 2);
        memcpy(p + 2, &dmin, 2);
        for (unsigned j = 4; j < 144; j++) p[j] = (unsigned char)(rng() & 0xff);
    }
}

static int moe(unsigned n) {
    enum { E = 24, USED = 6, IN = 5120, MID = 2304, OUT = 5120 };
    const uint64_t gate_row = IN / 256 * 144, down_row = MID / 256 * 144;
    const uint64_t gate_expert = gate_row * MID, down_expert = down_row * OUT;
    const size_t wb = (size_t)E * (2 * gate_expert + down_expert);
    unsigned char *model = NULL;
    CHECK(!posix_memalign((void **)&model, 4096, wb));
    fill_q4k(model, wb / 144);
    const uint64_t gate_off = 0, up_off = E * gate_expert, down_off = 2 * E * gate_expert;
    const size_t pairs = (size_t)n * USED;
    int32_t *sel = malloc(pairs * 4);
    float *w = malloc(pairs * 4), *x = malloc((size_t)n * IN * 4),
          *full = malloc((size_t)n * OUT * 4), *part = malloc((size_t)n * OUT * 4),
          *sum = calloc((size_t)n * OUT, 4);
    CHECK(sel && w && x && full && part && sum);
    for (unsigned t = 0; t < n; t++) {
        for (unsigned j = 0; j < USED; j++) {
            int32_t id;
            unsigned k;
            do {
                id = (int32_t)(rng() % E);
                for (k = 0; k < j && sel[t * USED + k] != id; k++) {}
            } while (k < j);
            sel[t * USED + j] = id;
            w[t * USED + j] = 0.1f + 0.3f * (float)(rng() % 100u) / 100.0f;
        }
    }
    for (size_t i = 0; i < (size_t)n * IN; i++) x[i] = frand();
    CHECK(ds4_gpu_set_model_map(model, wb));
    ds4_gpu_tensor *xt = upload(x, (size_t)n * IN * 4), *out = upload(NULL, (size_t)n * OUT * 4),
                   *gate = upload(NULL, pairs * MID * 4), *up = upload(NULL, pairs * MID * 4),
                   *mid = upload(NULL, pairs * MID * 4), *down = upload(NULL, pairs * OUT * 4),
                   *st = upload(sel, pairs * 4), *wt = upload(w, pairs * 4);
    CHECK(xt && out && gate && up && mid && down && st && wt);
    bool f16 = false;
    CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, wb,
        gate_off, up_off, down_off, 12, 12, gate_expert, gate_row, down_expert, down_row,
        IN, MID, OUT, st, wt, E, USED, 10.0f, xt, 0, n, &f16, true));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(out, 0, full, (size_t)n * OUT * 4));
    if (n >= 32) {
        /* Prefill uses MMQ (Q8_1 activations). One rank owning every expert
         * gives the same arithmetic as the three-rank partition. */
        float *all = malloc((size_t)n * OUT * 4);
        CHECK(all);
        CHECK(ds4_gpu_tensor_write(st, 0, sel, pairs * 4) && ds4_gpu_tensor_write(wt, 0, w, pairs * 4));
        CHECK(ds4_gpu_routed_moe_batch_owned_tensor(out, gate, up, mid, down, model, wb,
            gate_off, up_off, down_off, 12, 12, gate_expert, gate_row, down_expert, down_row,
            IN, MID, OUT, st, wt, E, USED, 0, E, 10.0f, xt, 0, n, &f16));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(out, 0, all, (size_t)n * OUT * 4));
        fprintf(stderr, "q4k moe rows=%u MMQ versus Q8_K reference rel_err=%.3g\n", n,
                rel_error(all, full, (size_t)n * OUT));
        CHECK(rel_error(all, full, (size_t)n * OUT) < 2e-2);
        memcpy(full, all, (size_t)n * OUT * 4);
        free(all);
    }
    for (unsigned r = 0; r < 3; r++) {
        /* The owned path rewrites ids and weights in place. */
        CHECK(ds4_gpu_tensor_write(st, 0, sel, pairs * 4) && ds4_gpu_tensor_write(wt, 0, w, pairs * 4));
        CHECK(ds4_gpu_routed_moe_batch_owned_tensor(out, gate, up, mid, down, model, wb,
            gate_off, up_off, down_off, 12, 12, gate_expert, gate_row, down_expert, down_row,
            IN, MID, OUT, st, wt, E, USED, r * (E / 3), E / 3, 10.0f, xt, 0, n, &f16));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(out, 0, part, (size_t)n * OUT * 4));
        for (size_t i = 0; i < (size_t)n * OUT; i++) sum[i] += part[i];
    }
    const double err = rel_error(sum, full, (size_t)n * OUT);
    fprintf(stderr, "q4k moe rows=%u rel_err=%.3g\n", n, err);
    CHECK(err < 1e-5);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
    ds4_gpu_cleanup();
    free(model); free(sel); free(w); free(x); free(full); free(part); free(sum);
    return 1;
}

/* ------------------------------------------------------------ shared owner */

/* Reference owner: fewest selected routed experts, ties from tie. */
static unsigned host_owner(const int32_t *sel, unsigned tie, unsigned world) {
    unsigned count[3] = {0, 0, 0};
    for (unsigned i = 0; i < 6; i++) count[(unsigned)sel[i] / (384u / world)]++;
    unsigned owner = tie;
    for (unsigned k = 1; k < world; k++) {
        const unsigned r = (tie + k) % world;
        if (count[r] < count[owner]) owner = r;
    }
    return owner;
}

static int shared_owner(void) {
    enum { E = 5120, F = 2304 };
    const size_t gb = (size_t)F * (E / 32) * 34, db = (size_t)E * (F / 32) * 34;
    const size_t wb = 2 * gb + db;
    unsigned char *model = NULL;
    CHECK(!posix_memalign((void **)&model, 4096, wb));
    for (size_t off = 0; off < wb; off += 34) {
        uint16_t d = f16_bits(0.001f + 0.002f * (float)(rng() % 16u));
        memcpy(model + off, &d, 2);
        for (unsigned j = 0; j < 32; j++) model[off + 2 + j] = (unsigned char)(rng() & 0xff);
    }
    float *x = malloc(E * 4), *out = malloc(E * 4), *ref = malloc(E * 4);
    CHECK(x && out && ref);
    for (unsigned i = 0; i < E; i++) x[i] = frand();
    ds4_gpu_set_deepseek41_model(true);
    CHECK(ds4_gpu_set_model_map(model, wb));
    ds4_gpu_tensor *xt = upload(x, E * 4), *ot = ds4_gpu_tensor_alloc(E * 4),
        *gt = ds4_gpu_tensor_alloc(F * 4), *ut = ds4_gpu_tensor_alloc(F * 4),
        *mt = ds4_gpu_tensor_alloc(F * 4), *st = ds4_gpu_tensor_alloc(6 * 4);
    CHECK(xt && ot && gt && ut && mt && st);
    /* Balanced, skewed, tied and single-slice routes. */
    static const int32_t routes[][6] = {
        {0, 1, 200, 201, 300, 301},   {5, 6, 7, 8, 130, 383},
        {10, 140, 270, 20, 150, 280}, {0, 1, 2, 3, 4, 5},
        {129, 130, 131, 132, 133, 134}, {191, 192, 255, 256, 383, 0},
    };
    unsigned cases = 0;
    for (unsigned r = 0; r < sizeof(routes) / sizeof(*routes); r++) {
        CHECK(ds4_gpu_tensor_write(st, 0, routes[r], sizeof(routes[r])));
        for (unsigned il = 0; il < 6; il++) {
            int have_ref = 0;
            for (unsigned world = 2; world <= 3; world++) {
                const unsigned tie = il % world, owner = host_owner(routes[r], tie, world);
                if (world == 2) {
                    /* The original two-rank rule: fewer experts below 192. */
                    unsigned left = 0;
                    for (unsigned i = 0; i < 6; i++) left += routes[r][i] < 192;
                    CHECK(owner == (left == 3 ? tie : left < 3 ? 0u : 1u));
                }
                for (unsigned rank = 0; rank < world; rank++) {
                    CHECK(ds4_gpu_dsv41_shared_owner_scalar(ot, gt, ut, mt, xt, st, model, wb,
                        0, gb, 2 * gb, rank, tie, world) == 1);
                    CHECK(ds4_gpu_synchronize());
                    CHECK(ds4_gpu_tensor_read(ot, 0, out, E * 4));
                    unsigned nonzero = 0;
                    for (unsigned i = 0; i < E; i++) nonzero += out[i] != 0.0f;
                    if (rank != owner) { CHECK(nonzero == 0); continue; }
                    CHECK(nonzero > E / 2);
                    if (!have_ref) { memcpy(ref, out, E * 4); have_ref = 1; }
                    else CHECK(!memcmp(ref, out, E * 4));
                    cases++;
                }
            }
        }
    }
    /* No owner outside the two- and three-rank layouts. */
    CHECK(ds4_gpu_dsv41_shared_owner_scalar(ot, gt, ut, mt, xt, st, model, wb,
        0, gb, 2 * gb, 3, 0, 3) == -1);
    CHECK(ds4_gpu_dsv41_shared_owner_scalar(ot, gt, ut, mt, xt, st, model, wb,
        0, gb, 2 * gb, 0, 0, 4) == -1);
    fprintf(stderr, "shared owner cases=%u\n", cases);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(ot); ds4_gpu_tensor_free(gt);
    ds4_gpu_tensor_free(ut); ds4_gpu_tensor_free(mt); ds4_gpu_tensor_free(st);
    ds4_gpu_cleanup();
    free(model); free(x); free(out); free(ref);
    return 1;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    const unsigned attn_rows[] = {1, 7, 64, 2048};
    for (unsigned i = 0; i < sizeof(attn_rows) / sizeof(*attn_rows); i++)
        if (!attention(attn_rows[i])) return 1;
    const unsigned moe_rows[] = {1, 8, 64, 256};
    for (unsigned i = 0; i < sizeof(moe_rows) / sizeof(*moe_rows); i++)
        if (!moe(moe_rows[i])) return 1;
    if (!shared_owner()) return 1;
    puts("V4.1 three-rank attention groups, owned Q4_K MoE and shared owner PASS");
    return 0;
}
