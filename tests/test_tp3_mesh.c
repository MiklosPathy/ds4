/* Model-free --tensor-parallel3 mesh test: three processes on loopback.
 * Covers mesh bring-up, rank-ordered gate sums (scalar and bulk), leader
 * command broadcast, aggregated acks, agreed cancellation and hashes. */
#define DS4_ROCM_BUILD 1
#include "../ds4_tp.c"
#include <assert.h>
#include <sys/wait.h>

void ds4_vision_embedding_free(ds4_vision_embedding *embedding) {
    if (!embedding) return;
    free(embedding->data);
    memset(embedding, 0, sizeof(*embedding));
}

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "rank %d: check failed at %s:%d: %s\n", rank, __FILE__, __LINE__, #x); \
    return 1; } } while (0)

static float partial(int rank, uint64_t i, uint32_t salt) {
    /* Non-trivial magnitudes make the addition order observable. */
    const float scale = rank == 0 ? 1.0e8f : rank == 1 ? 1.0f : -1.0e8f;
    return scale * (float)((i * 2654435761u + salt) % 1000u) / 7.0f + (float)rank * 0.1f;
}

/* volatile pins the rank order even under -ffast-math. */
static float ordered_sum(float a, float b, float c) {
    volatile float ab = a + b;
    volatile float abc = ab + c;
    return abc;
}

static uint64_t hash_floats(const float *v, uint64_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char *p = (const unsigned char *)v;
    for (uint64_t i = 0; i < n * sizeof(float); i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

static int run_rank(int rank, int port) {
    ds4_tp_options opt = {
        .role = rank ? DS4_TP_WORKER : DS4_TP_LEADER,
        .listen_host = "127.0.0.1", .listen_port = port,
        .leader_host = "127.0.0.1", .leader_port = port,
        .transport = DS4_TP_TRANSPORT_TCP, .world = 3, .rank = rank,
    };
    ds4_tp_identity id = {.gguf_bytes = 1, .model_id = 41, .n_layer = 40, .n_embd = 5120,
                          .n_vocab = 129280, .quant_bits = 4, .ctx_size = rank ? 0 : 4096};
    ds4_tp *tp = NULL;
    char err[512] = "";
    CHECK(ds4_tp_create(&tp, &opt, &id, err, sizeof(err)) || (fprintf(stderr, "%s\n", err), 0));
    CHECK(ds4_tp_world(tp) == 3 && ds4_tp_rank(tp) == rank);

    /* Scalar gates through the slab. */
    const uint64_t slab_bytes = ds4_tp_slab_bytes(40, 5120);
    uint8_t *slab = calloc(1, slab_bytes);
    CHECK(slab && ds4_tp_attach_slab(tp, slab, err, sizeof(err)));
    for (uint32_t layer = 0; layer < 40; layer++) {
        for (uint32_t gate = 0; gate < 2; gate++) {
            float *out = (float *)(slab + ds4_tp_slab_out_offset(tp, layer, gate));
            float *in = (float *)(slab + ds4_tp_slab_in_offset(tp, layer, gate));
            const uint32_t salt = layer * 2u + gate;
            for (uint64_t i = 0; i < 5120; i++) out[i] = partial(rank, i, salt);
            CHECK(ds4_tp_gate_exchange(tp, layer, gate, salt + 1u));
            for (uint64_t i = 0; i < 5120; i++) {
                const float want = ordered_sum(partial(0, i, salt), partial(1, i, salt), partial(2, i, salt));
                CHECK(memcmp(&in[i], &want, sizeof(want)) == 0);
            }
            /* Every rank must hold bit-identical sums. */
            CHECK(ds4_tp_hash_check(tp, salt + 1u, hash_floats(in, 5120), err, sizeof(err)) == 1);
        }
    }

    /* Bulk gates of prefill size. */
    const uint32_t rows[] = {1, 32, 257, 2048};
    for (unsigned n = 0; n < sizeof(rows) / sizeof(*rows); n++) {
        const uint64_t count = (uint64_t)rows[n] * 5120u;
        float *out = malloc(count * sizeof(float)), *in = malloc(count * sizeof(float));
        CHECK(out && in);
        for (uint64_t i = 0; i < count; i++) out[i] = partial(rank, i, 1000u + n);
        CHECK(ds4_tp_big_gate_exchange(tp, 39, 100u + n, out, in, count * sizeof(float)));
        for (uint64_t i = 0; i < count; i++) {
            const float want = ordered_sum(partial(0, i, 1000u + n), partial(1, i, 1000u + n),
                                           partial(2, i, 1000u + n));
            CHECK(memcmp(&in[i], &want, sizeof(want)) == 0);
        }
        free(out);
        free(in);
    }

    /* Control plane: broadcast, aggregated acks, agreed cancellation. */
    if (rank == 0) {
        int status = -1;
        CHECK(ds4_tp_send_session_create(tp, 7, 4096));
        CHECK(ds4_tp_wait_command_status(tp, 7, &status, "create", err, sizeof(err)));
        CHECK(status == 0);
        CHECK(ds4_tp_send_session_destroy(tp, 7));
        CHECK(ds4_tp_wait_command_status(tp, 7, &status, "destroy", err, sizeof(err)));
        CHECK(status == 5); /* rank 2 reports failure */
        bool cancelled = false;
        CHECK(ds4_tp_sync_checkpoint(tp, 1, 10, 20, false, &cancelled));
        CHECK(cancelled);
        CHECK(ds4_tp_sync_checkpoint(tp, 2, 20, 20, false, &cancelled));
        CHECK(!cancelled);
        CHECK(ds4_tp_hash_check(tp, 3, 0x1234, err, sizeof(err)) == 1);
        CHECK(ds4_tp_send_stop(tp));
    } else {
        ds4_tp_command c;
        CHECK(ds4_tp_recv_command(tp, &c, err, sizeof(err)));
        CHECK(c.type == DS4_TP_FRAME_SESSION_CREATE && c.session_id == 7 && c.value == 4096);
        ds4_tp_command_free(&c);
        CHECK(ds4_tp_send_command_ack(tp, 7, 0));
        CHECK(ds4_tp_recv_command(tp, &c, err, sizeof(err)));
        CHECK(c.type == DS4_TP_FRAME_SESSION_DESTROY && c.session_id == 7);
        ds4_tp_command_free(&c);
        CHECK(ds4_tp_send_command_ack(tp, 7, rank == 2 ? 5 : 0));
        bool cancelled = false;
        CHECK(ds4_tp_sync_checkpoint(tp, 1, 10, 20, rank == 2, &cancelled));
        CHECK(cancelled);
        CHECK(ds4_tp_sync_checkpoint(tp, 2, 20, 20, false, &cancelled));
        CHECK(!cancelled);
        CHECK(ds4_tp_hash_check(tp, 3, 0x1234, err, sizeof(err)) == 1);
        CHECK(ds4_tp_recv_command(tp, &c, err, sizeof(err)));
        CHECK(c.type == DS4_TP_FRAME_STOP);
        ds4_tp_command_free(&c);
    }
    ds4_tp_free(tp);
    free(slab);
    printf("rank %d: tp3 mesh OK\n", rank);
    return 0;
}

int main(void) {
    const int port = 20000 + (int)(getpid() % 20000);
    pid_t pids[3];
    for (int rank = 0; rank < 3; rank++) {
        pids[rank] = fork();
        assert(pids[rank] >= 0);
        if (pids[rank] == 0) _exit(run_rank(rank, port));
    }
    int failed = 0;
    for (int rank = 0; rank < 3; rank++) {
        int status = 0;
        waitpid(pids[rank], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failed = 1;
    }
    printf("tp3 mesh: %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
