/* Metadata-only compatibility fixtures; no model weights or GPU required. */
#include "../ds4.c"
#include <assert.h>

static void check(const char *arch, const char *family, uint32_t family_type,
                  bool expected) {
    uint8_t data[256] = {0};
    ds4_kv kv[2] = {0};
    ds4_model model = {.map = data, .size = sizeof(data), .kv = kv};
    const char *keys[] = {"general.architecture", "dspark.target_architecture"};
    const char *values[] = {arch, family};
    size_t pos = 0;
    for (size_t i = 0; i < 2; i++) {
        if (!values[i]) continue;
        ds4_kv *entry = &kv[model.n_kv++];
        entry->key = (ds4_str){.ptr = keys[i], .len = strlen(keys[i])};
        entry->type = i == 1 ? family_type : GGUF_VALUE_STRING;
        entry->value_pos = pos;
        uint64_t len = strlen(values[i]);
        assert(pos + sizeof(len) + len <= sizeof(data));
        memcpy(data + pos, &len, sizeof(len));
        pos += sizeof(len);
        memcpy(data + pos, values[i], len);
        pos += len;
    }
    assert(support_model_checkpoint_compatible(&model) == expected);
}

int main(void) {
    g_ds4_shape.family = DS4_MODEL_FAMILY_DEEPSEEK41;
    check("deepseek41-dspark", NULL, GGUF_VALUE_STRING, true);
    check("deepseek41-dspark", "deepseek41", GGUF_VALUE_STRING, true);
    check("deepseek41-dspark", "deepseek4", GGUF_VALUE_STRING, false);
    check("deepseek41-dspark", "deepseek41", GGUF_VALUE_UINT32, false);
    check("deepseek41-dspark", "unknown", GGUF_VALUE_STRING, false);
    check("deepseek4", "deepseek41", GGUF_VALUE_STRING, false);
    check(NULL, "deepseek41", GGUF_VALUE_STRING, false);
    check(NULL, NULL, GGUF_VALUE_STRING, false);
    g_ds4_shape.family = DS4_MODEL_FAMILY_DEEPSEEK4;
    check("deepseek41-dspark", NULL, GGUF_VALUE_STRING, false);
    check("deepseek41-dspark", "deepseek41", GGUF_VALUE_STRING, false);
    puts("DSpark metadata compatibility: 10 checks passed");
    return 0;
}
