/**
 * D4 measurement firmware — Edge Student (blockwise INT8 convolution blocks + FP32 GroupNorm/GELU, projection,
 * GRU and classifier), driven by the host script ../../host_run.py over UART0.
 *
 * Commands (one byte from the host):
 *   'H'  print "INFO {json}": chip, build configuration, heap at boot / after buffer allocation, model bytes
 *   'W'  causal 11-step window: reset GRU state, then for k = 0..10 print "RDY k", receive one epoch
 *        (3 x 3000 float32, little endian), run one step; finally print "RES {json}" with the logits after
 *        the 11th step and per-step timings (esp_timer, microseconds) and heap figures
 *   'S'  continuous streaming test: receive uint32 N, run N consecutive steps on the last received epoch WITHOUT
 *        resetting the GRU state; prints "STR ..." every 100 epochs and "STRESS {json}" at the end
 *
 * Differences from the round-1 benchmark firmware (E12): exact GELU (erf, as torch.nn.GELU), EOG pooling uses the
 * EOG length, task watchdog enabled, external PSRAM not used, timing split into model instantiation / INT8
 * convolution / FP32 parts. Each convolution block is instantiated from Flash for every epoch (as in round 1, to keep
 * the RAM peak low), so the per-epoch latency includes block instantiation.
 *
 * Memory layout (2026-10-17, needed on the ESP32 with the UART input path): the GroupNorm + GELU output of block 0
 * is quantized directly to the INT8 input grid of block 1 (exponent read from the block-1 model at boot, same
 * rounding as before), and the output of block 1 is averaged over time inside the GroupNorm loop (same summation
 * order). The results are bit-identical to keeping both outputs as FP32 buffers, but the static buffers shrink from
 * 208,800 B to 64,800 B.
 */
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_idf_version.h"
#include "sdkconfig.h"
#include "dl_model_base.hpp"

#define BIN(sym) extern const uint8_t sym##_start[] asm("_binary_" #sym "_start"); \
                 extern const uint8_t sym##_end[] asm("_binary_" #sym "_end");
BIN(eeg_block_0_espdl) BIN(eeg_block_1_espdl) BIN(eog_block_0_espdl) BIN(eog_block_1_espdl)
BIN(eeg_0_gn_weight_bin) BIN(eeg_0_gn_bias_bin) BIN(eeg_1_gn_weight_bin) BIN(eeg_1_gn_bias_bin)
BIN(eog_0_gn_weight_bin) BIN(eog_0_gn_bias_bin) BIN(eog_1_gn_weight_bin) BIN(eog_1_gn_bias_bin)
BIN(proj_linear_weight_bin) BIN(proj_linear_bias_bin) BIN(proj_ln_weight_bin) BIN(proj_ln_bias_bin)
BIN(gru_weight_ih_l0_bin) BIN(gru_weight_hh_l0_bin) BIN(gru_bias_ih_l0_bin) BIN(gru_bias_hh_l0_bin)
BIN(gru_norm_weight_bin) BIN(gru_norm_bias_bin) BIN(cls_ln_weight_bin) BIN(cls_ln_bias_bin)
BIN(cls_linear_weight_bin) BIN(cls_linear_bias_bin)
#define F(sym) ((const float *)sym##_start)
#define BLOB(sym) {#sym, sym##_start, sym##_end}

struct Blob { const char *name; const uint8_t *s, *e; };
static const Blob BLOBS[] = {
    BLOB(eeg_block_0_espdl), BLOB(eeg_block_1_espdl), BLOB(eog_block_0_espdl), BLOB(eog_block_1_espdl),
    BLOB(eeg_0_gn_weight_bin), BLOB(eeg_0_gn_bias_bin), BLOB(eeg_1_gn_weight_bin), BLOB(eeg_1_gn_bias_bin),
    BLOB(eog_0_gn_weight_bin), BLOB(eog_0_gn_bias_bin), BLOB(eog_1_gn_weight_bin), BLOB(eog_1_gn_bias_bin),
    BLOB(proj_linear_weight_bin), BLOB(proj_linear_bias_bin), BLOB(proj_ln_weight_bin), BLOB(proj_ln_bias_bin),
    BLOB(gru_weight_ih_l0_bin), BLOB(gru_weight_hh_l0_bin), BLOB(gru_bias_ih_l0_bin), BLOB(gru_bias_hh_l0_bin),
    BLOB(gru_norm_weight_bin), BLOB(gru_norm_bias_bin), BLOB(cls_ln_weight_bin), BLOB(cls_ln_bias_bin),
    BLOB(cls_linear_weight_bin), BLOB(cls_linear_bias_bin)};

static const int EP = 3000, NCH = 3, EMBED = 64, H = 32, NC = 5, STEPS = 11;
static const size_t EPOCH_BYTES = NCH * EP * sizeof(float);
static const uart_port_t PORT = UART_NUM_0;
static const uint32_t MEMCAP = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

static const int QBUF = 48 * 600;   // INT8 block-0 output = block-1 input, layout [t][c]
static int8_t *g_q;                  // static working buffers
static float *g_raw;
static size_t g_static_bytes;
static int g_exp_eeg1, g_exp_eog1;  // input exponents of the two block-1 models (read at boot)
static size_t g_iram_reserved = 0;  // ESP32 only: free IRAM heap reserved at boot (see app_main)

struct Timing { int64_t create, conv, gn, total; };
struct Mem { unsigned free_b, min_free, largest; };

static Mem mem_now() {
    return {(unsigned)heap_caps_get_free_size(MEMCAP), (unsigned)heap_caps_get_minimum_free_size(MEMCAP),
            (unsigned)heap_caps_get_largest_free_block(MEMCAP)};
}

// ---------------------------------------------------------------- FP32 operators
static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

// GroupNorm + GELU of an INT8 block output (layout [t][c]). The FP32 result is either quantized to the INT8 input
// grid of the next block (q_out, layout [t][c], exponent q_exp; rounding as for any block input) or averaged over
// time per channel (pool[c]).
static void group_norm_gelu(const int8_t *in, int exponent, const float *w, const float *b, int groups, int ch, int len,
                            int8_t *q_out, int q_exp, float *pool) {
    const float s = ldexpf(1.0f, exponent);
    const float inv_q = ldexpf(1.0f, -q_exp);
    const int cpg = ch / groups, n = cpg * len;
    for (int g = 0; g < groups; g++) {
        float sum = 0.0f, ss = 0.0f;
        for (int c = g * cpg; c < (g + 1) * cpg; c++)
            for (int t = 0; t < len; t++) sum += in[t * ch + c] * s;
        const float mean = sum / n;
        for (int c = g * cpg; c < (g + 1) * cpg; c++)
            for (int t = 0; t < len; t++) { float d = in[t * ch + c] * s - mean; ss += d * d; }
        const float inv = 1.0f / sqrtf(ss / n + 1e-5f);
        for (int c = g * cpg; c < (g + 1) * cpg; c++) {
            float acc = 0.0f;
            for (int t = 0; t < len; t++) {
                const float v = gelu(w[c] * (in[t * ch + c] * s - mean) * inv + b[c]);
                if (q_out) q_out[t * ch + c] = (int8_t)fmaxf(-128.0f, fminf(127.0f, roundf(v * inv_q)));
                else acc += v;
            }
            if (pool) pool[c] = acc / len;
        }
    }
}

static void layer_norm(float *out, const float *in, int n, const float *w, const float *b) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = in[i] - mean; var += d * d; }
    const float inv = 1.0f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = w[i] * (in[i] - mean) * inv + b[i];
}

static void linear(float *out, const float *in, const float *w, const float *b, int din, int dout) {
    for (int o = 0; o < dout; o++) {
        float s = b[o];
        for (int i = 0; i < din; i++) s += w[o * din + i] * in[i];
        out[o] = s;
    }
}

static void gru_step(float *h, const float *x) {          // PyTorch GRU, gate order r, z, n
    float gi[3 * 32], gh[3 * 32];
    linear(gi, x, F(gru_weight_ih_l0_bin), F(gru_bias_ih_l0_bin), EMBED, 3 * H);
    linear(gh, h, F(gru_weight_hh_l0_bin), F(gru_bias_hh_l0_bin), H, 3 * H);
    for (int i = 0; i < H; i++) {
        const float r = 1.0f / (1.0f + expf(-(gi[i] + gh[i])));
        const float z = 1.0f / (1.0f + expf(-(gi[H + i] + gh[H + i])));
        const float n = tanhf(gi[2 * H + i] + r * gh[2 * H + i]);
        h[i] = (1.0f - z) * n + z * h[i];
    }
}

// ---------------------------------------------------------------- INT8 convolution block (+ GroupNorm + GELU in FP32)
// Block input: FP32 [c][t] (block 0: the raw epoch, quantized here) or INT8 [t][c] already on the block's input grid
// (block 1: written by the GroupNorm/GELU of block 0 with exponent in_q_exp). Output: see group_norm_gelu.
static bool run_block(const uint8_t *espdl, const float *in_f, const int8_t *in_q, int in_q_exp, int in_c, int in_len,
                      const float *gn_w, const float *gn_b, int8_t *q_out, int q_exp, float *pool,
                      int *out_c, int *out_len, Timing &tm) {
    const int64_t t0 = esp_timer_get_time();
    dl::Model mdl((const char *)espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0, dl::MEMORY_MANAGER_GREEDY);
    const int64_t t1 = esp_timer_get_time();
    auto ins = mdl.get_inputs();
    auto outs = mdl.get_outputs();
    if (ins.empty() || outs.empty()) return false;
    dl::TensorBase *ti = ins.begin()->second, *to = outs.begin()->second;
    if (!ti || !ti->data || !to || !to->data) return false;
    int8_t *q = (int8_t *)ti->data;
    if (in_f) {
        const float inv_scale = ldexpf(1.0f, -(int)ti->exponent);
        for (int t = 0; t < in_len; t++)
            for (int c = 0; c < in_c; c++)
                q[t * in_c + c] = (int8_t)fmaxf(-128.0f, fminf(127.0f, roundf(in_f[c * in_len + t] * inv_scale)));
    } else {
        if ((int)ti->exponent != in_q_exp) return false;
        memcpy(q, in_q, (size_t)in_len * in_c);
    }
    mdl.run();
    const int64_t t2 = esp_timer_get_time();
    *out_c = to->shape.size() >= 3 ? to->shape[2] : to->shape[1];
    *out_len = to->shape.size() >= 3 ? to->shape[1] : to->shape[0];
    if (q_out && *out_c * *out_len > QBUF) return false;
    group_norm_gelu((const int8_t *)to->data, (int)to->exponent, gn_w, gn_b, 4, *out_c, *out_len, q_out, q_exp, pool);
    const int64_t t3 = esp_timer_get_time();
    tm.create += t1 - t0; tm.conv += t2 - t1; tm.gn += t3 - t2;
    return true;
}

static int input_exponent(const uint8_t *espdl) {
    dl::Model mdl((const char *)espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0, dl::MEMORY_MANAGER_GREEDY);
    auto ins = mdl.get_inputs();
    return ins.empty() ? 1000 : (int)ins.begin()->second->exponent;
}

static bool cnn_epoch(const float *raw, float *feat, Timing &tm) {
    float pooled[2 * EMBED];                         // room for any channel count; checked below
    int c0, l0, c1, l1;
    // EEG branch (input slots 0, 1)
    if (!run_block(eeg_block_0_espdl_start, raw, nullptr, 0, 2, EP, F(eeg_0_gn_weight_bin), F(eeg_0_gn_bias_bin),
                   g_q, g_exp_eeg1, nullptr, &c0, &l0, tm)) return false;
    if (!run_block(eeg_block_1_espdl_start, nullptr, g_q, g_exp_eeg1, c0, l0, F(eeg_1_gn_weight_bin), F(eeg_1_gn_bias_bin),
                   nullptr, 0, pooled, &c1, &l1, tm)) return false;
    const int eeg_c = c1;
    // EOG branch (input slot 2)
    if (!run_block(eog_block_0_espdl_start, raw + 2 * EP, nullptr, 0, 1, EP, F(eog_0_gn_weight_bin), F(eog_0_gn_bias_bin),
                   g_q, g_exp_eog1, nullptr, &c0, &l0, tm)) return false;
    if (eeg_c > EMBED) return false;
    if (!run_block(eog_block_1_espdl_start, nullptr, g_q, g_exp_eog1, c0, l0, F(eog_1_gn_weight_bin), F(eog_1_gn_bias_bin),
                   nullptr, 0, pooled + eeg_c, &c1, &l1, tm)) return false;
    if (eeg_c + c1 != EMBED) return false;
    float p[EMBED], n[EMBED];
    linear(p, pooled, F(proj_linear_weight_bin), F(proj_linear_bias_bin), EMBED, EMBED);
    layer_norm(n, p, EMBED, F(proj_ln_weight_bin), F(proj_ln_bias_bin));
    for (int i = 0; i < EMBED; i++) feat[i] = gelu(n[i]);
    return true;
}

static bool step(const float *raw, float *h, float *logits, Timing &tm) {
    tm = {0, 0, 0, 0};
    const int64_t t0 = esp_timer_get_time();
    float feat[EMBED], a[H], b[H];
    if (!cnn_epoch(raw, feat, tm)) return false;
    gru_step(h, feat);
    layer_norm(a, h, H, F(gru_norm_weight_bin), F(gru_norm_bias_bin));       // gru_norm (dropout = identity)
    layer_norm(b, a, H, F(cls_ln_weight_bin), F(cls_ln_bias_bin));           // classifier LayerNorm
    linear(logits, b, F(cls_linear_weight_bin), F(cls_linear_bias_bin), H, NC);
    tm.total = esp_timer_get_time() - t0;
    return true;
}

// ---------------------------------------------------------------- host link
static void read_exact(void *dst, size_t n) {
    uint8_t *p = (uint8_t *)dst;
    while (n) {
        int r = uart_read_bytes(PORT, p, n, pdMS_TO_TICKS(1000));
        if (r > 0) { p += r; n -= r; }
    }
}

static void cmd_info(const Mem &boot, const Mem &alloc) {
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    size_t espdl = 0, fp32 = 0;
    for (const Blob &bl : BLOBS) (strstr(bl.name, "_espdl") ? espdl : fp32) += (size_t)(bl.e - bl.s);
#ifdef CONFIG_COMPILER_OPTIMIZATION_PERF
    const char *opt = "performance (-O2)";
#else
    const char *opt = "other";
#endif
    printf("INFO {\"target\":\"%s\",\"chip_model\":%d,\"cores\":%d,\"revision\":%d,\"idf\":\"%s\",\"cpu_mhz\":%d,"
           "\"optimization\":\"%s\",\"psram_total\":%u,\"internal_total\":%u,\"reset_reason\":%d,"
           "\"boot_free\":%u,\"boot_largest\":%u,\"alloc_free\":%u,\"alloc_largest\":%u,\"static_buffer_bytes\":%u,"
           "\"model_files\":%d,\"espdl_bytes\":%u,\"fp32_bytes\":%u,\"eeg1_input_exp\":%d,\"eog1_input_exp\":%d,"
           "\"iram_reserved\":%u,\"firmware\":\"r2-int8-intermediate\"}\n",
           CONFIG_IDF_TARGET, (int)ci.model, ci.cores, ci.revision, esp_get_idf_version(), CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
           opt, (unsigned)heap_caps_get_total_size(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_total_size(MEMCAP),
           (int)esp_reset_reason(), boot.free_b, boot.largest, alloc.free_b, alloc.largest, (unsigned)g_static_bytes,
           (int)(sizeof(BLOBS) / sizeof(BLOBS[0])), (unsigned)espdl, (unsigned)fp32, g_exp_eeg1, g_exp_eog1,
           (unsigned)g_iram_reserved);
    fflush(stdout);
}

static void cmd_window() {
    float h[H] = {0}, logits[NC];
    Timing tm[STEPS];
    bool ok = true;
    for (int k = 0; k < STEPS; k++) {
        printf("RDY %d\n", k); fflush(stdout);
        read_exact(g_raw, EPOCH_BYTES);
        ok = ok && step(g_raw, h, logits, tm[k]);
        vTaskDelay(1);                                   // let the idle task run (task watchdog)
    }
    const Mem m = mem_now();
    printf("RES {\"ok\":%d,\"logits\":[%.9g,%.9g,%.9g,%.9g,%.9g]", ok, logits[0], logits[1], logits[2], logits[3], logits[4]);
    const char *keys[4] = {"step_us", "create_us", "conv_us", "fp32_gn_us"};
    for (int f = 0; f < 4; f++) {
        printf(",\"%s\":[", keys[f]);
        for (int k = 0; k < STEPS; k++) {
            const int64_t v = f == 0 ? tm[k].total : f == 1 ? tm[k].create : f == 2 ? tm[k].conv : tm[k].gn;
            printf("%s%lld", k ? "," : "", (long long)v);
        }
        printf("]");
    }
    printf(",\"free\":%u,\"min_free\":%u,\"largest\":%u}\n", m.free_b, m.min_free, m.largest);
    fflush(stdout);
}

static void cmd_stress() {
    uint32_t n = 0;
    read_exact(&n, sizeof(n));
    float h[H] = {0}, logits[NC];
    Timing tm;
    const Mem m0 = mem_now();
    double sum = 0, first = 0, last = 0;
    int64_t mn = INT64_MAX, mx = 0;
    uint32_t fails = 0;
    for (uint32_t e = 0; e < n; e++) {
        if (!step(g_raw, h, logits, tm)) fails++;
        sum += tm.total;
        if (e < 100) first += tm.total;
        if (e + 100 >= n) last += tm.total;
        mn = tm.total < mn ? tm.total : mn; mx = tm.total > mx ? tm.total : mx;
        if ((e + 1) % 100 == 0) {
            const Mem m = mem_now();
            printf("STR %u %u %u %u %lld\n", (unsigned)(e + 1), m.free_b, m.min_free, m.largest, (long long)tm.total);
            fflush(stdout);
        }
        vTaskDelay(1);
    }
    const Mem m1 = mem_now();
    const double k = n < 100 ? n : 100;
    printf("STRESS {\"n\":%u,\"fails\":%u,\"mean_us\":%.1f,\"first100_mean_us\":%.1f,\"last100_mean_us\":%.1f,"
           "\"min_us\":%lld,\"max_us\":%lld,\"free_start\":%u,\"free_end\":%u,\"min_free_end\":%u,"
           "\"largest_start\":%u,\"largest_end\":%u}\n",
           (unsigned)n, (unsigned)fails, sum / n, first / k, last / k, (long long)mn, (long long)mx,
           m0.free_b, m1.free_b, m1.min_free, m0.largest, m1.largest);
    fflush(stdout);
}

extern "C" void app_main(void) {
    uart_driver_install(PORT, 16 * 1024, 0, 0, NULL, 0);      // measurement link (test harness)
    vTaskDelay(pdMS_TO_TICKS(300));
#if CONFIG_IDF_TARGET_ESP32
    // ESP-DL allocates its tensor arena with MALLOC_CAP_INTERNAL, which on the ESP32 is served from instruction RAM
    // first; IRAM allows only 32-bit accesses and the INT8 kernels access bytes (LoadStoreError). The free IRAM heap
    // is therefore reserved here so that the arena is placed in byte-addressable data RAM.
    // Only the IRAM-only range 0x40080000-0x4009FFFF is kept reserved. EXEC memory of the dual-mapped D/IRAM region
    // comes back with its instruction-bus alias (>= 0x400A0000); it is released again after the loop, because data
    // allocations from that region use its byte-addressable data-bus address.
    void *dual[16];
    int n_dual = 0;
    for (;;) {
        const size_t l = heap_caps_get_largest_free_block(MALLOC_CAP_EXEC);
        if (l < 64) break;
        void *p = heap_caps_malloc(l - 32, MALLOC_CAP_EXEC);
        if (!p) break;
        const uintptr_t a = (uintptr_t)p;
        if (a >= 0x40080000u && a < 0x400A0000u) g_iram_reserved += l - 32;
        else if (n_dual < 16) dual[n_dual++] = p;
        else { heap_caps_free(p); break; }
    }
    for (int i = 0; i < n_dual; i++) heap_caps_free(dual[i]);
#endif
    const Mem boot = mem_now();
    const size_t a = QBUF * sizeof(int8_t), r = EPOCH_BYTES;
    g_q = (int8_t *)heap_caps_malloc(a, MEMCAP);
    g_raw = (float *)heap_caps_calloc(1, r, MEMCAP);
    g_static_bytes = a + r;
    const Mem alloc = mem_now();
    if (!g_q || !g_raw) { printf("FATAL buffer allocation\n"); return; }
    g_exp_eeg1 = input_exponent(eeg_block_1_espdl_start);
    g_exp_eog1 = input_exponent(eog_block_1_espdl_start);
    if (g_exp_eeg1 == 1000 || g_exp_eog1 == 1000) { printf("FATAL block-1 input exponent\n"); return; }
    // minimum free heap reported from here on (not since boot): excludes the start-up transients above
    heap_caps_monitor_local_minimum_free_size_start();
    printf("READY\n"); fflush(stdout);
    for (;;) {
        uint8_t c = 0;
        read_exact(&c, 1);
        if (c == 'H') cmd_info(boot, alloc);
        else if (c == 'W') cmd_window();
        else if (c == 'S') cmd_stress();
    }
}
