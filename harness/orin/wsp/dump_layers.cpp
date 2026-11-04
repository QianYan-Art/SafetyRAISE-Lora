// 阶段 1 取方向用：在“服务模型”（Q4_K_M GGUF）上对一段 token 序列做前向，
// 把每层残差输出 l_out-<il> 在位置区间 [start, end) 上的均值写出（f32，形状 [n_layer, n_embd]）。
// 基于 ~/work/scripts/dump_hidden.cpp；只在与区间重叠的 ubatch 上请求回调，其余 ubatch 不同步、不拷贝。
// 编译: g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include dump_layers.cpp -L$LLAMA/build/bin -lllama -lggml-base -lggml -Wl,-rpath,$LLAMA/build/bin -o dump_layers
// 用法: dump_layers <model.gguf> <tokens.i32> <out.f32> <start> <end> [n_ubatch=512]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct cap_t {
    int n_embd = 0, n_layer = 0;
    long start = 0, end = 0;        // 绝对位置区间
    long off = 0; int n_cur = 0;    // 当前 ubatch 的起始位置与 token 数
    bool active = false;
    std::vector<double> sum;        // [n_layer * n_embd]
    std::vector<long> cnt;          // [n_layer]
    std::vector<float> buf;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    cap_t * c = (cap_t *) ud;
    const bool is_lout = std::strncmp(t->name, "l_out-", 6) == 0;
    if (ask) return c->active && is_lout;
    if (!is_lout || !c->active) return true;
    const int il = std::atoi(t->name + 6);
    if (il < 0 || il >= c->n_layer) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || t->ne[1] != c->n_cur) return true;  // 末层只算输出行时跳过
    const long lo = std::max(c->start, c->off), hi = std::min(c->end, c->off + c->n_cur);
    if (lo >= hi) return true;
    const size_t row = (size_t) c->n_embd * sizeof(float);
    c->buf.resize((size_t) (hi - lo) * c->n_embd);
    ggml_backend_tensor_get(t, c->buf.data(), (size_t) (lo - c->off) * t->nb[1], (size_t) (hi - lo) * row);
    double * s = c->sum.data() + (size_t) il * c->n_embd;
    for (long r = 0; r < hi - lo; ++r) {
        const float * x = c->buf.data() + (size_t) r * c->n_embd;
        for (int k = 0; k < c->n_embd; ++k) s[k] += x[k];
    }
    c->cnt[il] += hi - lo;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s <model.gguf> <tokens.i32> <out.f32> <start> <end> [n_ubatch=512]\n", argv[0]); return 1; }
    const char * model_path = argv[1], * tok_path = argv[2], * out_path = argv[3];
    cap_t c; c.start = std::atol(argv[4]); c.end = std::atol(argv[5]);
    const int n_ubatch = argc > 6 ? std::atoi(argv[6]) : 512;

    std::vector<int32_t> toks;
    {
        FILE * f = std::fopen(tok_path, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", tok_path); return 1; }
        std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
        toks.resize(sz / 4);
        if (std::fread(toks.data(), 4, toks.size(), f) != toks.size()) { std::fprintf(stderr, "short read\n"); return 1; }
        std::fclose(f);
    }
    if (c.start < 0 || c.end > (long) toks.size() || c.start >= c.end) { std::fprintf(stderr, "bad range\n"); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    c.n_embd = llama_model_n_embd(model); c.n_layer = llama_model_n_layer(model);
    c.sum.assign((size_t) c.n_layer * c.n_embd, 0.0); c.cnt.assign(c.n_layer, 0);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) toks.size() + 64;
    cp.n_batch = n_ubatch; cp.n_ubatch = n_ubatch; cp.n_outputs_max = n_ubatch;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;   // 与服务一致（KV q8_0）
    cp.cb_eval = cb; cp.cb_eval_user_data = &c;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }
    // 可选：CVEC_RAW=<f32 文件，n_embd*(n_layer-1)，从第 1 层起> CVEC_LO CVEC_HI，前向时加控制向量（编辑验证用）
    if (const char * cv = std::getenv("CVEC_RAW")) {
        FILE * f = std::fopen(cv, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", cv); return 1; }
        std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
        std::vector<float> d(sz / 4);
        if (std::fread(d.data(), 4, d.size(), f) != d.size()) { std::fprintf(stderr, "short read cvec\n"); return 1; }
        std::fclose(f);
        if (llama_set_adapter_cvec(ctx, d.data(), d.size(), c.n_embd, std::atoi(std::getenv("CVEC_LO")), std::atoi(std::getenv("CVEC_HI"))) != 0) { std::fprintf(stderr, "cvec failed\n"); return 1; }
        std::fprintf(stderr, "cvec applied %s [%s,%s]\n", cv, std::getenv("CVEC_LO"), std::getenv("CVEC_HI"));
    }

    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);
    for (size_t i = 0; i < toks.size(); i += n_ubatch) {
        const int n = (int) std::min<size_t>(n_ubatch, toks.size() - i);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = toks[i + j]; batch.pos[j] = (llama_pos) (i + j);
            batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
            batch.logits[j] = (i + j + 1 == toks.size());   // 只要最后一个位置的输出
        }
        c.off = (long) i; c.n_cur = n;
        c.active = (long) i < c.end && (long) (i + n) > c.start;
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed at %zu\n", i); return 1; }
        if ((i / n_ubatch) % 8 == 0) { std::fprintf(stderr, "\r%zu / %zu", i + n, toks.size()); std::fflush(stderr); }
    }
    std::vector<float> outv((size_t) c.n_layer * c.n_embd, 0.f);
    int n_ok = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        if (c.cnt[il] > 0) ++n_ok;
        for (int k = 0; k < c.n_embd; ++k)
            outv[(size_t) il * c.n_embd + k] = c.cnt[il] ? (float) (c.sum[(size_t) il * c.n_embd + k] / c.cnt[il]) : 0.f;
    }
    FILE * out = std::fopen(out_path, "wb");
    if (!out) { std::fprintf(stderr, "cannot open output\n"); return 1; }
    std::fwrite(outv.data(), sizeof(float), outv.size(), out); std::fclose(out);
    std::fprintf(stderr, "\ndone: %zu tokens, range [%ld,%ld), layers captured %d/%d (cnt0=%ld)\n", toks.size(), c.start, c.end, n_ok, c.n_layer, c.cnt[0]);
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
