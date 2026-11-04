// 权重编辑用：统计 down_proj 的输入（ffn_swiglu-<il>，即 silu(gate)*up，长度 n_ff）在层 [lo,hi] 上的逐神经元分布。
// 模式 stats：逐神经元 sum / sumsq / max|a|（位置 >= skip），以及开头 skip 个位置（注意力汇聚位）上的 max|a|。
// 模式 gram ：读入每层候选神经元下标（idx 文件，int32，[hi-lo+1, K]），累加候选子向量的 sum 与 Gram（double，上三角），
//            另记开头 skip 个位置上的候选子向量，供检查编辑在汇聚位上的增益。
// 可选环境变量 CVEC_RAW=<f32 文件，n_embd*(n_layer-1)，从第 1 层起> CVEC_LO CVEC_HI：前向时加控制向量（用于对照）。
// 编译: g++ -O3 -march=native -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include dump_ffn.cpp -L$LLAMA/build/bin -lllama -lggml-base -lggml -Wl,-rpath,$LLAMA/build/bin -o dump_ffn
// 用法: dump_ffn <model.gguf> <tokens.i32> <out_prefix> <lo> <hi> stats|gram [idx.i32 K] [skip=4] [n_ubatch=512]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct cap_t {
    int lo = 0, hi = 0, nl = 0, n_ff = 0, K = 0, skip = 4;
    bool gram = false;
    long off = 0; int n_cur = 0;
    std::vector<double> sum, sq, mx, sinkmx;   // stats: [nl * n_ff]
    std::vector<int32_t> idx;                  // gram: [nl * K]
    std::vector<double> gs, gg, sink;          // gram: sum [nl*K], Gram [nl*K*K]（上三角），汇聚位 [nl*skip*K]
    std::vector<long> cnt;
    std::vector<float> buf, sub;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    cap_t * c = (cap_t *) ud;
    if (std::strncmp(t->name, "ffn_swiglu-", 11) != 0) return !ask;
    const int il = std::atoi(t->name + 11);
    if (il < c->lo || il > c->hi) return !ask;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[1] != c->n_cur) return true;
    if (c->n_ff == 0) {
        c->n_ff = (int) t->ne[0];
        if (!c->gram) { size_t n = (size_t) c->nl * c->n_ff; c->sum.assign(n, 0); c->sq.assign(n, 0); c->mx.assign(n, 0); c->sinkmx.assign(n, 0); }
    }
    if (t->ne[0] != c->n_ff) return true;
    const int L = il - c->lo, F = c->n_ff, n = c->n_cur;
    c->buf.resize((size_t) n * F);
    ggml_backend_tensor_get(t, c->buf.data(), 0, (size_t) n * F * sizeof(float));
    if (!c->gram) {
        double * s = &c->sum[(size_t) L * F], * q = &c->sq[(size_t) L * F], * m = &c->mx[(size_t) L * F], * sk = &c->sinkmx[(size_t) L * F];
        for (int r = 0; r < n; ++r) {
            const float * x = c->buf.data() + (size_t) r * F;
            if (c->off + r < c->skip) { for (int k = 0; k < F; ++k) sk[k] = std::max(sk[k], (double) std::fabs(x[k])); continue; }
            for (int k = 0; k < F; ++k) { const double v = x[k]; s[k] += v; q[k] += v * v; m[k] = std::max(m[k], std::fabs(v)); }
            c->cnt[L]++;
        }
    } else {
        const int K = c->K; const int32_t * id = &c->idx[(size_t) L * K];
        c->sub.resize((size_t) n * K);
        int m = 0;
        for (int r = 0; r < n; ++r) {
            const float * x = c->buf.data() + (size_t) r * F;
            if (c->off + r < c->skip) { for (int k = 0; k < K; ++k) c->sink[((size_t) L * c->skip + (c->off + r)) * K + k] = x[id[k]]; continue; }
            float * y = c->sub.data() + (size_t) m * K;
            for (int k = 0; k < K; ++k) y[k] = x[id[k]];
            ++m;
        }
        double * gs = &c->gs[(size_t) L * K], * gg = &c->gg[(size_t) L * K * K];
        for (int r = 0; r < m; ++r) {
            const float * y = c->sub.data() + (size_t) r * K;
            for (int a = 0; a < K; ++a) {
                const double ya = y[a]; gs[a] += ya;
                double * row = gg + (size_t) a * K;
                for (int b = a; b < K; ++b) row[b] += ya * y[b];
            }
        }
        c->cnt[L] += m;
    }
    return true;
}

static std::vector<char> slurp(const char * p) {
    FILE * f = std::fopen(p, "rb"); if (!f) { std::fprintf(stderr, "cannot open %s\n", p); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> v(sz); if (std::fread(v.data(), 1, sz, f) != (size_t) sz) { std::fprintf(stderr, "short read %s\n", p); std::exit(1); }
    std::fclose(f); return v;
}
static void spit(const std::string & p, const void * d, size_t n) {
    FILE * f = std::fopen(p.c_str(), "wb"); if (!f) { std::fprintf(stderr, "cannot write %s\n", p.c_str()); std::exit(1); }
    std::fwrite(d, 1, n, f); std::fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s <model> <tokens.i32> <out_prefix> <lo> <hi> stats|gram [idx.i32 K] [skip] [n_ubatch]\n", argv[0]); return 1; }
    cap_t c; c.lo = std::atoi(argv[4]); c.hi = std::atoi(argv[5]); c.nl = c.hi - c.lo + 1;
    c.gram = std::strcmp(argv[6], "gram") == 0;
    int ai = 7;
    if (c.gram) {
        std::vector<char> v = slurp(argv[ai++]); c.K = std::atoi(argv[ai++]);
        c.idx.resize(v.size() / 4); std::memcpy(c.idx.data(), v.data(), v.size());
        if ((int) c.idx.size() != c.nl * c.K) { std::fprintf(stderr, "idx size mismatch\n"); return 1; }
    }
    if (argc > ai) c.skip = std::atoi(argv[ai++]);
    const int n_ubatch = argc > ai ? std::atoi(argv[ai++]) : 512;
    if (c.gram) { c.gs.assign((size_t) c.nl * c.K, 0); c.gg.assign((size_t) c.nl * c.K * c.K, 0); c.sink.assign((size_t) c.nl * c.skip * c.K, 0); }
    c.cnt.assign(c.nl, 0);
    std::vector<char> tv = slurp(argv[2]); std::vector<int32_t> toks(tv.size() / 4); std::memcpy(toks.data(), tv.data(), tv.size());

    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) toks.size() + 64; cp.n_batch = n_ubatch; cp.n_ubatch = n_ubatch; cp.n_outputs_max = n_ubatch;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;
    cp.cb_eval = cb; cp.cb_eval_user_data = &c;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }
    if (const char * cv = std::getenv("CVEC_RAW")) {
        std::vector<char> d = slurp(cv); const int ne = llama_model_n_embd(model);
        if (llama_set_adapter_cvec(ctx, (const float *) d.data(), d.size() / 4, ne, std::atoi(std::getenv("CVEC_LO")), std::atoi(std::getenv("CVEC_HI"))) != 0) { std::fprintf(stderr, "cvec failed\n"); return 1; }
        std::fprintf(stderr, "cvec applied %s [%s,%s]\n", cv, std::getenv("CVEC_LO"), std::getenv("CVEC_HI"));
    }
    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);
    for (size_t i = 0; i < toks.size(); i += n_ubatch) {
        const int n = (int) std::min<size_t>(n_ubatch, toks.size() - i);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) { batch.token[j] = toks[i + j]; batch.pos[j] = (llama_pos) (i + j); batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = (i + j + 1 == toks.size()); }
        c.off = (long) i; c.n_cur = n;
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed at %zu\n", i); return 1; }
    }
    if (c.n_ff == 0 || c.cnt[0] == 0) { std::fprintf(stderr, "no ffn_swiglu tensors captured\n"); return 2; }
    const std::string P = argv[3];
    std::vector<double> hdr = { (double) c.nl, (double) c.n_ff, (double) c.K, (double) c.skip };
    for (int L = 0; L < c.nl; ++L) hdr.push_back((double) c.cnt[L]);
    spit(P + ".hdr", hdr.data(), hdr.size() * 8);
    if (!c.gram) { spit(P + ".sum", c.sum.data(), c.sum.size() * 8); spit(P + ".sq", c.sq.data(), c.sq.size() * 8); spit(P + ".mx", c.mx.data(), c.mx.size() * 8); spit(P + ".sink", c.sinkmx.data(), c.sinkmx.size() * 8); }
    else { spit(P + ".gs", c.gs.data(), c.gs.size() * 8); spit(P + ".gg", c.gg.data(), c.gg.size() * 8); spit(P + ".gsink", c.sink.data(), c.sink.size() * 8); }
    std::fprintf(stderr, "done %zu tokens, n_ff=%d, cnt[0]=%ld\n", toks.size(), c.n_ff, c.cnt[0]);
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
