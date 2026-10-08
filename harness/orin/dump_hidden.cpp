// 用 llama.cpp 跑“服务模型”(合并后的 Q4_K_M GGUF),把一段 token 序列在每个位置的最后一层 output_norm 之后的隐藏状态(MTP 头的输入 h_t)
// 以 bf16 写到文件。用途:(1)诊断——用 HF 里的 MTP 头在“服务模型的真实特征”上算教师强制一致率,与训练时用的特征(NF4+LoRA)对比,判断训练/服务是否失配;
// (2)若失配明显,用这些特征重训/微调 MTP 头。
// 输入 tokens 文件:int32 小端数组(用 Python numpy.array(ids, dtype='<i4').tofile 写出)。输出:bf16 小端,形状 [n_tokens, n_embd],逐行。
// 编译: g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include dump_hidden.cpp -L$LLAMA/build/bin -lllama -lggml-base -Wl,-rpath,$LLAMA/build/bin -o dump_hidden
// 用法: dump_hidden <model.gguf> <tokens.i32> <out.bf16> [n_ubatch=512]
#include "llama.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);   // 就近舍入到偶数
    return (uint16_t) (u >> 16);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.i32> <out.bf16> [n_ubatch=512]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * tok_path   = argv[2];
    const char * out_path   = argv[3];
    const int    n_ubatch   = argc > 4 ? std::atoi(argv[4]) : 512;

    std::vector<int32_t> toks;
    {
        FILE * f = std::fopen(tok_path, "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", tok_path); return 1; }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        toks.resize(sz / 4);
        if (std::fread(toks.data(), 4, toks.size(), f) != toks.size()) { std::fprintf(stderr, "short read\n"); return 1; }
        std::fclose(f);
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) toks.size() + 64;
    cp.n_batch         = n_ubatch;
    cp.n_ubatch        = n_ubatch;
    cp.n_outputs_max   = n_ubatch;
    cp.embeddings      = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k          = GGML_TYPE_Q8_0;      // 与服务一致(KV q8_0)
    cp.type_v          = GGML_TYPE_Q8_0;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }

    const int n_embd = llama_model_n_embd(model);
    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);
    FILE * out = std::fopen(out_path, "wb");
    if (!out) { std::fprintf(stderr, "cannot open output\n"); return 1; }

    std::vector<uint16_t> row(n_embd);
    for (size_t i = 0; i < toks.size(); i += n_ubatch) {
        const int n = (int) std::min<size_t>(n_ubatch, toks.size() - i);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j]     = toks[i + j];
            batch.pos[j]       = (llama_pos) (i + j);
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed at %zu\n", i); return 1; }
        for (int j = 0; j < n; ++j) {
            const float * e = llama_get_embeddings_ith(ctx, j);
            if (!e) { std::fprintf(stderr, "no embeddings at %zu\n", i + j); return 1; }
            for (int k = 0; k < n_embd; ++k) row[k] = f32_to_bf16(e[k]);
            std::fwrite(row.data(), 2, n_embd, out);
        }
        if ((i / n_ubatch) % 8 == 0) { std::fprintf(stderr, "\r%zu / %zu", i + n, toks.size()); std::fflush(stderr); }
    }
    std::fprintf(stderr, "\ndone: %zu tokens, n_embd %d\n", toks.size(), n_embd);
    std::fclose(out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
