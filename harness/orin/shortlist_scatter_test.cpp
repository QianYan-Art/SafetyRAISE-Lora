// 草稿词表子集补丁的算子序列单元测试(与 src/models/qwen35.cpp 里 graph_mtp 的写法完全一致):
//   l_sub = mul_mat(head[K 行], x)           [K, n_out]
//   l_full = fill(-inf) 之后把 l_sub 的每一列按 ids 散射到完整词表   [n_vocab, n_out]
// 期望值在主机上直接按定义算出(先用同一个 mul_mat 得到 K 个 logits,再手工散射),与整条算子链的输出逐元素比较。
// 编译(Orin,对照构建目录里的库):
//   g++ -O1 -std=c++17 -I$LLAMA/ggml/include shortlist_scatter_test.cpp -L$LLAMA/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$LLAMA/build/bin -o scatter_test
//   加 CUDA:再加 -DUSE_CUDA -lggml-cuda
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

struct run_result {
    std::vector<float> sub;   // [K, n_out] 参考 mul_mat 输出
    std::vector<float> full;  // [n_vocab, n_out] 整条算子链输出
};

static run_result run(ggml_backend_t be, ggml_type type_head, int64_t n_embd, int64_t n_vocab, int64_t K, int64_t n_out,
                      const std::vector<float> & head_f32, const std::vector<float> & x, const std::vector<int64_t> & ids) {
    ggml_init_params ip = { /*.mem_size =*/ 1024 * ggml_tensor_overhead() + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * head = ggml_new_tensor_2d(ctx, type_head, n_embd, K);
    ggml_tensor * xin  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_out);
    ggml_tensor * idt  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, K);

    // 参考: 只做 mul_mat
    ggml_tensor * l_sub_ref = ggml_mul_mat(ctx, head, xin);
    // 整条链(与 graph_mtp 一致)
    ggml_tensor * l_sub   = ggml_mul_mat(ctx, head, xin);                                   // [K, n_out]
    ggml_tensor * l_sub_t = ggml_cont(ctx, ggml_transpose(ctx, l_sub));                     // [n_out, K]
    // 注意: CUDA 的 PAD 内核把 ne1 放进 gridDim.y(上限 65535),词表 248320 行放在 ne1 上会 "invalid argument",
    // 所以沿维度 0 补齐(ne1 = n_out 很小),再转置成 [n_out, n_vocab] 做行散射
    ggml_tensor * base_t  = ggml_pad(ctx, l_sub, (int) (n_vocab - K), 0, 0, 0);             // [n_vocab, n_out]
    base_t = ggml_fill_inplace(ctx, base_t, -INFINITY);
    ggml_tensor * l_full  = ggml_cont(ctx, ggml_transpose(ctx, base_t));                    // [n_out, n_vocab]
    l_full = ggml_set_rows(ctx, l_full, l_sub_t, idt);
    ggml_tensor * out = ggml_cont(ctx, ggml_transpose(ctx, l_full));                        // [n_vocab, n_out]

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, l_sub_ref);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); exit(2); }

    // 写入数据: 头权重按目标类型量化
    if (type_head == GGML_TYPE_F32) {
        ggml_backend_tensor_set(head, head_f32.data(), 0, head_f32.size() * sizeof(float));
    } else {
        std::vector<uint8_t> q(ggml_row_size(type_head, n_embd) * K);
        ggml_quantize_chunk(type_head, head_f32.data(), q.data(), 0, K, n_embd, nullptr);
        ggml_backend_tensor_set(head, q.data(), 0, q.size());
    }
    ggml_backend_tensor_set(xin, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(idt, ids.data(), 0, ids.size() * sizeof(int64_t));

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(3); }

    run_result r;
    r.sub.resize((size_t) (K * n_out));
    r.full.resize((size_t) (n_vocab * n_out));
    if (K * n_out > 0) ggml_backend_tensor_get(l_sub_ref, r.sub.data(), 0, r.sub.size() * sizeof(float));
    if (n_vocab * n_out > 0) ggml_backend_tensor_get(out, r.full.data(), 0, r.full.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

static int check(const char * name, ggml_backend_t be, ggml_type type_head, int64_t n_embd, int64_t n_vocab, int64_t K, int64_t n_out, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> head((size_t) (n_embd * K)), x((size_t) (n_embd * n_out));
    for (auto & v : head) v = nd(rng);
    for (auto & v : x) v = nd(rng);
    std::vector<int64_t> perm(n_vocab);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int64_t> ids(perm.begin(), perm.begin() + K);  // 互不重复的随机 token id

    run_result r = run(be, type_head, n_embd, n_vocab, K, n_out, head, x, ids);

    // 主机上按定义散射
    std::vector<float> expect((size_t) (n_vocab * n_out), -INFINITY);
    for (int64_t t = 0; t < n_out; t++) {
        for (int64_t k = 0; k < K; k++) {
            expect[(size_t) (t * n_vocab + ids[(size_t) k])] = r.sub[(size_t) (t * K + k)];
        }
    }
    size_t bad = 0, n_inf = 0, n_fin = 0;
    for (size_t i = 0; i < expect.size(); i++) {
        const bool ei = std::isinf(expect[i]), gi = std::isinf(r.full[i]);
        if (ei || gi) {
            n_inf++;
            if (!(ei && gi && (expect[i] < 0) == (r.full[i] < 0))) bad++;
        } else {
            n_fin++;
            if (expect[i] != r.full[i]) bad++;  // 数值必须逐位相同(同一个 mul_mat,只是搬运)
        }
    }
    printf("[%s] type=%s n_embd=%lld n_vocab=%lld K=%lld n_out=%lld : 有限 %zu 个, -inf %zu 个, 不一致 %zu -> %s\n", name, ggml_type_name(type_head),
           (long long) n_embd, (long long) n_vocab, (long long) K, (long long) n_out, n_fin, n_inf, bad, bad == 0 ? "OK" : "FAIL");
    return bad == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    int fails = 0;
    std::vector<std::pair<const char *, ggml_backend_t>> backends;
    backends.emplace_back("CPU", ggml_backend_cpu_init());
#ifdef USE_CUDA
    backends.emplace_back("CUDA0", ggml_backend_cuda_init(0));
#endif
    for (auto & [name, be] : backends) {
        if (!be) { printf("[%s] 后端初始化失败\n", name); fails++; continue; }
        for (ggml_type ty : { GGML_TYPE_F32, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K }) {
            for (int64_t n_out : { 1, 2, 3, 6 }) {
                fails += check(name, be, ty, 512, 4096, 1024, n_out, 1234 + (uint32_t) n_out);
            }
        }
        fails += check(name, be, GGML_TYPE_Q6_K, 5120, 248320, 32768, 1, 7);   // 真实尺寸
        fails += check(name, be, GGML_TYPE_Q6_K, 5120, 248320, 32768, 6, 8);
        fails += check(name, be, GGML_TYPE_Q6_K, 5120, 248320, 16384, 3, 9);
        ggml_backend_free(be);
    }
    printf("%s (%d 项失败)\n", fails == 0 ? "ALL_OK" : "SOME_FAILED", fails);
    return fails == 0 ? 0 : 1;
}
