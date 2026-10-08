// 测 CPU 采样一个 token 的开销(词表 248320 行、服务端默认采样链 top_k 40 → top_p 0.95 → min_p 0.05 → temp 0.8 → dist):
// 投机验证时每个槽位每个位置都要采一次,6 槽位 × 6 位置就是 36 次/迭代,串行在服务主线程上,所以单次开销决定 GPU 要空等多久。
// 编译: g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include sampler_cost.cpp -L$LLAMA/build/bin -lllama -lggml-base -Wl,-rpath,$LLAMA/build/bin -o sampler_cost
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int n_vocab = 248320;
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> logits(n_vocab);
    for (auto & v : logits) v = nd(rng);
    for (int i = 0; i < 40; i++) logits[i * 1000] += 20.f;  // 造一个有尖峰的分布

    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(1234));

    std::vector<llama_token_data> cur(n_vocab);
    double t_build = 0, t_apply = 0;
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_vocab; i++) cur[i] = { i, logits[i], 0.0f };
        llama_token_data_array arr = { cur.data(), (size_t) n_vocab, -1, false };
        auto t1 = std::chrono::steady_clock::now();
        llama_sampler_apply(chain, &arr);
        auto t2 = std::chrono::steady_clock::now();
        t_build += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t_apply += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    printf("词表 %d, %d 次: 构造候选数组 %.3f ms/次, 采样链 %.3f ms/次, 合计 %.3f ms/次\n", n_vocab, iters, t_build / iters, t_apply / iters, (t_build + t_apply) / iters);
    llama_sampler_free(chain);
    return 0;
}
