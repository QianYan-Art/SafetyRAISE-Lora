#!/bin/bash
# 预填充速度与数值对照(补丁 0004 的验证):
#   ① llama-bench:MMQ(默认)与"反量化 + cuBLAS"(bf16 / f16 计算)在不同 ubatch、不同 KV 精度下的预填充速度;
#   ② llama-perplexity 的 KL 散度:cuBLAS 路径相对 MMQ 的数值差异(同一段文本、同一量化模型,先存基线 logits 再比);
#   ③ nsys:MMQ 与 bf16 cuBLAS 各抓一次预填充,汇总到内核类别,看预填充时间花在哪。
# 用法: prefill_study.sh <GGUF 基名> [输出目录=~/work/runs]    需要 llama-opt 构建(含补丁 0004)、文本 ~/work/data/kld_text.txt
# 结果: <输出目录>/prefill_study.txt(速度)、kld_*.txt(散度)、prof_prefill_*_summary.txt(剖析)
NAME=${1:?need gguf basename}; OUT=${2:-$HOME/work/runs}; BIN=$HOME/work/llama-opt/build/bin; GG=$HOME/models/gguf/$NAME.gguf
TXT=$HOME/work/data/kld_text.txt; KLD=$HOME/work/runs/kld_base.bin; RES=$OUT/prefill_study.txt
cd "$BIN" || exit 1
: > "$RES"
bench() {  # bench <标签> <ubatch> <KV 参数> <环境变量...>
  local label=$1 ub=$2 kv=$3; shift 3
  echo "=== bench $label ub=$ub kv=[$kv] env=[$*] $(date +%T)" | tee -a "$RES"
  # shellcheck disable=SC2086
  env "$@" ./llama-bench -m "$GG" -ngl 99 -p 2048,8192 -n 0 -b 2048 -ub "$ub" -fa on $kv -r 2 -o md 2>&1 | grep "^|" | tee -a "$RES"
}
KVQ="-ctk q8_0 -ctv q8_0"; KVF="-ctk f16 -ctv f16"
BF="SR_CUBLAS_MIN_NE11=128"; HF="SR_CUBLAS_MIN_NE11=128 SR_CUBLAS_QUANT_PREC=f16"   # 默认 bf16 输入 + fp32 累加;QUANT_PREC=f16 为上游的 fp16 选择
# 批大小曲线:一次前向处理 n 个 token 的耗时(n=1 相当于解码,6~42 相当于 1~7 路的投机验证批,512 是预填充批)。
# 用来看 MMQ / MMVQ(≤8 列) / cuBLAS(≥32 列) 各自在哪个批大小上更快,并给多槽位解码建模(每次迭代的验证批 = 在生成的槽位数 × (1+草稿长度))
curve() {  # curve <标签> <环境变量...>
  local label=$1; shift
  echo "=== 批大小曲线 $label env=[$*] $(date +%T)" | tee -a "$RES"
  # shellcheck disable=SC2086
  env "$@" ./llama-bench -m "$GG" -ngl 99 -p 1,2,3,4,6,8,12,16,24,32,36,48,64,128,256,512 -n 0 -b 2048 -ub 512 -fa on $KVQ -r 3 -o md 2>&1 | grep "^|" | tee -a "$RES"
}
curve mmq X=1
curve mmvq8 SR_MMVQ_MAX_NE11=8
curve cublas32 SR_CUBLAS_MIN_NE11=32
echo "=== test-backend-ops(SR_CUBLAS_MIN_NE11=1:列数≥9 的 q4_K/q6_K 矩阵乘全部走 cuBLAS bf16,对照 CPU 参考) $(date +%T)" | tee -a "$RES"
SR_CUBLAS_MIN_NE11=1 timeout 1200 ./test-backend-ops test -b CUDA0 -o MUL_MAT -p 'type_a=q[46]_K' > "$OUT/tbo_cublas_bf16.txt" 2>&1
echo "通过 $(grep -c ' OK' "$OUT/tbo_cublas_bf16.txt") 项,失败 $(grep -c 'FAIL' "$OUT/tbo_cublas_bf16.txt") 项" | tee -a "$RES"
grep -i "FAIL\|NMSE.*>" "$OUT/tbo_cublas_bf16.txt" | head -8 | tee -a "$RES"
tail -3 "$OUT/tbo_cublas_bf16.txt" | tee -a "$RES"
bench mmq 512 "$KVQ" X=1
bench mmq 2048 "$KVQ" X=1
bench mmq_kvf16 512 "$KVF" X=1
bench cublas_bf16 512 "$KVQ" $BF
bench cublas_bf16 2048 "$KVQ" $BF
bench cublas_f16 512 "$KVQ" $HF
bench cublas_f16 2048 "$KVQ" $HF

echo "=== KL 散度(基线 = MMQ,KV q8_0,ub 2048,3 块 × 2048 token) $(date +%T)" | tee -a "$RES"
kl() {  # kl <标签> <额外参数> <环境变量...>
  local label=$1 extra=$2; shift 2
  # shellcheck disable=SC2086
  env "$@" ./llama-perplexity -m "$GG" -ngl 99 -f "$TXT" -c 2048 -b 2048 -ub 2048 --chunks 3 -fa on $KVQ --kl-divergence-base "$KLD" $extra > "$OUT/kld_$label.txt" 2>&1
  echo "--- $label" | tee -a "$RES"
  grep -i "Final estimate\|PPL\|Mean.*KLD\|Maximum KLD\|99.9%\|Median KLD\|Same top\|RMS" "$OUT/kld_$label.txt" | tail -14 | tee -a "$RES"
}
kl base "" X=1
kl cublas_bf16 "--kl-divergence" $BF
kl cublas_f16 "--kl-divergence" $HF
rm -f "$KLD"

prof() {  # prof <标签> <环境变量...>
  local label=$1; shift
  echo "=== nsys 预填充 $label $(date +%T)" | tee -a "$RES"
  # shellcheck disable=SC2086
  env "$@" nsys profile --trace=cuda --sample=none --cpuctxsw=none -o "$OUT/prof_prefill_$label" --force-overwrite true \
      ./llama-bench -m "$GG" -ngl 99 -p 2048 -n 0 -b 2048 -ub 512 -fa on $KVQ -r 1 -o md > /dev/null 2>&1
  nsys export --type sqlite --force-overwrite true -o "$OUT/prof_prefill_$label.sqlite" "$OUT/prof_prefill_$label.nsys-rep" > /dev/null 2>&1
  python3 "$HOME/work/scripts/prof_summary.py" "$OUT/prof_prefill_$label.sqlite" 30 > "$OUT/prof_prefill_${label}_summary.txt" 2>&1
  head -14 "$OUT/prof_prefill_${label}_summary.txt" | tee -a "$RES"
}
prof mmq X=1
prof cublas_bf16 $BF
echo "PREFILL_STUDY_DONE $(date +%T)" | tee -a "$RES"
