#!/bin/bash
# 数值验证：留出序列（dev_011，思考+正文区间）上逐层残差均值；原模型 / 原模型+控制向量 0.25 / 编辑模型 / 零编辑
W=~/work/wsp/phase1; L=$W/edit.log; cd $W
log() { echo "$(date +%T) $*" >> $L; }
T=fit/syn_dev_011.i32; A=22170; B=30781; D=~/work/wsp/scripts/dump_layers_cv
$D ~/models/gguf/v3h2m-Q4_K_M.gguf $T fit/lout_orig.f32 $A $B >> verify.log 2>&1 || { log VERIFY_FAIL orig; exit 1; }
CVEC_RAW=fit/cv025.f32 CVEC_LO=36 CVEC_HI=52 $D ~/models/gguf/v3h2m-Q4_K_M.gguf $T fit/lout_cv.f32 $A $B >> verify.log 2>&1 || { log VERIFY_FAIL cv; exit 1; }
$D ~/work/wsp/gguf/v3h2m-wsp-Q4_K_M.gguf $T fit/lout_wsp.f32 $A $B >> verify.log 2>&1 || { log VERIFY_FAIL wsp; exit 1; }
$D ~/work/wsp/gguf/v3h2m-null-Q4_K_M.gguf $T fit/lout_null.f32 $A $B >> verify.log 2>&1 || { log VERIFY_FAIL null; exit 1; }
log VERIFY_DONE
