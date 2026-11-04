#!/bin/bash
# 按幅度补偿重建编辑模型（scale 0.278），只重跑编辑模型的数值验证，然后接行为验证链
W=~/work/wsp/phase1; L=$W/edit.log; cd $W
mv fit/lout_wsp.f32 fit/lout_wsp_s0.25.f32
rm -f ~/work/wsp/gguf/v3h2m-wsp-Q4_K_M.gguf
bash ../scripts/edit_make.sh v3h2m-wsp fit/v_k512_l0.01.npy 0.278 || exit 1
~/work/wsp/scripts/dump_layers_cv ~/work/wsp/gguf/v3h2m-wsp-Q4_K_M.gguf fit/syn_dev_011.i32 fit/lout_wsp.f32 22170 30781 >> verify.log 2>&1 || { echo "$(date +%T) VERIFY_FAIL wsp2" >> $L; exit 1; }
echo "$(date +%T) REVERIFY_DONE" >> $L
~/venvs/ft/bin/python ../scripts/edit_compare.py > fit/compare_s0.278.txt 2>&1
bash ../scripts/chain_wsp.sh
