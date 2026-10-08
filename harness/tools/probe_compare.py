"""比较两份 speed_probe2.py 的输出文本:逐轮给出是否逐字相同、第一处分歧位置(字符),以及解码速度/接受率对比。
用法: probe_compare.py <A.json> <B.json>
"""
import json, sys

A = json.load(open(sys.argv[1], encoding="utf-8"))
B = json.load(open(sys.argv[2], encoding="utf-8"))
for ra, rb in zip(A["rounds"], B["rounds"]):
    ta, tb = ra["text"], rb["text"]
    n = min(len(ta), len(tb))
    div = next((i for i in range(n) if ta[i] != tb[i]), None)
    same = div is None and len(ta) == len(tb)
    print(f"round {ra['round']}: {'逐字相同' if same else '分歧于字符 %s(A 长 %d / B 长 %d)' % (div if div is not None else n, len(ta), len(tb))}"
          f" | 解码 {ra['decode_tok_s']} → {rb['decode_tok_s']} tok/s | 接受率 {ra['accept_rate']} → {rb['accept_rate']}")
