# 协议变体 P1：在生成者系统提示末尾追加实验协议——claims 留空（引文由后续单独步骤生成）、思考里不起草报告全文/JSON。
# 只用于离线重放实验，不改交付协议。输入 requests.jsonl，只取开发案。
import json, sys
P1 = ("\n\n本次实验协议（覆盖上文传输协议中关于 claims 的要求）：claims 返回空数组 []，引文由后续单独步骤生成，"
      "你不需要在思考中核对引文是否逐字唯一，也不需要在思考中预演 JSON 转义。"
      "思考只用于分析案情：事实与证据边界、事故经过、致因、责任层次、需补证据；不要在思考中起草报告全文或 JSON。"
      "分析完成后，在答案中一次写出完整 JSON；obligation_resolutions 仍须覆盖全部事实义务。")
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(dst, "w", encoding="utf-8") as w:
    for l in open(src, encoding="utf-8"):
        r = json.loads(l)
        if r["split"] != "dev": continue
        m = [dict(x) for x in r["messages"]]
        assert m[0]["role"] == "system"
        m[0]["content"] = m[0]["content"] + P1
        r["messages"] = m; r["id"] = "p1" + r["id"]; r["variant"] = "P1"
        w.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(n, "requests ->", dst)
