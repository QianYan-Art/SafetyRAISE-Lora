# “抓核心”粗评（辅助阅读，不代替阅读）。核心机制取合成案件的生成场景 scenario.type（不在提示里，模型看不到），
# 按《Causal AI》笔记的分析模板设几项：
#   hit   致因节是否把核心机制写成因果关系（核心词与因果连接词同句）
#   first 致因节第一个展开的因素是否就是核心机制
#   focus 致因+责任两节里提到核心词的句子占比
#   cf    是否用到反事实/必要性/充分性/敏感性表述（如“即使…也”“若当时…”“取决于”“关键在于”）
#   neg   否认句占比（含“不能/无法/不足以”且不含核心词的句子）——均匀罗列“不能认定”的程度
import json, re, sys
CORE = {
 "syn_dev_000": [r"安全距离|间距|车距", r"弯道|视线|观察"],
 "syn_dev_003": [r"湿滑|附着|积水", r"制动距离|制动"],
 "syn_dev_004": [r"跟车|车距|安全距离", r"制动"],
 "syn_dev_006": [r"逆行", r"迎面|对向|避让|观察"],
 "syn_dev_008": [r"倒车", r"观察|盲区|后方|行人位置"],
 "syn_dev_010": [r"横过|横穿|人行横道|让行", r"速度|观察|避让"],
 "syn_dev_011": [r"跟车|车距|安全距离", r"前车|连环|追尾"],
 "syn_dev_012": [r"让行|通行时序|进入路口|观察", r"电动自行车|非机动车"],
 "syn_dev_013": [r"轨迹|接触顺序|碰撞顺序|操作", r"摩托车|货车|客车"],
 "syn_dev_018": [r"明暗|光线|视觉适应|亮度", r"追尾|跟车|车距|制动"],
 "syn_dev_027": [r"转弯", r"直行|让行|避让|可见"],
 "syn_dev_038": [r"进站|靠站|轨迹交叉|盲区", r"非机动车|电动|自行车"],
}
CAUSAL = r"导致|造成|引发|使|影响|促成|诱因|原因|因果|相关|关联|增加|降低|压缩|延长"
CF = r"即使|即便|若当时|如果当时|假如|若没有|如果没有|取决于|关键在于|决定性|足以避免|能否避免|仍会|仍可能发生"
NEG = r"不能|无法|不足以|难以|尚不能"
def sents(t): return [s for s in re.split(r"[。；\n]", t) if len(s.strip()) > 4]
def is_core(s, case): a, b = CORE[case]; return bool(re.search(a, s)) and bool(re.search(b, s))
if __name__ == "__main__":
    Q = json.load(open(sys.argv[1], encoding="utf-8")); rows = []
    for o in Q:
        if o["kind"] != "report" or o["case"] not in CORE: continue
        r = o["rep"]; i = r.find("## 四、"); j5 = r.find("## 五、"); j6 = r.find("## 六、")
        if i < 0: continue
        cause = r[i:j5 if j5 > 0 else None]; both = r[i:j6 if j6 > 0 else None]
        cs = sents(cause)[1:]; bs = sents(both)
        hit = any(is_core(s, o["case"]) and re.search(CAUSAL, s) for s in cs)
        # 第一个展开的因素：致因节第一个小标题/粗体段落所在句
        first = bool(cs) and is_core(" ".join(cs[:2]), o["case"])
        focus = sum(is_core(s, o["case"]) for s in bs) / max(1, len(bs))
        cf = len(re.findall(CF, both))
        neg = sum(bool(re.search(NEG, s)) and not is_core(s, o["case"]) for s in bs) / max(1, len(bs))
        rows.append(dict(tag=o["tag"].split("#")[0], case=o["case"], hit=hit, first=first, focus=focus, cf=cf, neg=neg, n=len(bs)))
    json.dump(rows, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    med = lambda v: sorted(v)[len(v)//2]
    print(f"{'来源':14s} n  核心成因果句 首因素是核心 核心句占比 反事实/敏感性(中位) 非核心否认句占比 两节句数")
    for t in dict.fromkeys(x["tag"] for x in rows):
        v = [x for x in rows if x["tag"] == t]
        print(f"{t:14s} {len(v):2d} {sum(x['hit'] for x in v)/len(v):9.0%} {sum(x['first'] for x in v)/len(v):10.0%} {med([x['focus'] for x in v]):10.2f} {med([x['cf'] for x in v]):12d} {med([x['neg'] for x in v]):14.2f} {med([x['n'] for x in v]):6d}")
