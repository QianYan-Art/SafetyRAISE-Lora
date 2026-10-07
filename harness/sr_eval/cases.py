"""合成案件生成:生产形态的 37 字段事故信息 + 按生产指导意见提示生成的指导意见清单。

形态依据(09 号真实案例的计数统计,不含正文):事故信息约 24/36 个字段非空、总计约 210 字,
指导意见 JSON 约 1.9K 字——即"事实很稀疏、提示很长"。因此合成案按 sparse/medium/rich 三档控制事实量。
全部标记 kind="synthetic",永远不含真实人名、真实号牌、真实地址。
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from . import prompt as P
from .llm import LLMClient

ACCIDENT_TEMPLATE = json.loads(P.load_asset("input_accident_template.json"))
FIELDS = tuple(ACCIDENT_TEMPLATE)

TYPES = [
    "同向行驶追尾", "路口转弯与直行车辆冲突", "侧面刮擦后失控撞护栏", "倒车时碰撞行人", "电动自行车与机动车交叉碰撞",
    "行人横过道路被撞", "单车失控侧翻", "三车连环追尾", "货车右转与非机动车碰撞", "开车门引发的二次事故",
    "停车场低速碰擦", "逆行非机动车与机动车迎面相撞", "施工占道区域碰撞锥桶后追尾", "隧道口光线突变引发追尾",
    "雨天湿滑导致制动距离不足", "夜间无照明路段撞上行人", "学校周边路口与行人冲突", "高速出入口交织区变道碰撞",
    "公交车进站与非机动车碰撞", "摩托车未戴头盔与机动车碰撞",
]
ROADS = ["城市主干道", "城市支路", "乡村公路", "高速公路匝道", "桥面", "隧道出口", "停车场内部道路", "无信号灯十字路口", "有信号灯十字路口"]
WEATHER = ["晴", "多云", "小雨", "大雨积水", "大雾", "小雪结冰", "阴"]
LIGHT = ["白天", "夜间有路灯", "夜间无路灯", "黄昏"]
PARTIES = ["两辆机动车", "机动车与电动自行车", "机动车与行人", "单车事故", "三辆机动车", "货车与小型客车", "摩托车与机动车"]
RICHNESS = {
    "sparse": "只填写画面/现场能直接观察到的字段,共约 18–24 个字段非空,每个值尽量 2–12 个字,所有非空值合计约 150–300 字;"
              "事故认定原因、事故责任、主要违法行为、事故初查原因、致死原因一律留空。",
    "medium": "约 24–30 个字段非空,所有非空值合计约 350–600 字;可含时间、地点(用虚构路名)、主要违法行为的概括性描述;事故责任留空。",
    "rich": "约 28–34 个字段非空,所有非空值合计约 800–1200 字;事故初查原因可写 60–150 字的经过描述;事故认定原因可有概括性描述;事故责任仍留空或写'待认定'。",
}


def scenario_for(seed: int) -> dict[str, str]:
    rnd = random.Random(seed)
    types = TYPES[:]
    random.Random(seed // len(TYPES)).shuffle(types)
    return {
        "type": types[seed % len(types)],
        "road": rnd.choice(ROADS),
        "weather": rnd.choice(WEATHER),
        "light": rnd.choice(LIGHT),
        "parties": rnd.choice(PARTIES),
        "richness": ["sparse", "sparse", "medium", "rich"][rnd.randrange(4)],  # 贴近真实:稀疏居多
        "conflict": "yes" if rnd.random() < 0.15 else "no",
    }


def accident_prompt(scn: dict[str, str]) -> str:
    conflict = (
        "\n另外:请在字段之间埋入 1 处不显眼的轻微矛盾(例如事故形态与现场形态描述不完全一致),不要在任何字段里指出矛盾。"
        if scn["conflict"] == "yes" else ""
    )
    return (
        "你是交通事故合成数据的作者。请虚构一起交通事故,按给定模板输出结构化事故信息 JSON,用于测试报告生成系统。\n"
        f"场景要求:事故情形={scn['type']};道路={scn['road']};天气={scn['weather']};光照={scn['light']};参与方={scn['parties']}。\n"
        f"信息量:{RICHNESS[scn['richness']]}{conflict}\n"
        "硬性规则:\n"
        "1. 只输出一个 JSON 对象,键名必须与模板完全一致(不增删改),所有值都是字符串;未知或不写的字段输出空字符串 \"\"。\n"
        "2. 完全虚构:不得出现真实人名、真实号牌(号牌号码留空或写'已脱敏')、真实门牌与真实企业;地点用泛称或虚构路名。\n"
        "3. 数值(速度、年龄、人数、时间等)只在确有必要时写,且全文一致;不要写结论性的责任认定。\n"
        "4. 字段值贴近真实系统:多为简短中文短语,而不是长句。\n"
        "输出模板(键名顺序即模板顺序):\n" + json.dumps(ACCIDENT_TEMPLATE, ensure_ascii=False, indent=2)
    )


def _json_from(text: str) -> dict[str, Any] | None:
    for cand in (text.strip(), *(m for m in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)),
                 text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            obj = json.loads(cand)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def validate_accident(obj: dict[str, Any] | None) -> tuple[dict[str, str] | None, str]:
    if obj is None:
        return None, "not_json"
    extra = [k for k in obj if k not in ACCIDENT_TEMPLATE]
    if extra:
        return None, f"unknown_keys:{extra[:3]}"
    out = {k: (str(obj.get(k, "")) if not isinstance(obj.get(k, ""), (dict, list)) else json.dumps(obj[k], ensure_ascii=False)) for k in FIELDS}
    filled = sum(bool(v.strip()) for v in out.values())
    if filled < 10:
        return None, f"too_sparse:{filled}"
    return out, "ok"


def guidance_skeleton() -> dict[str, Any]:
    text = P.load_asset("guidance_prompt.md")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    return json.loads(m.group(1))


def _key_shape(obj: Any) -> Any:
    return {k: _key_shape(v) for k, v in obj.items()} if isinstance(obj, dict) else None


def validate_guidance(obj: dict[str, Any] | None) -> str:
    if obj is None:
        return "not_json"
    if _key_shape(obj) != _key_shape(guidance_skeleton()):
        return "keys_mismatch"
    for v in _walk(obj):
        if not isinstance(v, str) or not v.strip() or not re.match(r"^(需要关注|信息不足|待核实)[:：]", v):
            return "bad_value"
    return "ok"


def _walk(o: Any):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    else:
        yield o


def guidance_prompt(accident: dict[str, str]) -> str:
    template = P.load_asset("guidance_prompt.md")
    return template.replace(P.REPORT_ACCIDENT_PLACEHOLDER, json.dumps(accident, ensure_ascii=False, indent=2), 1)


def generate_case(client: LLMClient, model: str, seed: int, split: str, *, max_tokens: int = 24000, timeout: float = 900,
                  meta: dict[str, Any] | None = None) -> dict[str, Any]:
    scn = scenario_for(seed)
    meta = {**(meta or {}), "seed": seed}
    r1 = client.chat(model, [{"role": "user", "content": accident_prompt(scn)}], max_tokens=max_tokens,
                     timeout=timeout, purpose="case_accident", meta=meta)
    accident, why = validate_accident(_json_from(r1.content))
    if accident is None:
        raise ValueError(f"事故信息不合格: {why}")
    r2 = client.chat(model, [{"role": "user", "content": guidance_prompt(accident)}], max_tokens=max_tokens,
                     timeout=timeout, purpose="case_guidance", meta=meta)
    g = _json_from(r2.content)
    why = validate_guidance(g)
    if why != "ok":
        raise ValueError(f"指导意见不合格: {why}")
    return {
        "case_id": f"syn_{split}_{seed:03d}", "kind": "synthetic", "split": split, "seed": seed, "scenario": scn,
        "accident_data": accident, "guidance": g,
        "label": "合成案件,未经人工核实;不含真实个人信息",
        "gen_meta": {"model": model, "slug": client.models[model].slug, "cost_usd": round(r1.cost_usd + r2.cost_usd, 6),
                     "filled_fields": sum(bool(v.strip()) for v in accident.values()),
                     "accident_chars": sum(len(v) for v in accident.values())},
    }
