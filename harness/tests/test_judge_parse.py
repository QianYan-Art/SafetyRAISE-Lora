"""judge JSON 提取的稳健性(MiniMax 偶发 not_json)。"""
import json
from sr_eval import judge as J

DIMS = {d: {"score": 4, "reason": "x"} for d in J.DIMENSIONS}
OBJ = json.dumps({"dimensions": DIMS, "concision": {"score": 3, "reason": "y"}, "defects": [], "overall_comment": "z"}, ensure_ascii=False)


def test_plain_json():
    assert J.parse_judgement(OBJ, "报告")["valid"]


def test_think_block_with_braces_before_json():
    text = "<think>先看看 {事故信息} 里 {\"a\": 1} 的字段……</think>\n" + OBJ
    assert J.parse_judgement(text, "报告")["valid"]


def test_prose_then_json_then_prose():
    text = "好的,评审结果如下:\n" + OBJ + "\n以上。{完毕}"
    assert J.parse_judgement(text, "报告")["valid"]


def test_failure_keeps_head_for_diagnosis():
    text = "没有 JSON 的输出"
    r = J.parse_judgement(text, "报告")
    assert r == {"valid": False, "error": "not_json", "raw_head": text, "raw_len": len(text)}


def test_concurrent_judge_writes_merge_not_overwrite(tmp_path):
    """两个评审进程各自把结果合并写入同一个文件时不能互相覆盖(模拟 cmd_judge 的"读-合并-原子替换")。"""
    import os
    path = tmp_path / "judgements.json"

    def write(key, judge, res):
        cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        cur.setdefault(key, {})[judge] = res
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur), encoding="utf-8")
        os.replace(tmp, path)

    write("k", "luna", {"valid": True, "total": 20})
    write("k", "minimax", {"valid": True, "total": 22})
    got = json.loads(path.read_text(encoding="utf-8"))
    assert got["k"]["luna"]["total"] == 20 and got["k"]["minimax"]["total"] == 22
