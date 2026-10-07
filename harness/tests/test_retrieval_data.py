from sr_eval import retrieval_data as RD


def test_number_roundtrip():
    for n in (1, 9, 10, 19, 20, 38, 43, 76, 100, 105, 119, 120, 199):
        assert RD.cn_to_int(RD.int_to_cn(n)) == n, n
    assert RD.cn_to_int("四十三") == 43 and RD.cn_to_int("一百一十九") == 119
    assert RD.cn_to_int("一百一百一十九") is None          # 不存在的条号(模型笔误)不能被当成合法条号


def test_parse_item_variants():
    assert RD.parse_item("《中华人民共和国道路交通安全法（2021修正）》第四十三条") == ("道路交通安全法", 43)
    assert RD.parse_item("《道交法》第四十二条") == ("道路交通安全法", 42)
    assert RD.parse_item("第76条") == (None, 76)
    assert RD.parse_item("《道路交通安全法实施条例》第39条") == ("道路交通安全法实施条例", 39)


class FakeKB:
    chunks = [{"chunk_id": "law#0043", "title": "道路交通安全法(2021修正)", "content": "第四十三条 同车道行驶的机动车,后车应当与前车保持足以采取紧急制动措施的安全距离。"},
              {"chunk_id": "reg#0039", "title": "道路交通安全法实施条例", "content": "第三十九条 机动车在雨天行驶应当降低速度。"}]
    rules = []

    def search(self, q, k):
        return [{"id": "law#0043"}] if "43" in q or "四十三" in q else [{"id": "reg#0039"}]


def test_find_targets_and_verify():
    t = RD.find_targets(FakeKB(), "《道路交通安全法》第四十三条")
    assert [x["id"] for x in t] == ["law#0043"]
    ctx = {"targets": t}
    good = {"closing": "我准备在责任分析里引用《道路交通安全法》第四十三条,说明后车应当保持安全距离,但上面可见的片段里没有这一条的原文,不能凭记忆引用,所以先检索核对条文再写报告。",
            "query": "道路交通安全法 第四十三条 后车保持安全距离", "reason": "缺少第四十三条原文", "top_k": 2}
    assert RD.verify_writer_item(FakeKB(), ctx, good) == (True, "ok")
    assert not RD.verify_writer_item(FakeKB(), ctx, {**good, "query": "雨天降低速度"})[0]       # 没检到目标
    assert not RD.verify_writer_item(FakeKB(), ctx, {**good, "closing": "太短"})[0]
    assert not RD.verify_writer_item(FakeKB(), ctx, {**good, "top_k": 9})[0]


def test_extract_array_skips_inner_arrays():
    text = '说明 [1,2] 然后\n```json\n[{"id":"a","closing":"x","query":"q","reason":"r","top_k":2,"tags":[{"k":1}]}]\n```'
    arr = RD.extract_array(text)
    assert arr and arr[0]["id"] == "a"
