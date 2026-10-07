from sr_eval import retrieval_data as RD

VISIBLE = [{"id": "road_traffic_safety_law_2021#0007", "title": "中华人民共和国道路交通安全法(2021修正)"},
           {"id": "road_traffic_safety_regulation_1614_pdf#0007", "title": "中华人民共和国道路交通安全法实施条例"}]


def test_paragraph_names_visible_titles_and_articles():
    rep = "后车应当保持安全距离,依据《道路交通安全法》第四十三条 [依据: road_traffic_safety_law_2021#0007]。"
    p = RD.grounding_paragraph(rep, VISIBLE, 0)
    assert "第四十三条" in p and "道路交通安全法(2021修正)" in p


def test_paragraph_is_none_when_nothing_to_ground():
    assert RD.grounding_paragraph("没有任何引用的报告", VISIBLE, 0) is None
    assert RD.grounding_paragraph("引用了不可见的 [依据: unknown_source#0001]", VISIBLE, 0) is None     # 推不出就不写,不编造


def test_variants_differ():
    rep = "依据 [依据: road_traffic_safety_law_2021#0007]。"
    assert len({RD.grounding_paragraph(rep, VISIBLE, i) for i in range(4)}) == 4
