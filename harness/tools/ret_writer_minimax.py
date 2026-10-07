"""MiniMax 写手:读 writer_input.md,调用 MiniMax(直连)输出 JSON 数组,写入 out_minimax.json。
用法: ret_writer_minimax.py <目录>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sr_eval import cli  # noqa: E402
from sr_eval import retrieval_data as RD  # noqa: E402
from sr_eval.llm import LLMError  # noqa: E402
from sr_eval.policy import OutboundPolicy  # noqa: E402


def main(d: str) -> None:
    out_dir = Path(d)
    client = cli.make_client()
    OutboundPolicy.load().check(kb_class="public_statutes", case_kind="synthetic")
    text = (out_dir / "writer_input.md").read_text(encoding="utf-8")
    last = ""
    for attempt in range(3):
        try:
            res = client.chat("minimax", [{"role": "user", "content": text}], max_tokens=24000, reasoning={"effort": "high"},
                              timeout=1800, purpose="ret_writer", meta={"case_id": "ret_writer"})
        except LLMError as exc:
            last = f"{exc.code}: {exc.detail[:200]}"
            print(f"第 {attempt + 1} 次失败:{last}", flush=True)
            continue
        (out_dir / "out_minimax.json").write_text(res.content, encoding="utf-8")
        arr = RD.extract_array(res.content)
        print(f"MiniMax 输出 {len(res.content)} 字符,解析出 {len(arr or [])} 个对象", flush=True)
        if arr:
            return
    print(f"MiniMax 失败:{last}", flush=True)


main(sys.argv[1])
