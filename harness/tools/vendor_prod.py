"""把线上同款代码(只读源 <SafetyRAISE-system-repo>\\backend,提交 7200e30,经服务器容器内哈希核对一致)冻结复制到项目内。
目的:Codex/评测台只在项目目录内工作,不碰生产仓库;所有后训练环境代码以这份冻结副本为准,并带 SHA-256 清单。
只复制 app/、config/(提示词、模板、两份 yaml)、requirements.txt;不复制 data/、tests/、evals/、native/、密钥、运行产物。
用法: python harness/tools/vendor_prod.py   (已存在则校验;--force 重新复制)
"""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

SRC = Path(r"<SafetyRAISE-system-repo>\backend")
REPO = SRC.parent
DST = Path(__file__).resolve().parents[1] / "vendor" / "safetyraise_7200e30"


def sha(p: Path) -> str:
    # 统一换行再哈希:服务器镜像是 LF,本地检出是 CRLF,内容相同
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    force = "--force" in sys.argv
    if DST.exists() and not force:
        man = json.loads((DST / "MANIFEST.json").read_text(encoding="utf-8"))
        bad = [r for r, h in man["files"].items() if not (DST / r).exists() or sha(DST / r) != h]
        print("已存在,校验", "通过" if not bad else f"失败 {bad[:5]}", f"({len(man['files'])} 个文件)")
        return
    if DST.exists():
        shutil.rmtree(DST)
    files = {}
    for sub in ("app", "config"):
        for p in sorted((SRC / sub).rglob("*")):
            if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            rel = p.relative_to(SRC)
            out = DST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
            files[rel.as_posix()] = sha(out)
    shutil.copy2(SRC / "requirements.txt", DST / "requirements.txt")
    files["requirements.txt"] = sha(DST / "requirements.txt")
    head = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%H %s"], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    (DST / "MANIFEST.json").write_text(json.dumps({
        "source": str(SRC), "source_commit": head, "copied_at": time.strftime("%Y-%m-%d %H:%M"),
        "note": "线上容器 safetyraise-backend-1(镜像 2026-09-25)内同名文件哈希与此一致(换行归一后);只读冻结副本,不得手改",
        "files": files}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("复制", len(files), "个文件 →", DST, "|", head)


main()
