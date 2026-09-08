"""Explicit model installer; inference never downloads mutable model revisions."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import store as db

MS = {"campp": "v2.0.2", "vad": "v2.0.4", "paraformer": "v2.0.4"}


def manifest(folder, repo, revision):
    files = {str(f.relative_to(folder)).replace("\\", "/"): {"sha256": db.digest_file(f), "bytes": f.stat().st_size}
             for f in folder.rglob("*") if f.is_file() and f.name != "manifest.json" and ".cache" not in f.parts and not f.name.startswith(".")}
    result = {"source": repo, "revision": revision, "files": files, "downloaded_at": db.now()}
    (folder / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def install(ident, test=True):
    m = db.get("local_models", ident)
    if not m or ident not in {r[0] for r in db.PRESETS}:
        raise ValueError("只能自动下载预设模型")
    folder = Path(m["path"])
    folder.mkdir(parents=True, exist_ok=True)
    db.update("local_models", ident, status="下载中", error="")
    try:
        if ident in MS or ident == "wespeaker":
            from scripts.model_fetch import fetch
            info = fetch(ident, folder)
            revision = info["revision"]
        else:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError("需要先接受 pyannote/speaker-diarization-community-1 模型协议，并在本地设置 HF_TOKEN 后重新下载")
            from huggingface_hub import HfApi, snapshot_download
            existing = folder / "manifest.json"
            revision = json.loads(existing.read_text(encoding="utf-8"))["revision"] if existing.exists() else HfApi().model_info(m["repo"], token=token).sha
            snapshot_download(m["repo"], revision=revision, token=token, local_dir=str(folder))
        if ident == "pyannote":
            info = manifest(folder, m["repo"], revision)
        fingerprint = __import__("hashlib").sha256(json.dumps(info["files"], sort_keys=True).encode()).hexdigest()
        db.update("local_models", ident, status="未测试", version=f"{revision}@{fingerprint[:16]}", manifest=info, error="")
        if test:
            from backend.app.models import test_model
            result = test_model(ident)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        # Do not output signed URLs or tokens from downloader errors.
        error = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: 下载或加载失败，请检查网络、依赖和模型授权"
        db.update("local_models", ident, status="未安装" if not (folder / "manifest.json").exists() else "测试失败", error=error)
        print(f"{ident}: {error}", flush=True)
        return False
    return True


def samples():
    import httpx
    repo = "iic/speech_campplus_sv_zh-cn_16k-common"
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        for name in ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav", "speaker2_a_cn_16k.wav"):
            target = db.WORK / "samples" / name
            if not target.exists():
                r = c.get(f"https://modelscope.cn/api/v1/models/{repo}/repo", params={"Revision": "v1.0.0", "FilePath": f"examples/{name}"})
                r.raise_for_status()
                target.write_bytes(r.content)
    manifest(db.WORK / "samples", f"https://modelscope.cn/models/{repo}", "v1.0.0/examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[x[0] for x in db.PRESETS])
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    samples()
    results = [install(ident, not args.skip_test) for ident in ([args.only] if args.only else ["campp", "vad", "wespeaker", "paraformer", "pyannote"])]
    sys.exit(0 if all(results) else 2)
