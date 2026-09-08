"""Real local inference adapters. No network calls or synthetic result fallback."""
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from . import store as db

os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
CACHE = {}
INFERENCE_LOCK = threading.RLock()


def model_record(ident):
    m = db.get("local_models", ident)
    if not m:
        raise ValueError("模型不存在")
    if not m["enabled"]:
        raise ValueError(f"模型已停用：{m['name']}")
    if not Path(m["path"]).exists():
        raise ValueError(f"模型尚未安装：{m['name']}")
    if m["status"] == "未安装":
        raise ValueError(m.get("error") or "模型尚未安装，请运行下载程序")
    return m


def load(ident):
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    m = model_record(ident)
    device = m["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请选择 CPU 或自动设备")
    key = (ident, m["path"], m.get("version"), device, m["adapter"], m["load_mode"])
    if key in CACHE:
        return CACHE[key], m
    p = Path(m["path"])
    # Validate every registered artifact against the immutable installation manifest before loading.
    for relative, info in m.get("manifest", {}).get("files", {}).items():
        artifact = (p / relative).resolve()
        if not artifact.is_relative_to(p.resolve()) or not artifact.is_file() or db.digest_file(artifact) != info["sha256"]:
            raise ValueError(f"模型文件校验失败：{relative}，请重新下载或注册新版本")
    if m["load_mode"] == "命令行进程":
        if m["adapter"] != "command" or not p.is_file():
            raise ValueError("命令行模型须指定 command 适配器和可执行文件路径，协议见 README")
        obj = str(p)
    elif m["adapter"] in ("campp", "vad", "paraformer"):
        from funasr import AutoModel
        obj = AutoModel(model=str(p), device=device, disable_update=True, disable_pbar=True, trust_remote_code=False)
    elif m["adapter"] == "wespeaker":
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        if device != "cpu":
            raise ValueError("当前 WeSpeaker ONNX 运行时仅安装 CPU provider")
        obj = ort.InferenceSession(str(p / "cnceleb_resnet34_LM.onnx" if p.is_dir() else p), opts, providers=["CPUExecutionProvider"])
    elif m["adapter"] == "pyannote":
        from pyannote.audio import Pipeline
        obj = Pipeline.from_pretrained(str(p))
        obj.to(torch.device(device))
    else:
        raise ValueError("此加载方式没有对应的推理适配器")
    CACHE[key] = obj
    return obj, m


def unit(vector):
    a = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(a)
    if a.size < 8 or not np.all(np.isfinite(a)) or norm < 1e-8:
        raise ValueError("模型返回无效声纹向量")
    return a / norm


def command(obj, operation, audio, parameters=None):
    with tempfile.TemporaryDirectory(prefix="miaowu-") as tmp:
        path = Path(tmp) / "input.wav"
        sf.write(path, audio, 16000)
        payload = {"operation": operation, "audio_path": str(path), "sample_rate": 16000, "parameters": parameters or {}}
        proc = subprocess.run([obj], input=json.dumps(payload), capture_output=True, text=True, timeout=120, shell=False)
        if proc.returncode:
            raise ValueError(f"命令行模型退出码 {proc.returncode}")
        return json.loads(proc.stdout)


def embedding(ident, audio):
    import torch
    from torchaudio.compliance.kaldi import fbank
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 3200:
        raise ValueError("有效语音不足 0.2 秒，不能建立声纹")
    with INFERENCE_LOCK, torch.inference_mode():
        obj, m = load(ident)
        if m["adapter"] == "campp":
            result = obj.generate(input=audio, disable_pbar=True)
            value = result[0]["spk_embedding"]
            return unit(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
        if m["adapter"] == "wespeaker":
            # Official WeSpeaker Chinese frontend: 16-bit PCM scale, 80-bin hamming fbank + CMN.
            feats = fbank(torch.from_numpy(audio * 32768).unsqueeze(0), num_mel_bins=80,
                frame_length=25, frame_shift=10, sample_frequency=16000, window_type="hamming", dither=0)
            feats -= feats.mean(dim=0)
            return unit(obj.run(None, {obj.get_inputs()[0].name: feats.numpy()[None].astype(np.float32)})[0])
        if m["adapter"] == "command":
            return unit(command(obj, "embedding", audio)["embedding"])
        raise ValueError("所选模型不支持 Speaker Embedding")


def speech_segments(audio, minimum=0.5):
    with INFERENCE_LOCK:
        obj, _ = load("vad")
        res = obj.generate(input=np.asarray(audio, np.float32), disable_pbar=True)
    end = len(audio) / 16
    return [{"start_ms": max(0, float(a)), "end_ms": min(end, float(b))} for a, b in res[0]["value"] if min(end, b) - max(0, a) >= minimum * 1000]


def speech_windows(audio, minimum=0.5, width_ms=1500):
    result = []
    for s in speech_segments(audio, minimum):
        a, b = s["start_ms"], s["end_ms"]
        count = max(1, int(np.ceil((b-a) / width_ms)))
        # Equal subdivisions avoid dropping short tails and yield non-overlapping evidence.
        edges = np.linspace(a, b, count+1)
        result.extend({"start_ms": float(x), "end_ms": float(y)} for x, y in zip(edges, edges[1:]))
    return result


def test_model(ident):
    start = time.perf_counter()
    try:
        sample = db.WORK / "samples" / "speaker1_a_cn_16k.wav"
        if not sample.exists():
            raise ValueError("缺少真实语音测试样本，请先运行 scripts/download_models.py")
        audio, sr = sf.read(sample, dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            raise ValueError("测试样本格式必须为 16kHz 单声道")
        with INFERENCE_LOCK:
            obj, m = load(ident)
            if m["adapter"] in ("campp", "wespeaker", "command"):
                v = embedding(ident, audio[:48000])
                output = {"embedding_dimension": len(v), "norm": float(np.linalg.norm(v))}
            elif m["adapter"] == "vad":
                result = obj.generate(input=audio, disable_pbar=True)
                if not result[0]["value"]:
                    raise ValueError("测试语音未检测到有效片段")
                output = {"segments": result[0]["value"]}
            elif m["adapter"] == "paraformer":
                cache, texts = {}, []
                for i in range(0, len(audio), 9600):
                    r = obj.generate(input=audio[i:i+9600], cache=cache, is_final=i+9600>=len(audio), chunk_size=[0,10,5], encoder_chunk_look_back=4, decoder_chunk_look_back=1, disable_pbar=True)
                    texts.extend(x.get("text", "") for x in r)
                if not "".join(texts).strip():
                    raise ValueError("流式识别测试没有返回文本")
                output = {"text": "".join(texts)}
            else:
                import torch
                result = obj({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000})
                ann = getattr(result, "speaker_diarization", result)
                output = {"segments": [[x.start, x.end, label] for x, _, label in ann.itertracks(yield_label=True)]}
                if not output["segments"]:
                    raise ValueError("pyannote 未返回说话人片段")
        value = {"status": "可用", "elapsed_ms": round((time.perf_counter()-start)*1000), "output": output, "sample_sha256": db.digest_file(sample), "tested_at": db.now()}
        db.update("local_models", ident, status="可用", error="", test_result=value)
    except Exception as exc:
        value = {"status": "测试失败", "error": str(exc)[:1200], "elapsed_ms": round((time.perf_counter()-start)*1000), "tested_at": db.now()}
        db.update("local_models", ident, status="测试失败", error=value["error"], test_result=value)
    return value
