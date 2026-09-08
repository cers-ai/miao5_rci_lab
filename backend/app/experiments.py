import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from . import store as db
from .models import embedding, model_record, speech_windows, unit, INFERENCE_LOCK, load
from .realtime import Runtime

POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="experiment")
CANCEL = {}


def required(table, ident):
    row = db.get(table, ident)
    if not row:
        raise ValueError(f"记录不存在: {table}/{ident}")
    return row


def audio_file(ident):
    f = required("experiment_files", ident)
    audio, rate = sf.read(f["normalized_path"], dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError("内部录音格式错误")
    return audio


def snapshot_model(ident):
    m = model_record(ident)
    return {key: m.get(key) for key in ("id", "name", "adapter", "version", "path", "device", "manifest")}


def new_run(kind, config, owner, mode=None):
    ids = config.get("file_ids", [config["file_id"]] if config.get("file_id") else [])
    files = [{k: required("experiment_files", f).get(k) for k in ("id", "name", "sha256", "duration_ms")} for f in dict.fromkeys(ids)]
    return db.put("experiment_runs", {"kind": kind, "config": config, "owner": owner, "mode": mode,
        "status": "queued", "progress": "排队等待运行", "inputs": files, "models": [], "result": None, "reviews": [], "error": None})


def check_cancel(ident):
    if CANCEL.get(ident) and CANCEL[ident].is_set():
        raise InterruptedError("实验已取消，已有实时输出保留")


def persist_result(ident, result):
    (db.WORK / "experiment_outputs" / f"{ident}.json").write_text(__import__("json").dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return db.update("experiment_runs", ident, result=result)


def queue(kind, config, owner, mode=None):
    r = new_run(kind, config, owner, mode)
    CANCEL[r["id"]] = threading.Event()
    POOL.submit(execute, r["id"])
    return r


def execute(ident):
    start = time.perf_counter()
    try:
        r = required("experiment_runs", ident)
        c = r["config"]
        check_cancel(ident)
        db.update("experiment_runs", ident, status="running", progress="加载模型并处理真实音频", started_at=db.now())
        if r["kind"] == "E1":
            result = discovery(ident, c)
        elif r["kind"] == "E2":
            result = verification(ident, c)
        else:
            result = replay(ident, c)
        check_cancel(ident)
        persist_result(ident, result)
        db.update("experiment_runs", ident, status="completed", progress="完成", elapsed_ms=(time.perf_counter()-start)*1000, completed_at=db.now())
    except Exception as exc:
        db.update("experiment_runs", ident, status="cancelled" if isinstance(exc, InterruptedError) else "failed",
            error=str(exc)[:2000], progress="已停止", elapsed_ms=(time.perf_counter()-start)*1000, completed_at=db.now())
    finally:
        CANCEL.pop(ident, None)


def discovery(ident, c):
    m = model_record(c["model_id"])
    if m["adapter"] not in ("campp", "wespeaker", "pyannote", "command"):
        raise ValueError("当前模型不支持说话人发现")
    models = [snapshot_model(c["model_id"])]
    if m["adapter"] != "pyannote":
        models.append(snapshot_model("vad"))
    db.update("experiment_runs", ident, models=models)
    audio = audio_file(c["file_id"])
    if m["adapter"] == "pyannote":
        import torch
        with INFERENCE_LOCK:
            obj, _ = load(c["model_id"])
            kwargs = {"num_speakers": c["speaker_count"]} if c["speaker_count"] else {}
            output = obj({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}, **kwargs)
        ann = getattr(output, "speaker_diarization", output)
        segments = [{"start_ms": x.start*1000, "end_ms": x.end*1000, "speaker_id": label, "confidence": None} for x,_,label in ann.itertracks(yield_label=True)]
    else:
        segments = speech_windows(audio, c["min_speech_duration"])
        if not segments:
            return {"speakers": [], "segments": [], "note": "未检测到满足最短时长的语音"}
        vectors = []
        for i,s in enumerate(segments):
            check_cancel(ident)
            vectors.append(embedding(c["model_id"], audio[int(s["start_ms"]*16):int(s["end_ms"]*16)]))
            db.update("experiment_runs", ident, progress=f"提取声纹 {i+1}/{len(segments)}")
        vectors = np.stack(vectors)
        n = c["speaker_count"]
        if n and n > len(segments):
            raise ValueError("指定说话人数超过有效语音窗口数")
        if len(vectors) == 1:
            labels = np.array([0])
        else:
            labels = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average", distance_threshold=None if n else c["clustering_threshold"]).fit_predict(vectors)
        centers = {label: unit(vectors[labels==label].mean(axis=0)) for label in set(labels)}
        # Stable labels in order of first appearance.
        names = {label: f"SPEAKER_{i+1:02d}" for i,label in enumerate(dict.fromkeys(labels))}
        for s,v,label in zip(segments, vectors, labels):
            s.update(speaker_id=names[label], confidence=float(np.clip(np.dot(v,centers[label]),-1,1)))
    for i,s in enumerate(segments):
        s["id"] = f"segment-{i}"
        s["overlap"] = any(j!=i and max(s["start_ms"], t["start_ms"])<min(s["end_ms"], t["end_ms"]) for j,t in enumerate(segments))
    speakers = [{"speaker_id": name, "total_duration_ms": sum(s["end_ms"]-s["start_ms"] for s in segments if s["speaker_id"]==name),
        "segments": [s for s in segments if s["speaker_id"]==name]} for name in dict.fromkeys(s["speaker_id"] for s in segments)]
    return {"speakers": speakers, "segments": segments, "confidence_definition": "CAM++/WeSpeaker为片段与聚类中心的余弦相似度，不是概率；pyannote未提供片段置信度时为空"}


def create_profile(c, owner):
    r = required("experiment_runs", c["run_id"])
    if r["kind"]!="E1" or r["status"]!="completed":
        raise ValueError("请选择已完成的 E1 实验")
    selected = set(c["speaker_ids"])
    available = {s["speaker_id"] for s in r["result"]["speakers"]}
    if not selected <= available:
        raise ValueError("所选 Speaker 不存在")
    all_segments = r["result"]["segments"]
    if not set(c["excluded_segments"]) <= {s["id"] for s in all_segments}:
        raise ValueError("排除片段不存在")
    audio = audio_file(r["config"]["file_id"])
    segments, parts, seconds = [], [], 0
    for s in all_segments:
        if s["speaker_id"] not in selected or s["id"] in c["excluded_segments"] or s.get("overlap"):
            continue
        a, b = s["start_ms"], min(s["end_ms"], s["start_ms"] + (60-seconds)*1000)
        if b-a < 200:
            continue
        segments.append({**s, "end_ms": b})
        parts.append(audio[int(a*16):int(b*16)])
        seconds += (b-a)/1000
        if seconds >= 60:
            break
    if seconds < 3:
        raise ValueError("可用非重叠语音不足3秒；建议选择30～60秒高质量目标语音")
    weights = np.array([len(p) for p in parts])
    vectors = np.stack([embedding(c["model_id"], p) for p in parts])
    v = unit(np.average(vectors, axis=0, weights=weights))
    ident = db.uid()
    np.save(db.WORK / "target_profiles" / f"{ident}.npy", v)
    return db.put("target_profiles", {"id": ident, "name": c["name"], "owner": owner, "source_run_id": r["id"],
        "source_file_id": r["config"]["file_id"], "source_segments": segments, "model_id": c["model_id"], "model": snapshot_model(c["model_id"]),
        "embedding": v.tolist(), "duration_ms": seconds*1000, "warning": "建档语音少于30秒，跨录音稳定性需额外验证" if seconds<30 else ""})


def checked_profile(ident, model_id=None):
    p = required("target_profiles", ident)
    if model_id and model_id != p["model_id"]:
        raise ValueError("识别模型必须与建档模型一致；更换模型请重新建立声纹")
    current = snapshot_model(p["model_id"])
    if any(current.get(k) != p["model"].get(k) for k in ("version", "adapter", "path")):
        raise ValueError("建档后的模型版本或路径已改变，请重新建立声纹")
    return p


def verification(ident, c):
    p = checked_profile(c["profile_id"], c["model_id"])
    db.update("experiment_runs", ident, models=[p["model"], snapshot_model("vad")], target_profile=p)
    files = []
    for f in dict.fromkeys(c["file_ids"]):
        audio = audio_file(f)
        segments = speech_windows(audio)
        for i,s in enumerate(segments):
            check_cancel(ident)
            v = embedding(c["model_id"], audio[int(s["start_ms"]*16):int(s["end_ms"]*16)])
            score = float(np.clip(np.dot(v, p["embedding"]), -1, 1))
            s.update(similarity=score, target=score>=c["similarity_threshold"], id=f"{f}-{i}")
            db.update("experiment_runs", ident, progress=f"验证录音 {len(files)+1}/{len(c['file_ids'])}，窗口 {i+1}/{len(segments)}")
        files.append({"file_id": f, "name": required("experiment_files",f)["name"], "detected": any(s["target"] for s in segments),
            "max_similarity": max((s["similarity"] for s in segments), default=None), "segments": segments})
    return {"files": files, "similarity_definition": "L2归一化声纹余弦相似度，范围[-1,1]，不代表识别概率"}


def replay(ident, c):
    p = checked_profile(c["profile_id"])
    db.update("experiment_runs", ident, models=[p["model"], snapshot_model("vad")], target_profile=p)
    # Warm up outside runtime clock so model initialization isn't mistaken for lock latency.
    with INFERENCE_LOCK:
        load(p["model_id"])
        load("vad")
    audio = audio_file(c["file_id"])
    rt = Runtime(p, c)
    event = CANCEL[ident]
    try:
        for i in range(0, len(audio), rt.hop_samples):
            chunk = audio[i:i+rt.hop_samples]
            due = (i+len(chunk))/16000
            event.wait(max(0, due - (time.perf_counter()-rt.started)))
            check_cancel(ident)
            rt.feed(chunk)
            db.update("experiment_runs", ident, progress=f"实时回放 {rt.total/16000:.1f}/{len(audio)/16000:.1f} 秒", result=rt.output(None))
    except Exception:
        persist_result(ident, rt.output(None))
        raise
    return rt.output(c.get("ground_truth"))
