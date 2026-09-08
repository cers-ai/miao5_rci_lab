"""Shared streaming runtime for microphone PCM and clock-paced recording replay."""
import time
import numpy as np
from .models import embedding, speech_segments


class LockMachine:
    def __init__(self, lock_threshold, unlock_threshold, confirm_windows):
        if unlock_threshold >= lock_threshold or confirm_windows < 1:
            raise ValueError("无效锁定参数")
        self.high, self.low, self.required = lock_threshold, unlock_threshold, confirm_windows
        self.state, self.count = "NON_TARGET", 0

    def advance(self, similarity):
        old = self.state
        if self.state == "LOCKED":
            if similarity is None or similarity < self.low:
                self.state, self.count = "UNLOCKED", 0
        elif similarity is not None and similarity >= self.high:
            self.count += 1
            self.state = "LOCKED" if self.count >= self.required else "CANDIDATE"
        else:
            self.count = 0
            self.state = "NON_TARGET"
        event = "LOCK" if self.state == "LOCKED" and old != "LOCKED" else "UNLOCK" if old == "LOCKED" and self.state != "LOCKED" else None
        return self.state, event


def merge_intervals(intervals, duration):
    result = []
    for s in sorted(intervals, key=lambda x: x["start_ms"]):
        a, b = float(s["start_ms"]), float(s["end_ms"])
        if not 0 <= a < b <= duration + 1:
            raise ValueError("人工标注必须位于录音时间范围内")
        if result and a <= result[-1]["end_ms"]:
            result[-1]["end_ms"] = max(result[-1]["end_ms"], b)
        else:
            result.append({"start_ms": a, "end_ms": b})
    return result


def compute_metrics(timeline, duration_ms, ground_truth=None):
    locks = [r["decision_ms"] for r in timeline if r.get("event") == "LOCK"]
    unlocks = [r["decision_ms"] for r in timeline if r.get("event") == "UNLOCK"]
    base = {"first_lock_observed_ms": locks[0] if locks else None, "lock_events": len(locks), "unlock_events": len(unlocks),
        "first_lock_latency_ms": None, "false_locks": None, "unlock_latency_ms": None, "relock_latency_ms": None, "target_coverage": None,
        "ground_truth_available": ground_truth is not None, "metric_note": "未提供人工 Ground Truth，正式准确率和时延指标不可计算"}
    if ground_truth is None:
        return base
    gt = merge_intervals(ground_truth, duration_ms)
    locked = []
    for i, r in enumerate(timeline):
        if r["state"] == "LOCKED":
            a, b = r["decision_ms"], timeline[i+1]["decision_ms"] if i+1<len(timeline) else duration_ms
            if b > a:
                locked.append((a, min(b, duration_ms)))
    cover = sum(max(0, min(b, t["end_ms"])-max(a, t["start_ms"])) for a,b in locked for t in gt)
    total = sum(t["end_ms"]-t["start_ms"] for t in gt)
    delays = [next((max(0, a-t["start_ms"]) for a,b in locked if b>t["start_ms"] and a<t["end_ms"]), None) for t in gt]
    unlock_delays = []
    for t in gt:
        end = t["end_ms"]
        if end >= duration_ms:
            continue
        if any(a < end < b for a,b in locked):
            next_start = next((x["start_ms"] for x in gt if x["start_ms"]>end), duration_ms)
            u = next((u for u in unlocks if end <= u < next_start), None)
            unlock_delays.append(u-end if u is not None else None)
        else:
            unlock_delays.append(0.0)
    relocks = [d for d in delays[1:] if d is not None]
    valid_unlocks = [d for d in unlock_delays if d is not None]
    base.update(first_lock_latency_ms=delays[0] if delays else None,
        false_locks=sum(not any(t["start_ms"]<=x<t["end_ms"] for t in gt) for x in locks),
        unlock_latency_ms=float(np.mean(valid_unlocks)) if valid_unlocks else None,
        relock_latency_ms=float(np.mean(relocks)) if relocks else None,
        target_coverage=cover/total if total else None, per_target_lock_latency_ms=delays,
        per_target_unlock_latency_ms=unlock_delays, missed_target_intervals=sum(d is None for d in delays),
        metric_note="时延按音频开始到实际决策时间计算，包含1.5秒滑窗、0.5秒步长和推理耗时；覆盖率按LOCKED时间与人工标注交集计算")
    return base


class Runtime:
    sample_rate = 16000
    hop_samples = 8000
    window_samples = 24000

    def __init__(self, profile, config):
        self.profile, self.config = profile, config
        self.machine = LockMachine(config["lock_threshold"], config["unlock_threshold"], config["confirm_windows"])
        self.buffer = np.empty(0, np.float32)
        self.total = 0
        self.timeline = []
        self.started = time.perf_counter()

    def feed(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        if not np.all(np.isfinite(chunk)) or np.max(np.abs(chunk), initial=0) > 1.01:
            raise ValueError("PCM 必须为 [-1,1] 范围的有限数值")
        self.total += len(chunk)
        self.buffer = np.concatenate((self.buffer, chunk))[-self.window_samples:]
        start = time.perf_counter()
        similarity = None
        if len(self.buffer) >= self.window_samples:
            # Require speech in the most recent hop; trailing silence must unlock.
            segs = speech_segments(self.buffer, 0.2)
            voiced = [s for s in segs if s["end_ms"] > 1000]
            if voiced:
                parts = [self.buffer[int(s["start_ms"]*16):int(s["end_ms"]*16)] for s in segs]
                v = embedding(self.profile["model_id"], np.concatenate(parts))
                similarity = float(np.clip(np.dot(v, self.profile["embedding"]), -1, 1))
        state, event = self.machine.advance(similarity)
        row = {"audio_time_ms": self.total/16, "decision_ms": (time.perf_counter()-self.started)*1000,
               "similarity": similarity, "state": state, "event": event, "inference_ms": (time.perf_counter()-start)*1000}
        self.timeline.append(row)
        return row

    def output(self, ground_truth=None):
        return {"timeline": self.timeline, "duration_ms": self.total/16,
            "metrics": compute_metrics(self.timeline, self.total/16, ground_truth), "ground_truth": ground_truth,
            "runtime": {"sample_rate": 16000, "window_ms": 1500, "hop_ms": 500}}
