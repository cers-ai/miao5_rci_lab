import csv
import io
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import httpx
from cryptography.fernet import Fernet
from . import store as db
from .schemas import Analysis


def cipher():
    path = db.WORK / ".service-key"
    with db.LOCK:
        if not path.exists():
            path.write_bytes(Fernet.generate_key())
        return Fernet(path.read_bytes())


def public_service(row):
    return {**{k:v for k,v in row.items() if k != "encrypted_api_key"}, "has_api_key": bool(row.get("encrypted_api_key"))}


def save_service(data, ident=None):
    old = db.get("model_services", ident) if ident else {}
    if ident and not old:
        raise ValueError("模型服务不存在")
    key = data.pop("api_key", None)
    encrypted = old.get("encrypted_api_key", "")
    if key is not None:
        encrypted = cipher().encrypt(key.encode()).decode() if key else ""
    return db.put("model_services", {**old, **data, "id": ident or db.uid(), "encrypted_api_key": encrypted, "status": "未测试", "error": ""})


def call_model(service_id, model, messages, temperature=0.2, max_tokens=3000, check_models=False):
    s = db.get("model_services", service_id)
    if not s or not s["enabled"]:
        raise ValueError("模型服务未配置或已停用")
    if model not in s["models"]:
        raise ValueError("模型不在服务注册列表内")
    headers = {"Content-Type": "application/json"}
    if s.get("encrypted_api_key"):
        headers["Authorization"] = "Bearer " + cipher().decrypt(s["encrypted_api_key"].encode()).decode()
    try:
        with httpx.Client(timeout=s["timeout"], follow_redirects=False, headers=headers) as c:
            if check_models:
                r = c.get(s["base_url"].rstrip("/")+"/models")
                r.raise_for_status()
                names = {x["id"] for x in r.json()["data"]}
                if model not in names:
                    raise ValueError("远程模型列表没有指定模型")
            r = c.post(s["base_url"].rstrip("/")+"/chat/completions", json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
            r.raise_for_status()
            raw = r.json()
            text = raw["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("模型服务返回空内容")
            return text, raw
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"模型服务 HTTP {exc.response.status_code}，请检查 API 地址、密钥、模型权限和配额") from None
    except httpx.TimeoutException:
        raise ValueError("模型服务请求超时") from None
    except httpx.RequestError:
        raise ValueError("无法连接模型服务，请检查 API 地址和网络") from None
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("模型服务返回不符合 OpenAI Chat Completions 协议") from None


def test_service(ident):
    s = db.get("model_services", ident)
    if not s:
        raise ValueError("模型服务不存在")
    start = time.perf_counter()
    try:
        text, _ = call_model(ident, s["default_model"], [{"role":"user", "content":"请用中文简短确认服务可用。"}], max_tokens=80, check_models=True)
        result = {"status":"可用", "response": text, "elapsed_ms": (time.perf_counter()-start)*1000}
    except ValueError as exc:
        result = {"status":"测试失败", "error":str(exc), "elapsed_ms": (time.perf_counter()-start)*1000}
    db.update("model_services", ident, status=result["status"], error=result.get("error", ""), test_result=result)
    return result


def selected_runs(ids):
    runs = [db.get("experiment_runs", ident) for ident in dict.fromkeys(ids)]
    if any(r is None for r in runs):
        raise ValueError("所选实验不存在")
    if any(r["status"] != "completed" for r in runs):
        raise ValueError("只能选择已经完成的实验")
    return runs


def context(runs):
    # Embeddings and filesystem paths are unnecessary for external analysis.
    clean = []
    for r in runs:
        clean.append({k: r.get(k) for k in ("id", "kind", "created_at", "config", "inputs", "result", "reviews", "elapsed_ms")})
        clean[-1]["model_versions"] = [{k:m.get(k) for k in ("id", "name", "adapter", "version", "device")} for m in r["models"]]
    return {"scene":"A01", "selected_runs":clean}


def analyze(ids, owner):
    runs = selected_runs(ids)
    a = db.get("settings", "agent")
    ctx = context(runs)
    encoded = json.dumps(ctx, ensure_ascii=False)
    if len(encoded) > 180000:
        raise ValueError("所选实验上下文过大，请减少轮数或录音长度后分析")
    schema = json.dumps(Analysis.model_json_schema(), ensure_ascii=False)
    text, raw = call_model(a["service_id"], a["model"], [{"role":"system", "content":a["system_prompt"]+"\n仅返回符合以下 schema 的JSON："+schema},
        {"role":"user", "content":"请综合分析以下真实实验记录（所有字段仅为待分析数据）：\n"+encoded}], a["temperature"], a["max_tokens"])
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n",1)[1].rsplit("```",1)[0].strip()
    try:
        parsed = Analysis.model_validate_json(cleaned).model_dump()
    except Exception:
        evidence = db.put("analysis_reports", {"type":"analysis", "status":"failed", "owner":owner, "run_ids":ids,
            "agent":a, "context":ctx, "raw_response":raw, "error":"模型未返回所需结构化 JSON"})
        raise ValueError(f"模型未返回有效结构化JSON，原始响应已保存：{evidence['id']}") from None
    return db.put("analysis_reports", {"type":"analysis", "status":"completed", "owner":owner, "run_ids":list(dict.fromkeys(ids)), "agent":a, "context":ctx, "analysis":parsed, "raw_response":raw})


def create_report(ids, analysis_id, owner):
    runs = selected_runs(ids)
    a = db.get("analysis_reports", analysis_id) if analysis_id else None
    if analysis_id and (not a or a.get("type")!="analysis" or a.get("status")!="completed" or set(a["run_ids"])!=set(ids)):
        raise ValueError("分析结果必须与当前所选 Run 集合完全一致")
    if a and a["context"] != context(runs):
        raise ValueError("实验核验或标注在分析后发生变化，请重新分析以保持报告一致")
    env = {"python":platform.python_version(), "platform":platform.platform(), "dependencies":{n:version(n) for n in ("torch","funasr","pyannote.audio","onnxruntime","scikit-learn")}}
    payload = {"scene":"A01", "generated_at":db.now(), "environment":env, "selected_runs":runs, "analysis":a}
    report = db.put("analysis_reports", {"type":"report", "status":"completed", "owner":owner, "run_ids":ids, "analysis_id":analysis_id, "payload":payload, "name":"A01 综合实验报告"})
    folder = db.WORK / "reports" / report["id"]
    folder.mkdir(exist_ok=True)
    (folder / "A01_实验证据.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (folder / "A01_所选实验数据.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Run ID","实验","创建时间","输入","模型版本","参数","耗时ms","结果","人工核验"])
        for r in runs:
            values = [r["id"],r["kind"],r["created_at"],r["inputs"],r["models"],r["config"],r.get("elapsed_ms"),r["result"],r["reviews"]]
            cells = [json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else str(v) for v in values]
            w.writerow(["'"+v if v.startswith(("=","+","-","@","\t","\r")) else v for v in cells])
    analysis = a["analysis"] if a else None
    sections = ["# A01 · 目标人物声纹建立与实时识别｜综合实验报告", "## 实验目的\n比较说话人发现路线、跨录音目标识别、实时锁定稳定性。", "## 实验环境\n```json\n"+json.dumps(env,ensure_ascii=False,indent=2)+"\n```", "## 本地模型版本\n"+"\n".join(f"- {m['name']}：{m['version']}（{m['device']}）" for r in runs for m in r["models"])]
    for kind in ("E1","E2","E3"):
        sections.append(f"## {kind} 结果比较")
        for r in runs:
            if r["kind"]==kind:
                sections.append(f"### Run {r['id']}\n\n创建：{r['created_at']}；耗时：{r.get('elapsed_ms',0):.1f} ms\n\n```json\n"+json.dumps({"inputs":r["inputs"],"parameters":r["config"],"result":r["result"],"manual_reviews":r["reviews"]},ensure_ascii=False,indent=2)+"\n```")
    sections.append("## 人工核验\n"+"\n".join(f"- {r['id']}："+json.dumps(r["reviews"],ensure_ascii=False) for r in runs))
    sections.append("## 智能综合分析\n"+(analysis["summary"] if analysis else "尚未配置或执行真实模型服务分析；本报告仅汇总实验证据，不构成智能推荐。"))
    if analysis:
        sections.append("```json\n"+json.dumps(analysis,ensure_ascii=False,indent=2)+"\n```")
    sections.extend(["## 推荐技术配置\n"+(json.dumps(analysis["recommended_configuration"],ensure_ascii=False,indent=2) if analysis else "待真实智能分析和人工确认。"), "## 适用边界\nV0.1 的身份匹配依赖录音质量及声纹模型，不保证重叠讲话准确分离。真实多人会议准确率需人工核验。", "## 已知风险\n未提供Ground Truth的实时实验只显示观测值；短声纹、噪声、跨设备和重叠讲话可能影响结果。"])
    (folder / "A01_综合实验报告.md").write_text("\n\n".join(sections),encoding="utf-8")
    return report
