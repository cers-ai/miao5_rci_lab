import asyncio
import hashlib
import json
import secrets
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from . import experiments as ex, schemas as s, services, store as db
from .models import INFERENCE_LOCK, load, test_model
from .realtime import Runtime, compute_metrics, merge_intervals


@asynccontextmanager
async def lifespan(app):
    for r in db.all_rows("experiment_runs"):
        if r["status"] in ("running", "queued"):
            db.update("experiment_runs",r["id"],status="failed",error="服务重启中断了实验，已有输出保留")
    for m in db.all_rows("local_models"):
        if m["enabled"] and m.get("manifest") and (db.WORK/"samples/speaker1_a_cn_16k.wav").exists():
            db.update("local_models",m["id"],status="未测试",error="启动加载测试中")
            ex.POOL.submit(test_model,m["id"])
    yield


app = FastAPI(title="妙悟实时认知智能实验室", version="0.1.0", lifespan=lifespan)
ALLOWED_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"}
LOGIN_ATTEMPTS = {}


@app.middleware("http")
async def origin_guard(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("origin") and request.headers["origin"] not in ALLOWED_ORIGINS:
        return JSONResponse({"detail":"不允许跨站修改请求"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"detail":str(exc)},status_code=400)


def user_from_token(token):
    session = db.get("sessions", hashlib.sha256(token.encode()).hexdigest()) if token else None
    if not session or session["expires"] < time.time():
        raise HTTPException(401, "请先登录")
    u = db.get("users", session["user_id"])
    if not u or not u["enabled"]:
        raise HTTPException(401,"用户已停用")
    return {k:v for k,v in u.items() if k!="password_hash"}


def current_user(request: Request):
    return user_from_token(request.cookies.get("miaowu_session"))


def admin(u=Depends(current_user)):
    if u["role"]!="管理员":
        raise HTTPException(403,"需要管理员权限")
    return u


def require(table, ident):
    row = db.get(table, ident)
    if row is None:
        raise HTTPException(404,"记录不存在")
    return row


@app.get("/api/health")
def health():
    return {"status":"ok", "version":"0.1.0", "ffmpeg":bool(shutil.which("ffmpeg")), "frontend_port":3000}


@app.post("/api/login")
def login(body:s.Login, request:Request, response:Response):
    key = request.client.host if request.client else "local"
    attempts = [t for t in LOGIN_ATTEMPTS.get(key,[]) if time.time()-t<60]
    if len(attempts)>=10:
        raise HTTPException(429,"登录尝试过多，请一分钟后重试")
    u = next((u for u in db.all_rows("users") if u["username"]==body.username),None)
    if not u or not db.verify_password(body.password,u["password_hash"]) or not u["enabled"]:
        LOGIN_ATTEMPTS[key] = attempts+[time.time()]
        raise HTTPException(401,"用户名、密码错误或账号已停用")
    LOGIN_ATTEMPTS.pop(key,None)
    token = secrets.token_urlsafe(32)
    db.put("sessions", {"id":hashlib.sha256(token.encode()).hexdigest(),"user_id":u["id"],"expires":time.time()+12*3600})
    response.set_cookie("miaowu_session",token,httponly=True,samesite="strict",max_age=12*3600)
    return {k:v for k,v in u.items() if k!="password_hash"}


@app.post("/api/logout")
def logout(request:Request,response:Response):
    token = request.cookies.get("miaowu_session", "")
    db.remove("sessions",hashlib.sha256(token.encode()).hexdigest())
    response.delete_cookie("miaowu_session")
    return {"ok":True}


@app.get("/api/me")
def me(u=Depends(current_user)):
    return u


@app.post("/api/password")
def password(body:s.Password,u=Depends(current_user)):
    row = require("users",u["id"])
    if not db.verify_password(body.current_password,row["password_hash"]):
        raise ValueError("当前密码不正确")
    db.update("users",u["id"],password_hash=db.hash_password(body.new_password))
    for session in db.all_rows("sessions"):
        if session["user_id"]==u["id"]:
            db.remove("sessions",session["id"])
    return {"ok":True,"message":"密码已修改，请重新登录"}


@app.get("/api/users")
def users(u=Depends(admin)):
    return [{k:v for k,v in row.items() if k!="password_hash"} for row in db.all_rows("users")]


@app.post("/api/users")
def create_user(body:s.UserCreate,u=Depends(admin)):
    if len(body.password)<8:
        raise ValueError("密码至少8位")
    with db.LOCK:
        if any(x["username"]==body.username for x in db.all_rows("users")):
            raise ValueError("用户名已存在")
        row = db.put("users", {"username":body.username,"password_hash":db.hash_password(body.password),"role":body.role,"enabled":body.enabled})
    return {k:v for k,v in row.items() if k!="password_hash"}


@app.put("/api/users/{ident}")
def update_user(ident:str,body:s.UserUpdate,u=Depends(admin)):
    require("users",ident)
    if ident==u["id"] and (not body.enabled or body.role!="管理员"):
        raise ValueError("不能停用或降级当前管理员")
    row = db.update("users",ident,**body.model_dump())
    return {k:v for k,v in row.items() if k!="password_hash"}


@app.get("/api/files")
def files(u=Depends(current_user)):
    return db.all_rows("experiment_files")


@app.post("/api/files")
def upload(file:UploadFile,u=Depends(current_user)):
    ident = db.uid()
    folder = db.WORK / "uploads" / ident
    folder.mkdir()
    original, normalized = folder/"original",folder/"audio.wav"
    try:
        size = 0
        with original.open("wb") as out:
            while data := file.file.read(1024*1024):
                size += len(data)
                if size>500*1024*1024:
                    raise ValueError("录音最大500MB")
                out.write(data)
        if not size:
            raise ValueError("不能上传空文件")
        probe = subprocess.run(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(original)],capture_output=True,text=True,timeout=30)
        if probe.returncode:
            raise ValueError("无法读取音频，请上传有效 WAV、MP3、M4A、FLAC 或 OGG 文件")
        meta = json.loads(probe.stdout)
        stream = next((x for x in meta["streams"] if x["codec_type"]=="audio"),None)
        if not stream:
            raise ValueError("文件不含音轨")
        duration = float(meta.get("format",{}).get("duration",stream.get("duration",0)))
        if duration > 7200:
            raise ValueError("单个录音不能超过2小时")
        result = subprocess.run(["ffmpeg","-v","error","-nostdin","-y","-i",str(original),"-vn","-ac","1","-ar","16000","-t","7201","-c:a","pcm_s16le",str(normalized)],capture_output=True,timeout=300)
        if result.returncode:
            raise ValueError("音频转码失败")
        info = sf.info(normalized)
        if info.duration < 0.2 or info.duration>7200:
            raise ValueError("录音时长须在0.2秒至2小时之间")
        return db.put("experiment_files", {"id":ident,"name":Path((file.filename or 'audio').replace('\\','/')).name,
            "duration_ms":info.duration*1000,"sample_rate":int(stream["sample_rate"]),"channels":int(stream["channels"]),
            "normalized_sample_rate":16000,"bytes":size,"sha256":db.digest_file(original),"normalized_sha256":db.digest_file(normalized),
            "original_path":str(original),"normalized_path":str(normalized),"owner":u["id"]})
    except Exception:
        shutil.rmtree(folder)
        raise
    finally:
        file.file.close()


@app.get("/api/files/{ident}")
def get_file(ident:str,u=Depends(current_user)):
    return require("experiment_files",ident)


@app.get("/api/files/{ident}/audio")
def get_audio(ident:str,u=Depends(current_user)):
    f = require("experiment_files",ident)
    return FileResponse(f["normalized_path"],media_type="audio/wav")


@app.delete("/api/files/{ident}")
def delete_file(ident:str,u=Depends(current_user)):
    f = require("experiment_files",ident)
    if any(any(x["id"]==ident for x in r.get("inputs",[])) for r in db.all_rows("experiment_runs")) or any(p["source_file_id"]==ident for p in db.all_rows("target_profiles")):
        raise HTTPException(409,"文件已被实验或声纹引用，不能删除实验证据")
    folder = Path(f["normalized_path"]).parent.resolve()
    if not folder.is_relative_to((db.WORK/"uploads").resolve()) or folder.name!=ident:
        raise ValueError("文件路径异常")
    shutil.rmtree(folder)
    db.remove("experiment_files",ident)
    return {"ok":True}


@app.get("/api/models/local")
def local_models(u=Depends(current_user)):
    return db.all_rows("local_models")


def save_model(body,ident=None):
    old = require("local_models",ident) if ident else {}
    data = body.model_dump()
    if (body.load_mode=="命令行进程") != (body.adapter=="command"):
        raise ValueError("命令行加载方式须选择 command 适配器")
    # Edits invalidate availability and cached identity until an actual test is run.
    version = old.get("version","")
    if any(old.get(k)!=data.get(k) for k in ("path","adapter","load_mode")):
        version = "local-"+db.uid()
    return db.put("local_models",{**old,**data,"id":ident or db.uid(),"version":version,"status":"未测试","error":""})


@app.post("/api/models/local")
def add_model(body:s.LocalModel,u=Depends(admin)):
    return save_model(body)


@app.put("/api/models/local/{ident}")
def update_model(ident:str,body:s.LocalModel,u=Depends(admin)):
    return save_model(body,ident)


@app.post("/api/models/local/{ident}/test")
def test_local(ident:str,u=Depends(admin)):
    require("local_models",ident)
    return test_model(ident)


@app.post("/api/models/local/{ident}/download")
def download_local(ident:str,u=Depends(admin)):
    from scripts.download_models import install
    require("local_models",ident)
    if ident not in {x[0] for x in db.PRESETS}:
        raise ValueError("此模型不是预设模型，请手动安装后填写路径")
    if db.get("local_models",ident)["status"]=="下载中":
        raise HTTPException(409,"模型正在下载")
    db.update("local_models",ident,status="下载中")
    ex.POOL.submit(install,ident)
    return {"status":"下载中"}


@app.get("/api/model-services")
def list_services(u=Depends(admin)):
    return [services.public_service(x) for x in db.all_rows("model_services")]


@app.post("/api/model-services")
def add_service(body:s.Service,u=Depends(admin)):
    return services.public_service(services.save_service(body.model_dump()))


@app.put("/api/model-services/{ident}")
def update_service(ident:str,body:s.Service,u=Depends(admin)):
    return services.public_service(services.save_service(body.model_dump(),ident))


@app.post("/api/model-services/{ident}/test")
def service_test(ident:str,u=Depends(admin)):
    return services.test_service(ident)


@app.get("/api/agent")
def get_agent(u=Depends(current_user)):
    return db.get("settings","agent")


@app.put("/api/agent")
def set_agent(body:s.Agent,u=Depends(admin)):
    service = require("model_services",body.service_id)
    if body.model not in service["models"]:
        raise ValueError("助手模型不属于所选服务")
    return db.put("settings",{**body.model_dump(),"id":"agent"})


@app.post("/api/agent/test")
def agent_test(u=Depends(admin)):
    a = db.get("settings","agent")
    text,_ = services.call_model(a["service_id"],a["model"],[{"role":"system","content":a["system_prompt"]},{"role":"user","content":"这是一项连通测试，没有实验数据。请确认已准备好分析，并说明证据不足时不会编造结论。"}],a["temperature"],min(a["max_tokens"],300))
    return {"response":text}


@app.post("/api/a01/e1/run")
def e1(body:s.E1,u=Depends(current_user)):
    model = require("local_models",body.model_id)
    c = body.model_dump()
    if model["adapter"]=="pyannote":
        c.pop("clustering_threshold")
        c.pop("min_speech_duration")
    elif c["speaker_count"]:
        c.pop("clustering_threshold")
    return ex.queue("E1",c,u["id"])


@app.post("/api/a01/target-profile")
def profile(body:s.Profile,u=Depends(current_user)):
    return ex.create_profile(body.model_dump(),u["id"])


@app.get("/api/a01/target-profiles")
def profiles(u=Depends(current_user)):
    return [{k:v for k,v in r.items() if k!="embedding"} for r in db.all_rows("target_profiles")]


@app.post("/api/a01/e2/run")
def e2(body:s.E2,u=Depends(current_user)):
    return ex.queue("E2",body.model_dump(),u["id"])


@app.post("/api/a01/e3/replay")
def e3(body:s.E3,u=Depends(current_user)):
    if not body.file_id:
        raise ValueError("回放实验须选择录音")
    if body.ground_truth is not None:
        merge_intervals([x.model_dump() for x in body.ground_truth],require("experiment_files",body.file_id)["duration_ms"])
    return ex.queue("E3",body.model_dump(),u["id"],"replay")


@app.get("/api/a01/runs")
def runs(u=Depends(current_user)):
    return [{k:v for k,v in r.items() if k not in ("target_profile",)} for r in db.all_rows("experiment_runs")]


@app.get("/api/a01/runs/{ident}")
def run(ident:str,u=Depends(current_user)):
    return require("experiment_runs",ident)


@app.post("/api/a01/runs/{ident}/cancel")
def cancel(ident:str,u=Depends(current_user)):
    r = require("experiment_runs",ident)
    if r["status"] not in ("queued","running"):
        raise ValueError("实验已结束")
    if ident in ex.CANCEL:
        ex.CANCEL[ident].set()
    return {"ok":True}


@app.put("/api/a01/runs/{ident}/review")
def review(ident:str,body:s.Review,u=Depends(current_user)):
    with db.LOCK:
        r = require("experiment_runs",ident)
        if r["status"]!="completed":
            raise ValueError("只能核验已完成实验")
        valid = {"run"}
        if r["kind"]=="E1":
            valid.update(x["speaker_id"] for x in r["result"]["speakers"])
        if r["kind"]=="E2":
            valid.update(x["file_id"] for x in r["result"]["files"])
        if body.item_id not in valid:
            raise ValueError("核验对象不存在")
        item = {**body.model_dump(),"reviewer":u["username"],"updated_at":db.now()}
        reviews = [x for x in r["reviews"] if x["item_id"]!=body.item_id]+[item]
        db.put("experiment_run_items",{"id":f"{ident}:{body.item_id}","run_id":ident,**item})
        return db.update("experiment_runs",ident,reviews=reviews)


@app.put("/api/a01/runs/{ident}/ground-truth")
def ground_truth(ident:str,body:s.GroundTruth,u=Depends(current_user)):
    with db.LOCK:
        r = require("experiment_runs",ident)
        if r["kind"]!="E3" or r["status"]!="completed":
            raise ValueError("请选择已完成的E3实验")
        result = r["result"]
        gt = merge_intervals([x.model_dump() for x in body.intervals],result["duration_ms"])
        result.update(ground_truth=gt,ground_truth_reviewer=u["username"],ground_truth_updated_at=db.now(),metrics=compute_metrics(result["timeline"],result["duration_ms"],gt))
        ex.persist_result(ident,result)
        return db.get("experiment_runs",ident)


@app.post("/api/a01/analyze")
def analyze(body:s.Selection,u=Depends(current_user)):
    return services.analyze(body.run_ids,u["id"])


@app.post("/api/a01/report")
def report(body:s.Selection,u=Depends(current_user)):
    return services.create_report(body.run_ids,body.analysis_id,u["id"])


@app.get("/api/reports")
def reports(u=Depends(current_user)):
    return [{k:v for k,v in r.items() if k not in ("payload","context","raw_response")} for r in db.all_rows("analysis_reports")]


@app.get("/api/reports/{ident}/{extension}")
def download_report(ident:str,extension:str,u=Depends(current_user)):
    r = require("analysis_reports",ident)
    names = {"md":"A01_综合实验报告.md","csv":"A01_所选实验数据.csv","json":"A01_实验证据.json"}
    if r["type"]!="report" or extension not in names:
        raise HTTPException(404,"报告格式不存在")
    return FileResponse(db.WORK/"reports"/ident/names[extension],filename=names[extension])


@app.websocket("/api/a01/e3/realtime")
async def realtime(ws:WebSocket):
    origin = ws.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        await ws.close(code=1008)
        return
    try:
        u = user_from_token(ws.cookies.get("miaowu_session"))
    except HTTPException:
        await ws.close(code=1008)
        return
    await ws.accept()
    r,rt = None,None
    recording = None
    normal_stop = False
    try:
        body = s.E3.model_validate(await asyncio.wait_for(ws.receive_json(),timeout=20))
        c = body.model_dump()
        c["file_id"], c["ground_truth"] = None,None
        p = ex.checked_profile(c["profile_id"])
        await asyncio.to_thread(load,p["model_id"])
        await asyncio.to_thread(load,"vad")
        r = ex.new_run("E3",c,u["id"],"microphone")
        import threading
        ex.CANCEL[r["id"]] = threading.Event()
        db.update("experiment_runs",r["id"],status="running",models=[p["model"],ex.snapshot_model("vad")],target_profile=p)
        folder = db.WORK/"uploads"/r["id"]
        folder.mkdir()
        path = folder/"audio.wav"
        recording = sf.SoundFile(path,"w",samplerate=16000,channels=1,subtype="PCM_16")
        rt = Runtime(p,c)
        await ws.send_json({"type":"ready","run_id":r["id"],"sample_rate":16000,"chunk_samples":8000})
        pending = np.empty(0,np.float32)
        marks = []
        while True:
            message = await asyncio.wait_for(ws.receive(),timeout=30)
            if message["type"]=="websocket.disconnect":
                raise WebSocketDisconnect()
            ex.check_cancel(r["id"])
            if message.get("text"):
                control = json.loads(message["text"])
                if control.get("type")=="stop":
                    normal_stop = True
                    break
                if control.get("type")=="mark":
                    marks.append({"audio_time_ms":rt.total/16,"label":str(control.get("label",""))[:100],"reviewer":u["username"]})
                continue
            data = message.get("bytes",b"")
            if not data or len(data)%2 or len(data)>32000:
                raise ValueError("需要16kHz单声道PCM16，每包不超过1秒")
            chunk = np.frombuffer(data,dtype="<i2").astype(np.float32)/32768
            recording.write(chunk)
            pending = np.concatenate((pending,chunk))
            if rt.total/16000>7200:
                raise ValueError("实时实验最长2小时")
            while len(pending)>=8000:
                row = await asyncio.to_thread(rt.feed,pending[:8000])
                pending = pending[8000:]
                db.update("experiment_runs",r["id"],result={**rt.output(),"manual_events":marks},progress=f"麦克风运行 {rt.total/16000:.1f} 秒")
                await ws.send_json({"type":"window",**row})
        if len(pending)>=3200:
            await asyncio.to_thread(rt.feed,pending)
        ex.persist_result(r["id"],{**rt.output(),"manual_events":marks})
        db.update("experiment_runs",r["id"],status="completed",progress="完成",elapsed_ms=(time.perf_counter()-rt.started)*1000)
        await ws.send_json({"type":"completed","run_id":r["id"],"result":rt.output()})
    except (Exception, asyncio.CancelledError) as exc:
        if r:
            if rt:
                ex.persist_result(r["id"],rt.output())
            db.update("experiment_runs",r["id"],status="cancelled" if isinstance(exc,(WebSocketDisconnect,InterruptedError)) else "failed",error="麦克风连接中断，已有结果已保存" if isinstance(exc,WebSocketDisconnect) else str(exc)[:1000])
        try:
            await ws.send_json({"type":"error","error":str(exc)[:1000] or "连接中断"})
        except Exception:
            pass
    finally:
        if recording:
            recording.close()
            info = sf.info(path)
            f = db.put("experiment_files",{"id":r["id"],"name":f"麦克风_{r['id']}.wav","owner":u["id"],"duration_ms":info.duration*1000,
                "sample_rate":16000,"channels":1,"normalized_sample_rate":16000,"bytes":path.stat().st_size,"sha256":db.digest_file(path),"normalized_sha256":db.digest_file(path),"original_path":str(path),"normalized_path":str(path)})
            db.update("experiment_runs",r["id"],inputs=[{k:f[k] for k in ("id","name","sha256","duration_ms")}])
            ex.CANCEL.pop(r["id"],None)
        try:
            await ws.close()
        except Exception:
            pass
