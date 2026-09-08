import io
import json
import time
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from backend.app import models, store as db


def wait_run(client,ident,timeout=180):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        r=client.get(f'/api/a01/runs/{ident}').json()
        if r['status'] in ('completed','failed','cancelled'):
            assert r['status']=='completed',r
            return r
        time.sleep(.2)
    raise AssertionError('真实模型实验超时')


def wav_bytes(audio):
    f=io.BytesIO();sf.write(f,audio,16000,format='WAV',subtype='PCM_16');return f.getvalue()


def test_real_embedding_discriminates_speakers(real_models):
    a,_=sf.read(real_models/'speaker1_a_cn_16k.wav',dtype='float32')
    b,_=sf.read(real_models/'speaker1_b_cn_16k.wav',dtype='float32')
    other,_=sf.read(real_models/'speaker2_a_cn_16k.wav',dtype='float32')
    results={}
    for ident in ('campp','wespeaker'):
        va,vb,vc=[models.embedding(ident,x) for x in (a,b,other)]
        same,different=float(va@vb),float(va@vc)
        assert same>different,(ident,same,different)
        assert len(va)>=192 and np.isfinite(va).all()
        results[ident]={'same_speaker':same,'different_speaker':different,'dimensions':len(va)}
    assert models.speech_segments(a)
    assert models.speech_segments(np.zeros(48000,dtype='float32'))==[]
    (Path(__file__).resolve().parents[1]/'workspace/logs/real-embedding-test.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


def test_real_full_experiment_pipeline(admin,real_models):
    a,_=sf.read(real_models/'speaker1_a_cn_16k.wav',dtype='float32')
    b,_=sf.read(real_models/'speaker1_b_cn_16k.wav',dtype='float32')
    other,_=sf.read(real_models/'speaker2_a_cn_16k.wav',dtype='float32')
    # Audio concatenation for a controlled two-person fixture; all speech is real human speech.
    meeting=np.concatenate((a,np.zeros(16000,dtype=np.float32),other,np.zeros(16000,dtype=np.float32),b))
    f=admin.post('/api/files',files={'file':('真实双人组合测试.wav',wav_bytes(meeting),'audio/wav')}).json()
    configs=[{'model_id':'campp','speaker_count':2},{'model_id':'campp','speaker_count':None,'clustering_threshold':.65},{'model_id':'wespeaker','speaker_count':2}]
    completed=[]
    for c in configs:
        run=admin.post('/api/a01/e1/run',json={'file_id':f['id'],**c}).json()
        r=wait_run(admin,run['id'])
        assert r['result']['segments'] and len(r['models'])==2
        completed.append(r)
    src=completed[0]
    first_speaker=src['result']['segments'][0]['speaker_id']
    profile=admin.post('/api/a01/target-profile',json={'name':'公开样本Speaker1','run_id':src['id'],'speaker_ids':[first_speaker],'model_id':'wespeaker'})
    assert profile.status_code==200,profile.text
    p=profile.json()
    assert len(p['embedding'])>=192 and p['source_segments']
    testing=[]
    for name,data in [('speaker1-b.wav',b),('speaker2.wav',other)]:
        testing.append(admin.post('/api/files',files={'file':(name,wav_bytes(data),'audio/wav')}).json()['id'])
    run=admin.post('/api/a01/e2/run',json={'profile_id':p['id'],'file_ids':testing,'model_id':'wespeaker','similarity_threshold':.6}).json()
    e2=wait_run(admin,run['id'])
    assert len(e2['result']['files'])==2
    assert e2['result']['files'][0]['max_similarity']>e2['result']['files'][1]['max_similarity']
    before=time.perf_counter()
    run=admin.post('/api/a01/e3/replay',json={'profile_id':p['id'],'file_id':testing[0],'lock_threshold':.55,'unlock_threshold':.4,'confirm_windows':1}).json()
    e3=wait_run(admin,run['id'])
    assert time.perf_counter()-before>=len(b)/16000
    assert e3['result']['timeline'] and any(x['similarity'] is not None for x in e3['result']['timeline'])
    assert e3['result']['metrics']['false_locks'] is None
    gt=admin.put(f"/api/a01/runs/{e3['id']}/ground-truth",json={'intervals':[{'start_ms':0,'end_ms':len(b)/16}]})
    assert gt.status_code==200 and gt.json()['result']['metrics']['ground_truth_available']
    ids=[r['id'] for r in completed]+[e2['id'],e3['id']]
    report=admin.post('/api/a01/report',json={'run_ids':ids})
    assert report.status_code==200,report.text
    evidence=admin.get(f"/api/reports/{report.json()['id']}/json").json()
    assert len(evidence['selected_runs'])==5
    evidence['test_note']='真人公开语音组合的自动链路验证，不替代真实会议质量验收'
    out=Path(__file__).resolve().parents[1]/'workspace/logs/real-pipeline-test.json'
    out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    # Model mismatch must fail and be persisted, never silently compare unrelated embedding spaces.
    bad=admin.post('/api/a01/e2/run',json={'profile_id':p['id'],'file_ids':testing,'model_id':'campp'}).json()
    end=time.monotonic()+20
    while time.monotonic()<end:
        r=admin.get(f"/api/a01/runs/{bad['id']}").json()
        if r['status']=='failed':break
        time.sleep(.1)
    assert r['status']=='failed' and '一致' in r['error']
    # Microphone protocol uses the same real Runtime, with PCM from a known real recording.
    with admin.websocket_connect('/api/a01/e3/realtime') as ws:
        ws.send_json({'profile_id':p['id'],'lock_threshold':.55,'unlock_threshold':.4,'confirm_windows':1})
        ready=ws.receive_json();assert ready['type']=='ready',ready
        for i in range(0,32000,8000):
            time.sleep(.5)
            ws.send_bytes((b[i:i+8000]*32767).astype('<i2').tobytes())
            message=ws.receive_json();assert message['type']=='window',message
        ws.send_json({'type':'mark','label':'测试标记'})
        ws.send_json({'type':'stop'})
        done=ws.receive_json();assert done['type']=='completed',done
    mic=admin.get(f"/api/a01/runs/{ready['run_id']}").json()
    assert mic['result']['manual_events'] and mic['inputs']
