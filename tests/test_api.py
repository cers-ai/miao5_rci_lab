import io
import json
import time
from backend.app import store as db


def test_auth_and_origin(client):
    client.post('/api/logout')
    assert client.get('/api/files').status_code==401
    assert client.post('/api/login',json={'username':'admin','password':'bad'}).status_code==401
    assert client.post('/api/login',json={'username':'admin','password':'abcd@1234'},headers={'origin':'http://evil.example'}).status_code==403


def test_users_permissions_and_password_storage(admin):
    name='testuser'+db.uid()
    r=admin.post('/api/users',json={'username':name,'password':'test123456','role':'实验员'})
    assert r.status_code==200 and 'password_hash' not in r.json()
    assert admin.post('/api/users',json={'username':name,'password':'test123456'}).status_code==400
    assert admin.put('/api/users/admin',json={'role':'管理员','enabled':False}).status_code==400
    assert admin.post('/api/login',json={'username':name,'password':'test123456'}).status_code==200
    assert admin.get('/api/users').status_code==403
    assert admin.post('/api/models/local/campp/test').status_code==403
    assert admin.post('/api/password',json={'current_password':'bad','new_password':'new123456'}).status_code==400
    assert admin.post('/api/password',json={'current_password':'test123456','new_password':'new123456'}).status_code==200
    assert admin.get('/api/me').status_code==401
    assert 'test123456' not in json.dumps(db.all_rows('users'))


def test_files_real_transcode_and_reference_guard(admin,audio_bytes):
    assert admin.post('/api/files',files={'file':('bad.wav',b'not audio','audio/wav')}).status_code==400
    assert admin.post('/api/files',files={'file':('empty.wav',b'','audio/wav')}).status_code==400
    r=admin.post('/api/files',files={'file':('../../test.wav',audio_bytes,'audio/wav')})
    assert r.status_code==200,r.text
    f=r.json()
    assert f['name']=='test.wav' and f['sample_rate']==16000 and f['duration_ms']==1000
    a=admin.get(f"/api/files/{f['id']}/audio")
    assert a.status_code==200 and a.content[:4]==b'RIFF'
    ranged=admin.get(f"/api/files/{f['id']}/audio",headers={'Range':'bytes=0-99'})
    assert ranged.status_code==206 and len(ranged.content)==100
    db.put('experiment_runs',{'kind':'E1','status':'failed','inputs':[{'id':f['id']}],'config':{},'result':None,'reviews':[]})
    assert admin.delete(f"/api/files/{f['id']}").status_code==409
    fresh=admin.post('/api/files',files={'file':('delete.wav',audio_bytes,'audio/wav')}).json()
    assert admin.delete(f"/api/files/{fresh['id']}").status_code==200
    assert admin.get(f"/api/files/{fresh['id']}").status_code==404


def test_model_failure_is_real(admin):
    model={'name':'missing','capability':'目标说话人识别','adapter':'wespeaker','path':'Z:/definitely-missing-model','device':'cpu'}
    m=admin.post('/api/models/local',json=model).json()
    res=admin.post(f"/api/models/local/{m['id']}/test").json()
    assert res['status']=='测试失败' and res['error']
    saved=db.get('local_models',m['id'])
    assert saved['status']=='测试失败'


def test_review_validation_and_report_selection(admin):
    r=db.put('experiment_runs',{'kind':'E1','status':'completed','inputs':[],'config':{},'result':{'speakers':[{'speaker_id':'SPEAKER_01'}]},'reviews':[],'models':[]})
    assert admin.put(f"/api/a01/runs/{r['id']}/review",json={'item_id':'missing','verdict':'正确'}).status_code==400
    ok=admin.put(f"/api/a01/runs/{r['id']}/review",json={'item_id':'SPEAKER_01','verdict':'同一人被拆成多个Speaker','note':'测试核验'})
    assert ok.status_code==200 and ok.json()['reviews'][0]['note']=='测试核验'
    again=admin.get(f"/api/a01/runs/{r['id']}").json()
    assert again['reviews'][0]['reviewer']=='admin'
    assert admin.post('/api/a01/analyze',json={'run_ids':[r['id']]}).status_code==400
    assert admin.post('/api/a01/report',json={'run_ids':['missing']}).status_code==400
