import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from backend.app import services, store as db


@pytest.fixture
def protocol_server():
    """Isolated HTTP protocol fixture, never registered in the user's workspace."""
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(json.dumps({'data':[{'id':'protocol-test-model'}]}).encode())
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            result={'summary':'隔离协议测试的结构化响应，不代表真实LLM分析。','findings':[], 'recommended_configuration':{},'parameter_observations':[],'risks':['仅协议测试'],'next_experiments':[]}
            content=json.dumps(result,ensure_ascii=False)
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(json.dumps({'choices':[{'message':{'content':content}}]}).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1',seen
    server.shutdown();server.server_close()


def test_actual_http_service_analysis_and_report(admin,protocol_server):
    url,seen=protocol_server
    body={'name':'isolated protocol fixture','provider':'OpenAI Compatible','base_url':url,'api_key':'test-secret-not-real','models':['protocol-test-model'],'default_model':'protocol-test-model','timeout':10}
    service=admin.post('/api/model-services',json=body).json()
    assert service['has_api_key'] and 'test-secret-not-real' not in json.dumps(service)
    assert 'test-secret-not-real' not in json.dumps(db.get('model_services',service['id']))
    assert admin.post(f"/api/model-services/{service['id']}/test").json()['status']=='可用'
    agent={**db.get('settings','agent'),'service_id':service['id'],'model':'protocol-test-model'}
    assert admin.put('/api/agent',json=agent).status_code==200
    run=db.put('experiment_runs',{'kind':'E1','status':'completed','inputs':[],'config':{},'models':[],'result':{'speakers':[]},'reviews':[],'elapsed_ms':12})
    analysis=admin.post('/api/a01/analyze',json={'run_ids':[run['id']]})
    assert analysis.status_code==200,analysis.text
    assert 'embedding' not in seen[-1]['messages'][-1]['content']
    report=admin.post('/api/a01/report',json={'run_ids':[run['id']],'analysis_id':analysis.json()['id']})
    assert report.status_code==200,report.text
    ident=report.json()['id']
    for ext in ('md','csv','json'):
        response=admin.get(f'/api/reports/{ident}/{ext}')
        assert response.status_code==200 and len(response.content)>50
    evidence=admin.get(f'/api/reports/{ident}/json').json()
    assert evidence['selected_runs'][0]['id']==run['id']
    assert evidence['analysis']['analysis']['risks']==['仅协议测试']
    db.update('experiment_runs',run['id'],reviews=[{'item_id':'run','verdict':'正确'}])
    assert admin.post('/api/a01/report',json={'run_ids':[run['id']],'analysis_id':analysis.json()['id']}).status_code==400
    # Generated report is an immutable snapshot even after source review changes.
    assert admin.get(f'/api/reports/{ident}/json').json()['selected_runs'][0]['reviews']==[]
    db.update('settings','agent',service_id='',model='')


def test_service_key_preserved_and_clear(admin,protocol_server):
    url,_=protocol_server
    data={'name':'key fixture','base_url':url,'api_key':'fake-key','models':['protocol-test-model'],'default_model':'protocol-test-model'}
    s=admin.post('/api/model-services',json=data).json()
    for key,expected in [(None,True),('',False)]:
        response=admin.put(f"/api/model-services/{s['id']}",json={**data,'api_key':key})
        assert response.status_code==200 and response.json()['has_api_key'] is expected
