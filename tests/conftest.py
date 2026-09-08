import os
import tempfile
from pathlib import Path
import pytest

os.environ['MIAOWU_WORKSPACE'] = tempfile.mkdtemp(prefix='miaowu-tests-')

@pytest.fixture(scope='session')
def client():
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as client:
        yield client

@pytest.fixture
def admin(client):
    response = client.post('/api/login',json={'username':'admin','password':'abcd@1234'})
    assert response.status_code == 200,response.text
    return client

@pytest.fixture
def audio_bytes():
    import io
    import soundfile as sf
    import numpy as np
    buf = io.BytesIO()
    sf.write(buf,np.zeros(16000,dtype=np.float32),16000,format='WAV')
    return buf.getvalue()

@pytest.fixture(scope='session')
def real_models():
    from backend.app import store as db
    root = Path(__file__).resolve().parents[1]
    for ident in ('campp','vad','wespeaker','paraformer'):
        folder = root/'workspace/models'/ident
        manifest = folder/'manifest.json'
        if not manifest.exists():
            pytest.skip(f'真实模型尚未下载: {ident}')
        import json
        info=json.loads(manifest.read_text(encoding='utf-8'))
        db.update('local_models',ident,path=str(folder),status='未测试',version=info['revision'],manifest=info)
    return root/'workspace/samples'
