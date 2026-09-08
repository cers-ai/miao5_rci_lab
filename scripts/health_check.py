import json
import urllib.request
for url in ('http://127.0.0.1:8000/api/health','http://localhost:3000','http://localhost:3000/api/health'):
    with urllib.request.urlopen(url,timeout=10) as r:
        print(json.dumps({'url':url,'status':r.status,'content_type':r.headers.get('content-type')},ensure_ascii=False))
