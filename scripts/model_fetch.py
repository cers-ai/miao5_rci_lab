"""Standard-library downloader with upstream hash validation and atomic files.

Can run before Python ML dependencies finish installing.
"""
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'campp': ('iic/speech_campplus_sv_zh-cn_16k-common','v2.0.2'),
    'vad': ('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch','v2.0.4'),
    'paraformer': ('iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online','v2.0.4'),
}


def sha(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def get_json(url):
    with urllib.request.urlopen(url,timeout=60) as response:
        return json.load(response)


def download(url,path,size,checksum=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==size and (not checksum or sha(path)==checksum):
        return
    for attempt in range(3):
        try:
            tmp=path.with_name(path.name+'.part')
            with urllib.request.urlopen(url,timeout=180) as response,tmp.open('wb') as out:
                shutil.copyfileobj(response,out,1024*1024)
            if tmp.stat().st_size!=size or (checksum and sha(tmp)!=checksum):
                raise ValueError(f'模型文件校验失败: {path.name}')
            tmp.replace(path)
            print(f'Downloaded {path.name}: {size/1024/1024:.1f} MB',flush=True)
            return
        except Exception:
            if attempt==2: raise
            time.sleep(2)


def fetch(ident,folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if ident in SOURCES:
        repo,revision=SOURCES[ident]
        endpoint=f'https://modelscope.cn/api/v1/models/{repo}/repo'
        listing=get_json(endpoint+'/files?'+urllib.parse.urlencode({'Revision':revision,'Recursive':'true'}))
        if not listing.get('Success'): raise ValueError('ModelScope 文件清单获取失败')
        files=[f for f in listing['Data']['Files'] if f['Type']=='blob' and not f['Path'].startswith('.')]
        def one(f):
            path=(folder/f['Path']).resolve()
            if not path.is_relative_to(folder.resolve()): raise ValueError('模型清单路径不合法')
            download(endpoint+'?'+urllib.parse.urlencode({'Revision':revision,'FilePath':f['Path']}),path,f['Size'],f.get('Sha256'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one,files))
        upstream={f['Path']:{'sha256':f['Sha256'],'bytes':f['Size']} for f in files}
    elif ident=='wespeaker':
        repo='wenet/wespeaker_pretrained_models:cnceleb_resnet34_LM.onnx'
        listing=get_json('https://modelscope.cn/api/v1/datasets/wenet/wespeaker_pretrained_models/oss/tree')
        f=next(f for f in listing['Data'] if f['Key']=='cnceleb_resnet34_LM.onnx')
        download(f['Url'],folder/f['Key'],f['Size'])
        revision=f"official-oss-{f['LastModified']}"
        upstream={f['Key']:{'sha256':sha(folder/f['Key']),'bytes':f['Size']}}
    else: raise ValueError('不支持的公开模型')
    previous=folder/'manifest.json'
    if previous.exists():
        old=json.loads(previous.read_text(encoding='utf-8'))
        if old.get('revision')==revision and old.get('files')!=upstream:
            raise ValueError('同一模型版本的文件哈希发生变化，拒绝静默更新')
    manifest={'source':repo,'revision':revision,'files':upstream,'downloaded_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    previous.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(fetch,k,ROOT/'workspace/models'/k):k for k in ('campp','vad','wespeaker','paraformer')}
        for job in concurrent.futures.as_completed(jobs):
            print(jobs[job],job.result()['revision'],flush=True)
    sample_dir=ROOT/'workspace/samples'
    sample_dir.mkdir(parents=True,exist_ok=True)
    for name in ('speaker1_a_cn_16k.wav','speaker1_b_cn_16k.wav','speaker2_a_cn_16k.wav'):
        shutil.copyfile(ROOT/'workspace/models/campp/examples'/name,sample_dir/name)
