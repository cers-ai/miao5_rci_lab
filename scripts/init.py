import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='初始化妙悟实验室并下载/测试预设模型')
    p.add_argument('--skip-models',action='store_true')
    args = p.parse_args()
    if sys.version_info[:2] != (3,11) or sys.prefix == sys.base_prefix:
        if not shutil.which('uv'):
            raise SystemExit('请安装 uv 并运行 uv sync --locked --python 3.11')
        subprocess.run(['uv','sync','--locked','--python','3.11'],cwd=ROOT,check=True)
        python = ROOT/'.venv'/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
        raise SystemExit(subprocess.call([str(python),__file__,*sys.argv[1:]],cwd=ROOT))
    import torch
    from backend.app import store as db
    print(json.dumps({'python':platform.python_version(),'ffmpeg':shutil.which('ffmpeg'),'ffprobe':shutil.which('ffprobe'),
        'cuda':torch.cuda.is_available(),'disk_free_gb':round(shutil.disk_usage(ROOT).free/2**30,1),'workspace':str(db.WORK)},ensure_ascii=False,indent=2))
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise SystemExit('请安装 ffmpeg（包含 ffprobe）并加入 PATH')
    if not args.skip_models:
        from scripts.download_models import install,samples
        samples()
        for ident in ('campp','vad','wespeaker','paraformer','pyannote'):
            install(ident)
    print('初始化完成；默认用户 admin，初始密码 abcd@1234。模型状态以真实测试结果为准。')
