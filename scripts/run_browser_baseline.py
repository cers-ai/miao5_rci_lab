"""Run existing browser tests on a new workspace and owned hidden services.

Ports 3000/8000 must be free; never attaches to or stops an existing user service.
Models/samples are read-only inputs from workspace; all new records are isolated.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HIDDEN = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a new evidence directory')
    for port in (3000, 8000):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    output.mkdir(parents=True)
    work = output / 'workspace'
    work.mkdir()
    shutil.copytree(ROOT / 'workspace/samples', work / 'samples')
    env = {**os.environ, 'MIAOWU_WORKSPACE': str(work),
           'MIAOWU_BROWSER_EVIDENCE_DIR': str(output), 'PYTHONUTF8': '1'}
    prepare = '''
import json
from pathlib import Path
from backend.app import store as db
root = Path.cwd()
for ident in ('campp', 'wespeaker', 'vad', 'paraformer'):
    folder = root / 'workspace/models' / ident
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    db.update('local_models', ident, path=str(folder), manifest=manifest,
              version=manifest['revision'], status='未测试')
'''
    subprocess.run([sys.executable, '-c', prepare], cwd=ROOT, env=env, check=True, creationflags=HIDDEN)
    node = shutil.which('node')
    if not node:
        raise ValueError('Node.js required')
    children, logs = [], []
    try:
        commands = [
            [sys.executable, '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '8000'],
            [node, str(ROOT/'frontend/node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', '3000', '--strictPort'],
        ]
        for i, command in enumerate(commands):
            stream = (output / f'service-{i}.log').open('wb')
            logs.append(stream)
            children.append(subprocess.Popen(command, cwd=ROOT if i == 0 else ROOT/'frontend', env=env,
                stdout=stream, stderr=subprocess.STDOUT, creationflags=HIDDEN))
        (output/'owned-processes.json').write_text(json.dumps({'pids': [p.pid for p in children], 'workspace': str(work)}, indent=2))
        # Wait for actual model self-tests, not merely listening TCP sockets.
        from scripts.workspace_backup import read_db
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in children):
                raise RuntimeError('A baseline service exited; inspect service logs')
            try:
                with urllib.request.urlopen('http://127.0.0.1:3000/api/health', timeout=2) as response:
                    assert response.status == 200
                with read_db(work/'lab.sqlite3') as con:
                    models = [json.loads(r[0]) for r in con.execute('SELECT data FROM local_models')]
                relevant = [m for m in models if m['id'] in ('campp','wespeaker','vad','paraformer')]
                if any(m['status'] == '测试失败' for m in relevant):
                    raise RuntimeError('Real model startup self-test failed')
                if len(relevant) == 4 and all(m['status'] == '可用' for m in relevant):
                    break
            except (OSError, AssertionError):
                pass
            time.sleep(.25)
        else:
            raise TimeoutError('Baseline startup timed out')
        print('Isolated services and real models ready; starting Playwright.', flush=True)
        result = subprocess.run([node, str(ROOT/'frontend/node_modules/@playwright/test/cli.js'),
            'test', '--config=playwright.baseline.config.ts'], cwd=ROOT/'frontend', env=env, creationflags=HIDDEN)
        return result.returncode
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        for stream in logs:
            stream.close()
        (output/'cleanup.json').write_text(json.dumps({'services_exited': all(p.poll() is not None for p in children)}, indent=2))


if __name__ == '__main__':
    # Direct script execution must also resolve scripts.* without importing the app.
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
