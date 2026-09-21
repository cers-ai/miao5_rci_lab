"""Offline evidence backup using SQLite's online backup API; never imports app code.

Source reads are read-only. Copies include secrets and must remain local/ignored.
The application should be idle: detected database or evidence changes abort backup.
Restore accepts only a new directory and preserves the original database verbatim.
Relocated copies are for inspection, not serving (legacy paths remain absolute).
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


FOLDERS = ('uploads', 'models', 'target_profiles', 'experiment_outputs', 'reports', 'samples')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_db(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def database_summary(path):
    with read_db(path) as con:
        con.execute('BEGIN')
        integrity = con.execute('PRAGMA integrity_check').fetchall()
        if integrity != [('ok',)]:
            raise ValueError('SQLite integrity check failed')
        tables = {}
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            rows = con.execute('SELECT * FROM ' + quote(name)).fetchall()
            encoded = sorted(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
            tables[name] = {'count': len(rows), 'sha256': hashlib.sha256('\n'.join(encoded).encode()).hexdigest()}
        return tables


def evidence_files(root):
    root = Path(root).resolve()
    result = []
    for name in FOLDERS:
        folder = root / name
        if folder.exists():
            for p in folder.rglob('*'):
                if p.is_symlink() or not p.resolve().is_relative_to(root):
                    raise ValueError('Evidence links outside workspace are not supported')
                if p.is_file():
                    result.append(p)
    key = root / '.service-key'
    if key.exists():
        if key.is_symlink():
            raise ValueError('Service key must be a regular local file')
        result.append(key)
    return sorted(result)


def inventory(root):
    root = Path(root).resolve()
    return {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in evidence_files(root)}


def safe_member(root, relative):
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()) or p == root.resolve():
        raise ValueError('Invalid backup member path')
    return p


def new_destination(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Source and destination must be separate directory trees')
    if destination.exists():
        raise ValueError('Destination already exists; no overwrite is allowed')
    destination.mkdir(parents=True)
    return source, destination


def backup(source, destination):
    source, destination = new_destination(source, destination)
    before = database_summary(source / 'lab.sqlite3')
    files = inventory(source)
    with read_db(source / 'lab.sqlite3') as original, sqlite3.connect(destination / 'lab.sqlite3') as copied:
        original.backup(copied)
    for relative in files:
        target = safe_member(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    if before != database_summary(source / 'lab.sqlite3') or before != database_summary(destination / 'lab.sqlite3'):
        raise ValueError('Database changed during backup; incomplete backup must not be restored')
    if files != inventory(source) or files != inventory(destination):
        raise ValueError('Evidence changed during backup; incomplete backup must not be restored')
    files['lab.sqlite3'] = {'bytes': (destination / 'lab.sqlite3').stat().st_size,
                          'sha256': sha256(destination / 'lab.sqlite3')}
    manifest = {'format_version': 1, 'source_root': str(source),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'tables': before, 'files': files,
                'excluded': ['logs', 'SQLite WAL/SHM (merged using backup API)'],
                'restoration_note': 'Legacy absolute paths preserved. Do not serve a relocated copy.'}
    (destination / 'backup-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return verify(destination)


def verify(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'backup-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1 or 'lab.sqlite3' not in manifest.get('files', {}):
        raise ValueError('Unsupported or incomplete backup manifest')
    for relative, expected in manifest['files'].items():
        p = safe_member(folder, relative)
        if not p.is_file() or p.stat().st_size != expected['bytes'] or sha256(p) != expected['sha256']:
            raise ValueError(f'Backup verification failed: {relative}')
    tables = database_summary(folder / 'lab.sqlite3')
    if tables != manifest['tables']:
        raise ValueError('Backup table mismatch')
    return {'verified': True, 'file_count': len(manifest['files']), 'tables': tables,
            'total_bytes': sum(x['bytes'] for x in manifest['files'].values())}


def restore(source, destination):
    verify(source)
    source, destination = new_destination(source, destination)
    manifest = json.loads((source / 'backup-manifest.json').read_text(encoding='utf-8'))
    for relative in manifest['files']:
        target = safe_member(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe_member(source, relative), target)
    shutil.copy2(source / 'backup-manifest.json', destination / 'backup-manifest.json')
    return verify(destination)


def audit_legacy_references(folder, original_root):
    """Resolve legacy absolute references against the copied root without DB edits."""
    folder, original_root = Path(folder).resolve(), Path(original_root).resolve()
    missing, external = [], []
    checked = 0
    def check(value):
        nonlocal checked
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ('path', 'original_path', 'normalized_path', 'embedding_path') and isinstance(v, str) and v:
                    p = Path(v)
                    if p.is_absolute():
                        try:
                            relative = p.resolve().relative_to(original_root)
                        except ValueError:
                            external.append(k)
                        else:
                            checked += 1
                            if not (folder / relative).exists():
                                missing.append(relative.as_posix())
                else:
                    check(v)
        elif isinstance(value, list):
            for v in value:
                check(v)
    with read_db(folder / 'lab.sqlite3') as con:
        for table in ('experiment_files', 'target_profiles', 'experiment_runs', 'analysis_reports'):
            for (data,) in con.execute('SELECT data FROM ' + quote(table)):
                check(json.loads(data))
        for (ident,) in con.execute('SELECT id FROM target_profiles'):
            checked += 1
            if not (folder / 'target_profiles' / f'{ident}.npy').is_file():
                missing.append(f'target_profiles/{ident}.npy')
    return {'checked_paths': checked, 'missing': sorted(set(missing)), 'external_path_fields': external}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['backup', 'verify', 'restore', 'audit'])
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, nargs='?')
    args = parser.parse_args()
    if args.action in ('backup', 'restore') and args.destination is None:
        parser.error('destination is required')
    if args.action == 'verify':
        result = verify(args.source)
    elif args.action == 'audit':
        manifest = json.loads((args.source / 'backup-manifest.json').read_text(encoding='utf-8'))
        result = audit_legacy_references(args.source, manifest['source_root'])
    else:
        result = {'backup': backup, 'restore': restore}[args.action](args.source, args.destination)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
