"""Backup and migration safety: no models or production workspace required."""
import json
import sqlite3

import pytest

from scripts import workspace_backup as wb
from scripts.a01_schema import migrate, NEW_TABLES


def legacy_workspace(root):
    root.mkdir()
    with sqlite3.connect(root / 'lab.sqlite3') as con:
        con.execute('PRAGMA journal_mode=WAL')
        for name in ('experiment_runs', 'target_profiles', 'experiment_files', 'analysis_reports'):
            con.execute(f'CREATE TABLE {name} (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
        con.execute('INSERT INTO experiment_runs VALUES (?,?)', ('legacy', '{"kind":"E1","status":"completed"}'))
    (root / 'uploads').mkdir()
    (root / 'uploads' / 'evidence').write_bytes(b'exact-source-evidence')
    (root / '.service-key').write_bytes(b'test-only-secret')
    return root


def test_backup_restore_preserves_wal_data_and_key(tmp_path):
    src = legacy_workspace(tmp_path / 'source')
    # Keep a connection open so committed WAL content has not been checkpointed.
    with sqlite3.connect(src / 'lab.sqlite3') as writer:
        writer.execute('INSERT INTO experiment_runs VALUES (?,?)', ('wal', '{"kind":"E2"}'))
        writer.commit()
        before = wb.database_summary(src / 'lab.sqlite3')
        wb.backup(src, tmp_path / 'backup')
    restored = tmp_path / 'restored'
    assert wb.restore(tmp_path / 'backup', restored)['verified']
    assert wb.database_summary(restored / 'lab.sqlite3') == before
    assert (restored / '.service-key').read_bytes() == b'test-only-secret'
    assert (restored / 'uploads/evidence').read_bytes() == b'exact-source-evidence'


def test_backup_rejects_tamper_and_overwrite(tmp_path):
    src = legacy_workspace(tmp_path / 'source')
    dest = tmp_path / 'backup'
    wb.backup(src, dest)
    with pytest.raises(ValueError, match='already exists'):
        wb.backup(src, dest)
    (dest / 'uploads/evidence').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='verification failed'):
        wb.restore(dest, tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()


def test_backup_detects_database_change_during_copy(tmp_path, monkeypatch):
    src = legacy_workspace(tmp_path / 'source')
    original_copy = wb.shutil.copy2
    def concurrent_write(source, destination):
        result = original_copy(source, destination)
        with sqlite3.connect(src / 'lab.sqlite3') as con:
            con.execute("UPDATE experiment_runs SET data='{}'")
        return result
    monkeypatch.setattr(wb.shutil, 'copy2', concurrent_write)
    dest = tmp_path / 'backup'
    with pytest.raises(ValueError, match='Database changed'):
        wb.backup(src, dest)
    assert not (dest / 'backup-manifest.json').exists()


def test_backup_rejects_unsafe_paths(tmp_path):
    src = legacy_workspace(tmp_path / 'source')
    with pytest.raises(ValueError, match='separate'):
        wb.backup(src, src / 'nested')
    dest = tmp_path / 'backup'
    wb.backup(src, dest)
    path = dest / 'backup-manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files']['../outside'] = {'bytes': 0, 'sha256': ''}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='member path'):
        wb.verify(dest)


def test_migration_idempotent_and_preserves_legacy_rows(tmp_path):
    src = legacy_workspace(tmp_path / 'source')
    database = src / 'lab.sqlite3'
    before = wb.database_summary(database)
    assert migrate(database)['changed']
    assert not migrate(database)['changed']
    after = wb.database_summary(database)
    assert all(after[k] == v for k, v in before.items())
    assert all(after[k]['count'] == 0 for k in NEW_TABLES)
    with sqlite3.connect(database) as con:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO model_jobs VALUES ('bad','not-json')")


def test_migration_rejects_future_or_incomplete_version(tmp_path):
    src = legacy_workspace(tmp_path / 'source')
    database = src / 'lab.sqlite3'
    with sqlite3.connect(database) as con:
        con.execute('PRAGMA user_version=99')
    with pytest.raises(ValueError, match='Unsupported'):
        migrate(database)
    with sqlite3.connect(database) as con:
        con.execute('PRAGMA user_version=2')
    with pytest.raises(ValueError, match='incomplete'):
        migrate(database)


def test_migration_does_not_accept_partial_unversioned_schema(tmp_path):
    src = legacy_workspace(tmp_path / 'source')
    database = src / 'lab.sqlite3'
    with sqlite3.connect(database) as con:
        con.execute('CREATE TABLE model_jobs(id TEXT, data TEXT)')
    before = wb.database_summary(database)
    with pytest.raises(ValueError, match='Unversioned'):
        migrate(database)
    assert wb.database_summary(database) == before
