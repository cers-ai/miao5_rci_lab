"""Explicit additive v2 schema migration. Not called on application startup.

Stage 0 only rehearses this on a restored copy; existing rows are never rewritten.
"""
import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

VERSION = 2
NEW_TABLES = ('target_people', 'model_jobs', 'discovery_results', 'speaker_confirmations',
              'ground_truth_files', 'ground_truth_segments', 'experiment_candidates',
              'recognition_score_sets', 'recognition_segments')


def migrate(database):
    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError('Existing database required')
    with sqlite3.connect(database) as con:
        con.execute('BEGIN IMMEDIATE')
        old = con.execute('PRAGMA user_version').fetchone()[0]
        if old not in (0, VERSION):
            raise ValueError(f'Unsupported database version: {old}')
        if old == VERSION:
            names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not set(NEW_TABLES + ('schema_migrations',)) <= names:
                raise ValueError('Version marker exists but schema is incomplete')
            return {'version': VERSION, 'changed': False}
        existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if set(NEW_TABLES + ('schema_migrations',)) & existing:
            raise ValueError('Unversioned v2 tables already exist; manual inspection required')
        if not {'experiment_runs', 'target_profiles', 'experiment_files'} <= existing:
            raise ValueError('Not a supported legacy workspace')
        con.execute('CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL, description TEXT NOT NULL)')
        for table in NEW_TABLES:
            con.execute(f'CREATE TABLE {table} (id TEXT PRIMARY KEY NOT NULL, data TEXT NOT NULL CHECK(json_valid(data)))')
        for table, fields in {
            'model_jobs': ('run_id', 'status'), 'discovery_results': ('run_id',),
            'speaker_confirmations': ('run_id', 'discovery_result_id'),
            'ground_truth_files': ('target_person_id', 'audio_id'),
            'ground_truth_segments': ('ground_truth_file_id',),
            'experiment_candidates': ('run_id', 'status'),
            'recognition_score_sets': ('run_id', 'cache_key'),
            'recognition_segments': ('score_set_id', 'audio_id'),
        }.items():
            for field in fields:
                con.execute(f"CREATE INDEX idx_{table}_{field} ON {table}(json_extract(data, '$.{field}'))")
        con.execute('INSERT INTO schema_migrations VALUES (?,?,?)',
                    (VERSION, datetime.now(timezone.utc).isoformat(), 'A01 additive schema v2; legacy data preserved'))
        con.execute(f'PRAGMA user_version={VERSION}')
        return {'version': VERSION, 'changed': True, 'added_tables': list(NEW_TABLES)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate(args.database), indent=2))
