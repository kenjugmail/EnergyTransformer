"""Transactional protected energy memory for explicitly reviewed edits.

Local SQLite persistence; not a distributed database or authentication service.
The neural parser is deliberately NOT allowed to call commit on its own.
Audit hashes detect inconsistency, not a hostile rewrite or a whole-file rollback.
Deletion clears live memory but does not erase the append-only edit history.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import struct
from typing import Iterator
import torch
from .bridge import ProtectedState
from .review import Edit

FORMAT = 'protected_energy_ledger_v16'
ZERO = '0' * 64


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def pack(state: ProtectedState) -> bytes:
    state = ProtectedState.restore(state.payload())
    return struct.pack('<272f', *(state.precision.tolist() + state.information.flatten().tolist()))


def unpack(data: bytes) -> ProtectedState:
    if len(data) != 1088:
        raise ValueError('Invalid coefficient payload length')
    values = torch.tensor(struct.unpack('<272f', data), dtype=torch.float32)
    return ProtectedState.restore({'precision': values[:16], 'information': values[16:].reshape(16, 16)})


class EnergyLedger:
    """Atomic compare-and-swap commits shared by processes on one local filesystem.

    request_id makes an exact retry idempotent. Reusing it for another request fails.
    All writers must use this interface. A Boolean authorization flag is not identity.
    """
    def __init__(self, path: str | Path, model_sha256: str):
        if not re.fullmatch(r'[0-9a-f]{64}', model_sha256):
            raise ValueError('Expected a SHA-256 model fingerprint')
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError('Database path must not be a symlink')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model_sha256
        with self._connection(write=True) as db:
            db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), format TEXT, model TEXT, revision INTEGER, payload BLOB, sha TEXT, head TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS events (revision INTEGER PRIMARY KEY, request_id TEXT UNIQUE, request_sha TEXT, event TEXT, hash TEXT)')
            payload = pack(ProtectedState.initial())
            db.execute('INSERT OR IGNORE INTO state VALUES (1,?,?,?,?,?,?)',
                       (FORMAT, self.model, 0, payload, digest(payload), ZERO))
            self._load(db)

    @contextmanager
    def _connection(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _load(self, db):
        row = db.execute('SELECT * FROM state WHERE id=1').fetchone()
        if row is None or row['format'] != FORMAT or row['model'] != self.model:
            raise ValueError('Database schema/model mismatch')
        if row['sha'] != digest(row['payload']) or row['revision'] < 0:
            raise ValueError('Invalid state integrity or revision')
        unpack(row['payload'])
        return row

    def read(self) -> tuple[ProtectedState, int]:
        with self._connection() as db:
            row = self._load(db)
            return unpack(row['payload']), row['revision']

    def commit(self, edits: list[Edit], *, expected_revision: int, request_id: str,
               authorized: bool = False, source_sha256: str | None = None) -> dict:
        if authorized is not True:
            raise PermissionError('Explicit external authorization is required')
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError('Invalid expected revision')
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', request_id):
            raise ValueError('Invalid request identifier')
        if source_sha256 is not None and not re.fullmatch(r'[0-9a-f]{64}', source_sha256):
            raise ValueError('Invalid source fingerprint')
        if not isinstance(edits, list) or not 1 <= len(edits) <= 64:
            raise ValueError('Provide 1 to 64 explicit edits')
        items = []
        for edit in edits:
            if not isinstance(edit, Edit):
                raise TypeError('Expected Edit instances')
            edit.validate()
            items.append({'operation': edit.operation, 'entity': edit.entity, 'value': edit.value})
        request = {'expected_revision': expected_revision, 'edits': items, 'source_sha256': source_sha256}
        request_sha = digest(canonical(request).encode())
        with self._connection(write=True) as db:
            row = self._load(db)
            previous = db.execute('SELECT * FROM events WHERE request_id=?', (request_id,)).fetchone()
            if previous is not None:
                if previous['request_sha'] != request_sha:
                    raise ValueError('Request identifier was already used for different content')
                return {'revision': previous['revision'], 'event_sha256': previous['hash'], 'replayed': True}
            if expected_revision != row['revision']:
                raise ValueError('Stale revision; reload and review again')
            candidate = unpack(row['payload'])
            for item in items:
                candidate = Edit(**item).apply(candidate)
            payload = pack(candidate)
            revision = row['revision'] + 1
            event = {'revision': revision, 'request_id': request_id, 'request_sha256': request_sha,
                     'request': request, 'previous_hash': row['head'],
                     'before_sha256': row['sha'], 'after_sha256': digest(payload)}
            encoded = canonical(event)
            event_sha = digest(encoded.encode())
            db.execute('INSERT INTO events VALUES (?,?,?,?,?)',
                       (revision, request_id, request_sha, encoded, event_sha))
            db.execute('UPDATE state SET revision=?,payload=?,sha=?,head=? WHERE id=1',
                       (revision, payload, digest(payload), event_sha))
            return {'revision': revision, 'event_sha256': event_sha, 'replayed': False}

    def audit(self) -> dict:
        """Replay the complete edit journal and verify the resulting live state."""
        with self._connection() as db:
            row = self._load(db)
            state, head, revision = ProtectedState.initial(), ZERO, 0
            for entry in db.execute('SELECT * FROM events ORDER BY revision'):
                event = json.loads(entry['event'])
                request = event['request']
                if (entry['revision'] != revision + 1 or event['revision'] != revision + 1
                    or event['request_id'] != entry['request_id']
                    or request['expected_revision'] != revision
                    or event['previous_hash'] != head
                    or event['before_sha256'] != digest(pack(state))
                    or digest(canonical(request).encode()) != entry['request_sha']
                    or event['request_sha256'] != entry['request_sha']
                    or digest(entry['event'].encode()) != entry['hash']):
                    raise ValueError('Broken audit chain')
                for item in request['edits']:
                    state = Edit(**item).apply(state)
                if event['after_sha256'] != digest(pack(state)):
                    raise ValueError('Edit replay mismatch')
                revision += 1
                head = entry['hash']
            if revision != row['revision'] or head != row['head'] or pack(state) != row['payload']:
                raise ValueError('Live state and journal disagree')
            return {'revision': revision, 'state_sha256': row['sha'], 'events': revision, 'valid': True}
