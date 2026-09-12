"""Reviewed, revision-checked memory edits; no new learned language capability.

The v14 parser proposes an edit and NEVER executes it here. A caller must review
its exact structured meaning and separately authorize approval. This is not an
authentication layer: a hosting application must authenticate/authorize callers.
Pending proposals are bounded, process-local, and deliberately not restored.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import secrets
import threading
import torch
from .bridge import ASK, DELETE, HOLD, SET, NAMES, VALUES, OPERATIONS
from .bridge import EnergyTextSession, ProtectedState, lexical_consistent


@dataclass(frozen=True)
class Edit:
    operation: str
    entity: str
    value: str | None = None

    def validate(self) -> None:
        if self.operation not in ('set', 'delete') or self.entity not in NAMES:
            raise ValueError('Unsupported operation or entity')
        if self.operation == 'set' and self.value not in VALUES:
            raise ValueError('Unsupported value')
        if self.operation == 'delete' and self.value is not None:
            raise ValueError('Deletion must not include a value')

    def apply(self, state: ProtectedState) -> ProtectedState:
        self.validate()
        return state.edit(SET if self.operation == 'set' else DELETE,
                          NAMES.index(self.entity),
                          VALUES.index(self.value) if self.value is not None else 0)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    revision: int
    state_sha256: str
    text_sha256: str
    edit: Edit
    confidence: float


def state_digest(state: ProtectedState) -> str:
    """Integrity/binding hash, NOT a signature or proof of truth."""
    h = hashlib.sha256(b'energy-reviewed-state-v15\0')
    for tensor in (state.precision, state.information):
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ReviewedEnergySession:
    def __init__(self, checkpoint: str | Path, *, max_pending: int = 128):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('max_pending must be a positive integer')
        self._engine = EnergyTextSession(checkpoint)
        self._revision = 0
        self._pending: dict[str, Proposal] = {}
        self._max_pending = max_pending
        self._lock = threading.RLock()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def state_payload(self) -> dict:
        """Return clones; callers cannot mutate the live memory through this API."""
        with self._lock:
            return self._engine.state.payload()

    @torch.inference_mode()
    def propose(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        with self._lock:
            try:
                tokens = self._engine.codec.encode([text])
            except ValueError as exc:
                return {'status': 'abstained', 'reason': str(exc)}
            outputs = self._engine.parser(tokens)
            op, key, value = [int(x.argmax(-1)) for x in outputs]
            confidence = [float(x.softmax(-1).amax()) for x in outputs]
            relevant = [confidence[0]]
            if op != HOLD:
                relevant.append(confidence[1])
            if op == SET:
                relevant.append(confidence[2])
            score = min(relevant)
            if score < self._engine.threshold or not lexical_consistent(text, op, key, value):
                return {'status': 'abstained', 'confidence': score}
            if op == HOLD:
                return {'status': 'unchanged', 'revision': self._revision}
            if op == ASK:
                return {'status': 'answered', 'entity': NAMES[key],
                        'answer': self.answer(NAMES[key]), 'revision': self._revision}
            if len(self._pending) >= self._max_pending:
                return {'status': 'abstained', 'reason': 'pending_limit'}
            edit = Edit(OPERATIONS[op], NAMES[key], VALUES[value] if op == SET else None)
            proposal = Proposal(secrets.token_hex(16), self._revision,
                                state_digest(self._engine.state),
                                hashlib.sha256(text.encode('utf-8')).hexdigest(), edit, score)
            self._pending[proposal.proposal_id] = proposal
            return {'status': 'needs_review', 'proposal': asdict(proposal)}

    @torch.inference_mode()
    def answer(self, entity: str) -> str:
        """Explicit entity query; requires no linguistic inference about the entity."""
        if entity not in NAMES:
            raise ValueError('Unsupported entity')
        with self._lock:
            return self._engine.decoder.text(
                self._engine.state.feature(NAMES.index(entity))[None])[0]

    def approve(self, proposal_id: str, *, reviewed_edit: Edit,
                expected_revision: int, authorized: bool = False) -> int:
        """Approve only the exact reviewed meaning against the unchanged base state."""
        if authorized is not True:
            raise PermissionError('Explicit authorization is required')
        reviewed_edit.validate()
        with self._lock:
            proposal = self._pending.get(proposal_id)
            if proposal is None:
                raise ValueError('Unknown, expired, or already applied proposal')
            if type(expected_revision) is not int or expected_revision != self._revision:
                raise ValueError('Stale revision')
            if proposal.revision != self._revision or proposal.state_sha256 != state_digest(self._engine.state):
                raise ValueError('Proposal base state changed')
            if proposal.edit != reviewed_edit:
                raise ValueError('Reviewed meaning differs from proposed meaning')
            self._engine.state = reviewed_edit.apply(self._engine.state)
            self._revision += 1
            self._pending.clear()  # every other proposal referred to the old state
            return self._revision

    def apply_edits(self, edits: list[Edit], *, expected_revision: int,
                    authorized: bool = False) -> int:
        """Atomic explicitly supplied edits, independent of the parser's output."""
        if authorized is not True:
            raise PermissionError('Explicit authorization is required')
        if not isinstance(edits, list) or not edits:
            raise ValueError('A nonempty list of edits is required')
        with self._lock:
            if type(expected_revision) is not int or expected_revision != self._revision:
                raise ValueError('Stale revision')
            candidate = self._engine.state
            for edit in edits:
                if not isinstance(edit, Edit):
                    raise TypeError('Expected Edit values')
                candidate = edit.apply(candidate)
            self._engine.state = candidate
            self._revision += 1
            self._pending.clear()
            return self._revision

    def discard(self, proposal_id: str) -> bool:
        with self._lock:
            return self._pending.pop(proposal_id, None) is not None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with self._lock:
            data = {'format': 'reviewed_energy_v15', 'model_sha256': self._engine.signature,
                    'revision': self._revision, 'state': self._engine.state.payload(),
                    'state_sha256': state_digest(self._engine.state)}
            tmp = path.with_name(path.name + '.' + secrets.token_hex(8) + '.tmp')
            try:
                torch.save(data, tmp)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)

    def restore(self, path: str | Path) -> None:
        data = torch.load(path, map_location='cpu', weights_only=True)
        required = {'format', 'model_sha256', 'revision', 'state', 'state_sha256'}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError('Invalid snapshot schema')
        if data['format'] != 'reviewed_energy_v15' or data['model_sha256'] != self._engine.signature:
            raise ValueError('Incompatible model snapshot')
        if type(data['revision']) is not int or data['revision'] < 0:
            raise ValueError('Invalid revision')
        state = ProtectedState.restore(data['state'])
        if state_digest(state) != data['state_sha256']:
            raise ValueError('Snapshot integrity check failed')
        with self._lock:
            self._engine.state = state
            self._revision = data['revision']
            self._pending.clear()
