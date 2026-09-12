#!/usr/bin/env python3
"""Demonstrate explicit edits, durable state, and exactly-once retry semantics."""
from pathlib import Path
import tempfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from energy_transformer.bridge import EnergyTextSession
from energy_transformer.ledger import EnergyLedger
from energy_transformer.review import Edit


def main():
    engine=EnergyTextSession('checkpoints/v14/bridge_seed11.pt')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'memory.db'
        ledger=EnergyLedger(path,engine.signature)
        # The caller explicitly reviews and authorizes the structured edit.
        edit=Edit('set','alice','garden')
        kwargs=dict(expected_revision=0,request_id='demo-explicit-edit',authorized=True)
        first=ledger.commit([edit],**kwargs)
        retry=ledger.commit([edit],**kwargs)
        reopened=EnergyLedger(path,engine.signature)
        state,revision=reopened.read()
        print('Answer after reopening:',engine.decoder.text(state.feature(0)[None])[0])
        print('Committed revision:',first['revision'],'Retry revision:',retry['revision'])
        print('Audit:',reopened.audit())
        print('No parser-proposed edit was automatically authorized.')

if __name__=='__main__': main()
