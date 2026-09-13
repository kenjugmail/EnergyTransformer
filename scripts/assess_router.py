#!/usr/bin/env python3
"""Keep specialized-task progress separate from generic-pretraining readiness."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    r=ROOT/'results/v20';d=pd.read_csv(r/'summary.csv');a=d.query("family=='hard_aux' and intervention=='none'").set_index('condition')
    checks={'specialist_primary_95pct':float(a.loc['primary','accuracy'])>=.95,'specialist_long_85pct':float(a.loc['long','accuracy'])>=.85,
       'generic_backbone_recall_gate':False,'layout_shift_85pct':float(a.loc['layout_shift','accuracy'])>=.85,'fresh_natural_language_evaluation':False,'target_gpu_acceptance':False,'resource_matched_modern_backbone':False}
    result={'decision':'NO_GO_FOR_SERIOUS_PRETRAINING','next_stage':'bounded language-interface integration experiment; specialist synthetic gate passed',
       'checks':checks,'interpretation':'A 32-address categorical memory with training-only local semantic labels is not the unchanged general v19 backbone. Original thresholds were not lowered; specialized success does not clear generic gates.',
       'observations':a.accuracy.to_dict(),'automatic_spend_or_jobs_authorized':False}
    (r/'SCALING_DECISION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
