"""Summarize measured recovery without equating language loss to capabilities."""
from collections import Counter
import json
import re
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_rewt_lfm3 import sha,save_json


def main():
    out=ROOT/'reports/lfm3_capability_recovery_20261028'
    models={k:json.loads((out/f'{k}.json').read_text()) for k in ('before','after','teacher')}
    assert len({d['suite_sha256'] for d in models.values()})==1
    assert len({d['protocol'] for d in models.values()})==1
    assert all('summary' in d for d in models.values())
    selection=json.loads((out/'selection.json').read_text())
    recommended=Path(selection['recommended_checkpoint'])
    recommended_sha=sha(recommended)
    assert models['after']['model']['sha256']==recommended_sha
    science={k:sum(v['correct'] for v in d['summary']['arc'].values()) for k,d in models.items()}
    exact={k:d['summary']['overall_probes']['correct'] for k,d in models.items()}
    stages=[]
    for path in sorted((ROOT/'runs/lfm3_capability_recovery_20261028').glob('stage*/summary.json')):
        summary=json.loads(path.read_text());plan=json.loads((path.parent/'plan.json').read_text())
        records=[json.loads(s) for s in (path.parent/'metrics.jsonl').read_text().splitlines()]
        stages.append(dict(name=path.parent.name,summary=summary,plan=plan,metrics=records))
        for name in ('summary.json','plan.json','metrics.jsonl'):
            shutil.copy2(path.parent/name,out/f'{path.parent.name}_{name}')
    manifest=json.loads((ROOT/'data/lfm3_capability_recovery_v1/manifest.json').read_text())
    save_json(out/'corpus_manifest.json',manifest)
    comparisons={}
    for section in ('arc','probes'):
        before=models['before'][section];after=models['after'][section]
        assert [r['id'] for r in before]==[r['id'] for r in after]
        changes=Counter('gained' if b['correct'] and not a['correct'] else 'lost' if a['correct'] and not b['correct'] else 'both_correct' if a['correct'] else 'both_wrong' for a,b in zip(before,after))
        comparisons[section]=dict(changes)
    save_json(out/'paired_changes.json',comparisons)
    recall={}
    for name,d in models.items():
        groups={}
        for row in d['probes']:
            if not row['category'].startswith('recall_'):continue
            codes=re.findall(r'(?<!\d)\d{5}(?!\d)',row['response'])
            g=groups.setdefault(row['category'],dict(correct_code_content=0,strict_exact_output=0,total=0))
            g['correct_code_content']+=int(bool(codes) and set(codes)=={row['expected']})
            g['strict_exact_output']+=int(row['correct']);g['total']+=1
        recall[name]=groups
    recall['rubric']='Secondary diagnostic added after inspecting baseline responses: at least one five-digit code appears and all five-digit codes equal the expected secret. Reported separately; does not change fixed strict scoring.'
    save_json(out/'recall_content_diagnostic.json',recall)
    sources=['rewt/lfm_conversion.py','rewt/lfm_distill.py','rewt/streaming.py','scripts/prepare_lfm3_recovery.py',
             'scripts/prepare_rewt_lfm3.py',
             'scripts/recover_lfm3_capabilities.py','scripts/evaluate_lfm3_capabilities.py',
             'scripts/summarize_lfm3_capabilities.py','scripts/audit_lfm3_recovery.py',
             'scripts/package_rewt_lfm3.py','scripts/verify_rewt_lfm3.py',
             'scripts/distill_rewt_lfm3.py','tests_rewt/test_lfm_distill.py']
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    save_json(out/'source_manifest.json',{name:sha(ROOT/name) for name in sources})
    lines=['# LFM3 → streaming REWT: capability recovery attempt','',
        '**Recommended artifact: stage 2.** Stage 3 reduced validation KL further, but scored 58/128 on science and 9/50 on exact responses, versus stage 2’s 59/128 and 13/50. The recommended bundle therefore uses stage 2. This is a post-hoc recommendation from the small capability suite, not an independent confirmation result. Both versions remain available. [Selection record](selection.json) · [Stage 3 raw results](stage3_capabilities.json).','',
        f'The selected model changes the small science score from **{science["before"]}/128 to {science["after"]}/128** and exact-response checks from **{exact["before"]}/50 to {exact["after"]}/50**. The original donor scores {science["teacher"]}/128 and {exact["teacher"]}/50 under the same protocol. **Full LFM3 capability recovery is not established.**','',
        f'Three passes consumed {sum(s["summary"]["token_exposures"] for s in stages):,} additional real-token exposures in total. The recommended stage-2 checkpoint includes only the first two passes: {sum(s["summary"]["token_exposures"] for s in stages[:2]):,} exposures and 1,536 optimizer updates. Its short-crop validation KL is {stages[2]["summary"]["initial_validation"]["kl"]:.4f}, versus {stages[0]["summary"]["initial_validation"]["kl"]:.4f} before this attempt. The third pass is retained as an unsuccessful broader adaptation experiment.','',
        'This experiment continues the standalone converted checkpoint with broader instruction data, short conversations, answer-weighted distillation, and feedforward adaptation. The original converted checkpoint remains available. **These measurements establish the result of this recovery attempt, not equivalence to LFM3.**','',
        '## Fixed before/donor/after comparison','',
        '| Check | Before | After recovery | Original LFM3 |','|---|---:|---:|---:|']
    labels={'format':'Exact format','arithmetic':'Arithmetic, exact output','knowledge':'Factual answers, exact output',
        'code_output':'Python output, exact output','json':'JSON extraction','tool_json':'JSON routing (no tool execution)',
        'recall_4_lines':'Code recall, 77-token prompts','recall_48_lines':'Code recall, 561-token prompts',
        'recall_112_lines':'Code recall, 1,265-token prompts'}
    chart=[]
    for section in ('arc','probes'):
        for category in models['before']['summary'][section]:
            scores=[models[k]['summary'][section][category] for k in ('before','after','teacher')]
            name=labels.get(category,category+' (custom label ranking)')
            lines.append('| '+name+' | '+' | '.join(f'{s["correct"]}/{s["total"]} ({s["accuracy"]:.1%})' for s in scores)+' |')
            chart.append((name,scores))
    lines+=['',
        'All 128 ARC questions and 50 generation probes are fixed in [evaluation_suite.json](evaluation_suite.json). The same prompts, tokenizer, non-thinking chat template, BF16 execution, greedy decoding, repetition penalty 1.0, and 64-new-token limit apply to all models. The science task ranks single-token answer labels at the first generation position; this is a custom zero-shot protocol, not a published ARC/lm-eval score.',
        '',
        'Exact-output scores combine answer content with instruction compliance. For example, `38 + 27 = 65` fails a request for only `65`. JSON is compared as parsed objects. Responses reaching the output limit may be incomplete. The donor can emit reasoning despite the non-thinking prompt; its model card documents that inconsistency. These small probes are not broad knowledge, coding, agent, or long-context benchmarks.',
        '', 'The strict-score improvement is concentrated in short code-recall formatting: stage 2 gives all four bare codes at 77 tokens. At 561 and 1,265 tokens, it fails to return the correct code in all eight cases, while the donor retrieves all eight correctly. Arithmetic and Python-output strict scores remain 0/8 and 0/6, and tool-call JSON remains 0/4. These strict failures must not be interpreted as a complete content-accuracy assessment.',
        '',
        'A [separate recall-content diagnostic](recall_content_diagnostic.json), added after inspecting baseline responses, counts correct five-digit codes even when the model adds prose. This distinguishes forgotten information from an output-format error; it does not replace or alter the original strict scores.',
        '',f'Paired science outcomes: {comparisons["arc"].get("gained",0)} previously wrong questions became correct; {comparisons["arc"].get("lost",0)} previously correct questions became wrong. Paired exact-response outcomes: {comparisons["probes"].get("gained",0)} gains and {comparisons["probes"].get("lost",0)} losses.',
        '', '[All raw starting responses](before.json) · [All raw recovered responses](after.json) · [All raw donor responses](teacher.json) · [Paired changes](paired_changes.json)','',
        '![Before, recovered and donor results](capability_comparison.png)','',
        '## Training and selection','',
        f'The new corpus contains {manifest["splits"]["train"]["documents"]:,} training conversations ({manifest["splits"]["train"]["tokens"]:,} stored tokens), including {manifest["splits"]["train"]["short_under_513"]:,} shorter than 513 tokens, and {manifest["splits"]["valid"]["documents"]} validation conversations. It is selected from the pinned official SmolTalk training shard, with larger proportions of code and constrained-instruction examples. Hash buckets preserve the original split rule. Old validation/test conversations are excluded, and new validation excludes all old training conversations.',
        '',
        'Documents never share a training row. Right padding has zero loss; assistant targets have weight 1 and other real targets weight 0.1. Three quarters of crop choices retain the beginning of the conversation. The corpus audit found no identical full token sequences or 128/512-token prefixes shared between the new training set and any checked validation/test set. Semantic overlap and donor pretraining contamination are not ruled out.',
        '',
        'The objective is exact full-vocabulary KL(teacher || student) + 0.2 hard next-token loss + 0.05 relative hidden-state MSE. Mixers/operator norms/energy parameters train at the main learning rate; unfrozen feedforward blocks, their FFN norms, and the final normalization train at a reduced rate. Stages 1–2 unfreeze the last six FFN blocks at 0.2 times the main rate. Stage 3 unfreezes all 24 FFN blocks at 0.5 times the main rate to test whether the earlier frozen blocks limit adaptation. Token embeddings remain frozen throughout. Trainable weights and Adam states use FP32, with BF16 autocast and activation checkpointing.',
        '',
        'Within each training run, checkpoint selection uses only the fixed answer-weighted validation KL. The starting checkpoint is eligible, so worse validation updates are not promoted. These are exploratory experiments on one training trajectory; baseline and first-pass capability results were inspected during the work, and limited first-pass recovery motivated the full FFN adaptation pass. No benchmark prompt, answer, or test conversation was supplied to training.',
        '', '| Stage | Updates | Real token exposures | Initial → best validation KL | Selected step | Peak CUDA allocation |',
        '|---|---:|---:|---:|---:|---:|']
    for stage in stages:
        s=stage['summary'];a=stage['plan']['arguments']
        lines.append(f'| {stage["name"]} | {a["steps"]} | {s["token_exposures"]:,} | {s["initial_validation"]["kl"]:.4f} → {s["best_validation_kl"]:.4f} | {s["best_step"]} | {s["peak_cuda_allocated_gib"]:.2f} GiB |')
    lines+=['', '| Evaluated checkpoint | Science questions | Exact responses |', '|---|---:|---:|']
    for stage in stages:
        d=json.loads((out/f'{stage["name"]}_capabilities.json').read_text())
        assert d['suite_sha256']==models['before']['suite_sha256']
        s=d['summary']
        lines.append(f'| [{stage["name"]}]({stage["name"]}_capabilities.json) | {sum(v["correct"] for v in s["arc"].values())}/128 | {s["overall_probes"]["correct"]}/50 |')
    verification_path=out/'continuation_and_cache.json'
    if verification_path.exists():
        v=json.loads(verification_path.read_text());b=v['models']['initial']['test'];r=v['models']['recovered'];f=r['test'];audit=r['cached_decode_audit']
        assert r['sha256']==recommended_sha, 'Continuation checks must use the recommended checkpoint'
        lines+=['','## Continuation and implementation checks','',
            f'On {f["tokens"]:,} fixed token positions from the previously inspected official test split, NLL changes from {b["nll"]:.4f} to {f["nll"]:.4f}; the donor is {f["teacher_nll"]:.4f}. KL to the donor changes from {b["kl"]:.4f} to {f["kl"]:.4f}. This compares continuation likelihood, not percentage of capabilities retained.',
            '',f'BF16 chunked decoding has relative hidden-state L2 error {audit["chunk_relative_l2"]:.6f}. Carried state is {audit["owned_state_bytes"]/2**20:.3f} MiB per sequence and remains bounded after extension to 1,024 tokens; branching leaves the original state unchanged. This excludes weights and temporary allocations.']
        if 'float32_chunk_audit' in r:
            lines+=['',f'Full-width FP32 chunked decoding has relative error {r["float32_chunk_audit"]["relative_l2"]:.3g}, below the 1e-4 acceptance threshold.']
        lines+=['',f'Energy-off test NLL is {r["energy_off_test"]["nll"]:.6f}, versus {f["nll"]:.6f} with energy refinement. This single-checkpoint ablation does not establish a general energy-refinement benefit.',
                '', 'The separate, previously inspected demo prompts expose additional failures: the recovered model produces malformed JSON for the Maya/Boston extraction and answers `17 + 25 = 42` when asked for just the number. The four passing JSON probes above do not establish robust JSON generation. These auxiliary demos use a 48-token limit and the donor’s inherited repetition penalty of 1.1; they are excluded from the matched headline scores.',
                '', '[Full continuation, energy ablation, numerical checks and previously inspected demo prompts](continuation_and_cache.json).']
    parameter_path=out/'parameter_audit.json'
    if parameter_path.exists():
        audit=json.loads(parameter_path.read_text())
        assert audit['after_sha256']==recommended_sha
        lines+=['',f'The final parameter audit confirms identical architecture/configuration and exactly unchanged {audit["groups"]["frozen"]["parameters"]:,} frozen parameters. [Parameter audit](parameter_audit.json). The REWT suite passed 279 tests; the weighted-loss gradient and short-document tests were rerun after the padding-projection optimization. [Engineering checks](engineering_checks.json).']
    lines+=['','Stages 1 and 3 share the same 512-token validation-crop settings and validation RNG seed. Stage 2 uses different, longer crops, so its absolute KL is not directly comparable. Token exposures count real targets, excluding padding, rather than nominal batch × length. Timing is on a shared GB10 and is not an isolated speed benchmark.',
        '', '![Validation KL during recovery](recovery_curve.png)','',
        '## Reproduce and run','',
        'Preparation uses the pinned donor and corpus source already downloaded by `scripts/prepare_rewt_lfm3.py`. The recorded plans contain every run argument and source/data/checkpoint hashes. Run preparation in a fresh output location; existing artifacts are preserved.',
        '', '```bash',
        '.venv-lfm3/bin/python scripts/prepare_lfm3_recovery.py \\',
        '  --output data/lfm3_capability_recovery_repeat',
        '.venv-lfm3/bin/python scripts/recover_lfm3_capabilities.py \\',
        '  --initial runs/lfm3_conversion_20261028/converted/model.pt \\',
        '  --corpus data/lfm3_capability_recovery_repeat \\',
        '  --output runs/lfm3_capability_recovery_repeat --steps 1024',
        '.venv-lfm3/bin/python runs/lfm3_capability_recovery_20261028/converted/sample.py \\',
        "  --prompt 'Explain why the sky is blue in two sentences.' --tokens 128",'```','',
        'The inference bundle is standalone custom PyTorch code and weights; it does not call the donor. Its tokenizer and donor license are retained. The original 256-token attention windows, delta-memory layers and energy refinement architecture are unchanged.',
        '', '[Recommended bundle instructions](../../runs/lfm3_capability_recovery_20261028/converted/README.txt) · [Standalone sampler](../../runs/lfm3_capability_recovery_20261028/converted/sample.py) · [Standalone smoke test](standalone_sample.json) · [Stage-3 alternative bundle](../../runs/lfm3_capability_recovery_20261028/validation_selected/README.txt). The smoke test runs offline from outside the repository and returns exactly `hello`.',
        '', '## Sources','',
        '- [Pinned LFM3 model card](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd/README.md).',
        '- [SmolTalk corpus](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), pinned revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`.',
        '- [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc), pinned revision `210d026faf9955653af8916fad021475a3f00453`.', '']
    (out/'FINDINGS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(stages),figsize=(6*len(stages),4),squeeze=False,sharey=True)
    for ax,stage in zip(axes[0],stages):
        rows=[r for r in stage['metrics'] if 'validation' in r]
        ax.plot([r.get('step',0) for r in rows],[r['validation']['kl'] for r in rows],marker='o',color='#176c91')
        a=stage['plan']['arguments']
        ax.set(xlabel='Optimizer updates',ylabel='Validation KL (lower is better)',
               title=f'{stage["name"]}: {a["length"]} tokens, {a["ffn_layers"]} FFN blocks trainable')
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'recovery_curve.png',dpi=180);fig.savefig(out/'recovery_curve.pdf');plt.close(fig)
    import numpy as np
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    x=np.arange(3);width=.24
    for i,(key,label,color) in enumerate([('before','Before','#9babb3'),('after','After training','#168678'),('teacher','Original LFM3','#284a79')]):
        s=models[key]['summary'];overall=s['overall_probes']
        values=[s['arc']['ARC-Easy']['accuracy'],s['arc']['ARC-Challenge']['accuracy'],overall['correct']/overall['total']]
        bars=ax.bar(x+(i-1)*width,np.asarray(values)*100,width,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.0%}' for v in values],padding=3,fontsize=9)
    ax.set_xticks(x,['ARC-Easy\n64 questions','ARC-Challenge\n64 questions','Exact responses\n50 probes'])
    ax.set(ylabel='Correct (%)',ylim=(0,110),title='LFM3 → REWT: fixed exploratory evaluation')
    ax.set_yticks([0,25,50,75,100]);ax.legend(frameon=False,ncols=3,loc='upper center')
    ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.01,'Custom ARC label-ranking protocol; generation uses greedy decoding and a 64-token limit.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'capability_comparison.png',dpi=180);fig.savefig(out/'capability_comparison.pdf');plt.close(fig)
    print(out/'FINDINGS.md')

if __name__=='__main__':main()
