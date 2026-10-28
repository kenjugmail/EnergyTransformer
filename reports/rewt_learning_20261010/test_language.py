"""Locked, document-disjoint language evaluation; no parameter updates."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from rewt.bpe import load_story_tokenizer
from rewt.data import file_sha256
from rewt.evaluate import generate_batch, refinement_audit, repetition_metrics, runtime_profile
from rewt.experiment import save_json, save_lines
from rewt.position_memory import load_learning_checkpoint
from rewt.story_data import StoryCorpus

PARENT = Path(__file__).resolve().parent
OUT = PARENT / 'language_test'


def weights_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(value.shape).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Keep the original locked test results; do not overwrite them.')
    training = json.loads((PARENT / 'language_bpe_extended/plan.json').read_text())
    fourier = [json.loads(p.read_text()) for p in
               (PARENT / 'language_bpe_extended').glob('fourier*.json')]
    selected = min(fourier, key=lambda r: r['best_validation_bpb'])
    records = [selected] + [json.loads((PARENT / 'language_bpe_extended' / f'{name}.json').read_text())
                            for name in ('neural80', 'transformer')]
    plan = dict(created=datetime.now(timezone.utc).isoformat(), checkpoints=records,
        primary=selected['case'], training_plan_sha256=file_sha256(PARENT / 'language_bpe_extended/plan.json'),
        tokenizer=training['tokenizer'], corpus=training['corpus'],
        test_offsets=training['test_offsets'], prompts=training['text_prompts'],
        context_tokens=128, maximum_new_tokens=96, generation_seed=1991, temperature=.5,
        selection='Lowest validation bpb among the completed Fourier language variants. The paired 80-wide neural-energy and Transformer controls use the same BPE vocabulary, batches, target-token budget, and continuation schedule.',
        decoding='Fixed temperature .5, informed by the preceding validation-only decoding sweep; no claim that this is an optimal decoding policy. Same sampling uniforms for all three arms.',
        model_inputs='Frozen training vocabulary and encoded prefix only; no document identity, reference continuation, task payload, regression head, or solver enters generation.',
        limitations='One independent initialization per arm, 4096 total updates. Equal target-token budgets do not imply equal CPU time. Text ratings by the implementing assistant are exploratory and are not independent human evaluation.',
        source_sha256={p.name: file_sha256(p) for p in (ROOT / 'rewt').glob('*.py')},
        script_sha256=file_sha256(Path(__file__)))
    for record in records:
        assert file_sha256(Path(record['checkpoint'])) == record['checkpoint_sha256']
    save_json(OUT / 'plan.json', plan)
    corpus = StoryCorpus(training['data_directory'])
    assert corpus.identity() == training['corpus']
    tokenizer = load_story_tokenizer(training['tokenizer']['path'])
    assert file_sha256(Path(training['tokenizer']['path'])) == training['tokenizer']['sha256']
    spec = importlib.util.spec_from_file_location('locked_language_metrics', PARENT / 'language_bpe_extended.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    results, all_rows = [], []
    for record in records:
        model, checkpoint = load_learning_checkpoint(record['checkpoint'])
        model.eval()
        before = weights_sha(model)
        start = time.process_time()
        metrics = helper.token_metrics(model, corpus, plan['test_offsets'], tokenizer, split='test')
        metric_cpu = time.process_time() - start
        rows = []
        start = time.process_time()
        for first in range(0, len(plan['prompts']), 10):
            prompts = plan['prompts'][first:first + 10]
            outputs = generate_batch(model, [bytes.fromhex(p['prompt_hex']) for p in prompts],
                max_new=plan['maximum_new_tokens'], sampling=True, temperature=plan['temperature'],
                seed=plan['generation_seed'] + first)
            for prompt, tokens in zip(prompts, outputs):
                raw = tokenizer.decode(tokens)
                rows.append(dict(case=record['case'], prompt_id=prompt['id'], prompt=prompt['prompt'],
                    generation=raw.decode('ascii'), generation_hex=raw.hex(), generation_token_hex=tokens.hex(),
                    generated_tokens=len(tokens), reference_hex=prompt['reference_hex'], metrics=repetition_metrics(raw)))
        generation_cpu = time.process_time() - start
        save_lines(OUT / f"{record['case']}_outputs.jsonl", rows)
        all_rows.extend(rows)
        batch = corpus.batch('test', None, 2, 32, starts=plan['test_offsets'][:2])
        audit = refinement_audit(model, batch.tokens)
        profile = runtime_profile(model, batch.tokens, repeats=3)
        profile['cpu_ms_per_token_median'] = profile.pop('cpu_ms_per_byte_median')
        profile['input_tokens'] = batch.tokens.numel()
        profile['input_raw_bytes'] = int(torch.tensor(tokenizer.byte_lengths)[batch.tokens].sum())
        after = weights_sha(model)
        assert before == after
        assert file_sha256(Path(record['checkpoint'])) == record['checkpoint_sha256']
        if audit:
            assert audit['all_accepted_states_nonincreasing']
        summary = {key: sum(row['metrics'][key] for row in rows) / len(rows) for key in rows[0]['metrics']}
        result = dict(case=record['case'], checkpoint_sha256=record['checkpoint_sha256'],
            parameters=model.parameters_count(), test=metrics, test_cpu_seconds=metric_cpu,
            generation_cpu_seconds=generation_cpu, text_count=len(rows), text_metrics=summary,
            energy_audit=audit, runtime_profile=profile, weights_before=before, weights_after=after,
            weights_unchanged=before == after)
        results.append(result)
        print(json.dumps(result), flush=True)
    # Rating packet omits model identities and reference continuations.
    random.Random(1993).shuffle(all_rows)
    anonymous, key = [], []
    for i, row in enumerate(all_rows):
        sample_id = f'sample_{i:03d}'
        anonymous.append(dict(id=sample_id, prompt=row['prompt'], generation=row['generation']))
        key.append(dict(id=sample_id, case=row['case'], prompt_id=row['prompt_id']))
    save_lines(OUT / 'anonymous_texts.jsonl', anonymous)
    save_lines(OUT / 'rating_key.jsonl', key)
    for name, digest in plan['source_sha256'].items():
        assert file_sha256(ROOT / 'rewt' / name) == digest
    save_json(OUT / 'results.json', results)


if __name__ == '__main__':
    main()
