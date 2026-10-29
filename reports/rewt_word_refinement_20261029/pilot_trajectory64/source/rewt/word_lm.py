"""Training/evaluation helpers shared by whole-word REWT language models."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .bpe import checkpoint_tokenizer
from .streaming import StreamingLM, load_streaming
from .words import WordTokenizer, next_words


def forward(model, tokens, state=None, *, rounds=None):
    kwargs = {} if rounds is None else dict(rounds=rounds)
    if isinstance(model, StreamingLM):
        logits, state, _ = model(tokens, state, return_state=not model.training, **kwargs)
        return logits, state
    output = model(tokens, state=state, **kwargs)
    return output.logits, output.state


def load_word_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('arm') == 'streaming_lm':
        model, saved = load_streaming(path)
    else:
        from .position_memory import load_learning_checkpoint
        model, saved = load_learning_checkpoint(path)
    tokenizer = checkpoint_tokenizer(saved)
    if not isinstance(tokenizer, WordTokenizer) or len(tokenizer.vocabulary) != model.config.vocab_size:
        raise ValueError('Expected a whole-word checkpoint with a matching vocabulary')
    return model, tokenizer, saved


@torch.inference_mode()
def evaluate_words(model, corpus, split, offsets, context, *, batch_size=16, rounds=None,
                   unigram=None):
    model.eval()
    device = model.embedding.weight.device
    total_loss = correct = count = word_count = word_correct = word_top5 = known_words = 0
    unknown_count = 0
    for first in range(0, len(offsets), batch_size):
        starts = offsets[first:first + batch_size]
        batch = corpus.batch(split, None, len(starts), context, starts=starts).to(device)
        if unigram is None:
            logits, _ = forward(model, batch.tokens, rounds=rounds)
        else:
            logits = unigram.to(device).log().expand(*batch.targets.shape, -1)
        total_loss += float(F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), reduction='sum'))
        prediction = logits.argmax(-1)
        words = torch.tensor(np.stack([corpus.word_masks[split][s + 1:s + context + 1] for s in starts]),
                             dtype=torch.bool, device=device)
        known = batch.targets != corpus.tokenizer.unk_id
        correct += int((prediction == batch.targets).sum())
        count += batch.targets.numel()
        word_count += int(words.sum())
        known_words += int((words & known).sum())
        word_correct += int(((prediction == batch.targets) & words & known).sum())
        top = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        word_top5 += int(((top == batch.targets[..., None]).any(-1) & words & known).sum())
        unknown_count += int((~known).sum())
    if not count or not word_count:
        raise ValueError('Evaluation requires both tokens and word positions')
    nll = total_loss / count
    return dict(token_nll=nll, token_perplexity=math.exp(min(700, nll)),
                token_accuracy_with_unk=correct / count,
                next_word_accuracy=word_correct / word_count,
                next_word_top5_accuracy=word_top5 / word_count,
                known_word_accuracy=word_correct / max(1, known_words),
                word_vocabulary_coverage=known_words / word_count,
                evaluated_tokens=count, evaluated_word_positions=word_count,
                unknown_target_tokens=unknown_count,
                scope='Teacher-forced next-position prediction. Word accuracy excludes punctuation '
                      'and counts every unseen word as incorrect. NLL/perplexity includes punctuation, EOS and <unk>.')


@torch.inference_mode()
def predict_words(model, tokenizer, prompt, *, top=10, generate=0, temperature=0., seed=42,
                  rounds=None):
    if not isinstance(tokenizer, WordTokenizer) or len(tokenizer.vocabulary) != model.config.vocab_size:
        raise ValueError('Whole-word tokenizer must match the model')
    if (type(generate) is not int or generate < 0 or not math.isfinite(temperature)
            or temperature < 0):
        raise ValueError('Nonnegative generation length and finite temperature required')
    model.eval()
    device = model.embedding.weight.device
    tokens = tokenizer.encode(prompt, bos=True)
    if hasattr(model.config, 'context') and len(tokens) + generate > model.config.context:
        raise ValueError('Prompt plus generation exceeds this model context')
    logits, state = forward(model, torch.tensor([tokens], device=device), rounds=rounds)
    candidates = next_words(logits[0, -1], tokenizer, top)
    generated = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(generate):
        scores = logits[0, -1].clone()
        scores[tokenizer.bos_id] = -torch.inf
        # Do not offer the unknown marker as a useful generated word.
        scores[tokenizer.unk_id] = -torch.inf
        token = int(torch.multinomial((scores / temperature).softmax(-1), 1, generator=generator)) if temperature else int(scores.argmax())
        generated.append(token)
        if token == tokenizer.eos_id:
            break
        logits, state = forward(model, torch.tensor([[token]], device=device), state, rounds=rounds)
    suffix = tokenizer.decode_continuation(generated, prefix=prompt).decode('utf-8')
    unknown = [w for w in tokenizer.split(prompt) if w not in tokenizer.ids]
    return dict(prompt=prompt, candidates=candidates, unknown_prompt_tokens=unknown,
                continuation=suffix, text=prompt + suffix, generated_token_ids=generated,
                tokenization=tokenizer.kind, vocabulary_size=len(tokenizer.vocabulary),
                candidate_scope='Whole words/numbers at the immediate next token; punctuation and special tokens omitted. '
                                'Probabilities retain the full vocabulary denominator.')
