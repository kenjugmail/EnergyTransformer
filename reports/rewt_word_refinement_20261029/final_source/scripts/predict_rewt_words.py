"""Show complete next-word predictions from a whole-word REWT checkpoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.word_lm import load_word_model, predict_words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/rewt_word_refinement_20261029/recommended.pt')
    parser.add_argument('--prompt', default='Once upon a')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--generate', type=int, default=0, help='optional number of word/punctuation tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        model, tokenizer, _ = load_word_model(args.checkpoint)
        result = predict_words(model.to(args.device), tokenizer, args.prompt, top=args.top,
                               generate=args.generate, temperature=args.temperature,
                               seed=args.seed, rounds=args.rounds)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print('Prompt:', args.prompt)
        print('Next whole words (punctuation omitted):')
        for row in result['candidates']:
            print(f'  {row["word"]:<20} {row["probability"]:.2%}')
        if result['unknown_prompt_tokens']:
            print('Unseen input words mapped to <unk>:', ', '.join(result['unknown_prompt_tokens']))
        if args.generate:
            print('\n' + result['text'])


if __name__ == '__main__':
    main()
