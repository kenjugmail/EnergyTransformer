"""Sample bytes from any native checkpoint on CPU, including mixed attention.

Usage: python -m scripts.sample_native --checkpoint PATH --prompt 'The '
Small research checkpoints are byte predictors, not instruction-tuned assistants.
"""
from energy_real.sample import main


if __name__ == '__main__':
    main()
