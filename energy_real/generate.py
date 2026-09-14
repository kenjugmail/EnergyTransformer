"""Compatibility entry point: use the unified prefill/decode benchmark."""
from .bench import tensor_bytes as state_bytes, parser, run

if __name__ == '__main__':
    run(parser().parse_args())
