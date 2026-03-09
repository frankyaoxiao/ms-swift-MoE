"""Export GPT-OSS Megatron checkpoint to HF format.

Monkey-patches PreTrainedTokenizerFast.__init__ to handle GPT-OSS's
tokenizer (vocab_file=None) which crashes in transformers 4.57.6's
convert_slow_tokenizer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nfs_safetensors_patch import apply_patch as apply_nfs_safetensors_patch

apply_nfs_safetensors_patch()

# Monkey-patch the __init__ to skip the broken tiktoken fallback branch
# when tokenizer.json is available but not being picked up properly.
import transformers.tokenization_utils_fast as _tuf
import copy

_orig_init = _tuf.PreTrainedTokenizerFast.__init__


def _patched_init(self, *args, **kwargs):
    """Wrap __init__ to catch the convert_slow_tokenizer error and retry
    by finding tokenizer.json from the name_or_path."""
    try:
        _orig_init(self, *args, **kwargs)
    except (AttributeError, ValueError) as e:
        if 'endswith' not in str(e) and 'vocab_file' not in str(e) and 'convert' not in str(e).lower():
            raise
        # The tokenizer.json wasn't passed. Try to find it.
        name_or_path = kwargs.get('name_or_path', '') or getattr(self, 'name_or_path', '')
        tokenizer_file = None
        # Check if name_or_path is a directory with tokenizer.json
        if name_or_path and os.path.isdir(name_or_path):
            candidate = os.path.join(name_or_path, 'tokenizer.json')
            if os.path.isfile(candidate):
                tokenizer_file = candidate
        # Try HF cache
        if tokenizer_file is None:
            from huggingface_hub import try_to_load_from_cache
            try:
                cached = try_to_load_from_cache(name_or_path, 'tokenizer.json')
                if cached and isinstance(cached, str) and os.path.isfile(cached):
                    tokenizer_file = cached
            except Exception:
                pass

        if tokenizer_file is None:
            raise RuntimeError(
                f"Cannot load tokenizer: convert_slow_tokenizer failed ({e}) "
                f"and could not find tokenizer.json for '{name_or_path}'"
            ) from e

        # Retry with tokenizer_file explicitly set
        kwargs['tokenizer_file'] = tokenizer_file
        _orig_init(self, *args, **kwargs)


_tuf.PreTrainedTokenizerFast.__init__ = _patched_init

from swift.megatron.export.export import megatron_export_main

if __name__ == '__main__':
    megatron_export_main()
