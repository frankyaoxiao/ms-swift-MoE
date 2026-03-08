"""Patch Swift's sharded safetensors saver for NFS-safe temporary filenames.

Swift currently writes temporary shard names like:
    model-00001-of-?????.safetensors

Our /data mount is NFS and rejects '?' in filenames with EINVAL, which causes
final merged checkpoint export to fail after training has already completed.
"""

from __future__ import annotations

import os


_TMP_TOTAL_PLACEHOLDER = "00000"


def _tmp_shard_name(index: int) -> str:
    return f"model-{index:05d}-of-{_TMP_TOTAL_PLACEHOLDER}.safetensors"


def apply_patch() -> bool:
    try:
        from safetensors.torch import save_file
        from swift.megatron.utils.io_utils import StreamingSafetensorSaver
    except Exception:
        return False

    if getattr(StreamingSafetensorSaver, "_nfs_safe_patch_applied", False):
        return True

    def _save_current_shard(self, shard_filename: str = None):
        if not self.current_shard:
            return
        if shard_filename is None:
            if self.is_peft_format:
                shard_filename = "adapter_model.safetensors"
            else:
                shard_filename = _tmp_shard_name(self.shard_index)

        shard_path = os.path.join(self.save_dir, shard_filename)
        save_file(self.current_shard, str(shard_path))
        for key in self.current_shard.keys():
            self.weight_map[key] = shard_filename

        self.total_size += self.current_shard_size
        self.current_shard = {}
        self.current_shard_size = 0
        self.shard_index += 1

    def _finalize(self):
        if not self.is_save_rank:
            return
        if self.current_shard:
            self._save_current_shard()
        if self.is_peft_format:
            return

        total_shards = self.shard_index - 1
        for i in range(1, total_shards + 1):
            old_path = os.path.join(self.save_dir, _tmp_shard_name(i))
            if total_shards == 1:
                new_name = "model.safetensors"
            else:
                new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            new_path = os.path.join(self.save_dir, new_name)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)

        if total_shards > 1:
            updated_weight_map = {}
            for key, filename in self.weight_map.items():
                updated_weight_map[key] = filename.replace(
                    _TMP_TOTAL_PLACEHOLDER, f"{total_shards:05d}", 1)

            self._save_index(updated_weight_map)

    StreamingSafetensorSaver._save_current_shard = _save_current_shard
    StreamingSafetensorSaver.finalize = _finalize
    StreamingSafetensorSaver._nfs_safe_patch_applied = True
    return True

