"""Local Python startup hooks for this repository."""

from nfs_safetensors_patch import apply_patch


apply_patch()
