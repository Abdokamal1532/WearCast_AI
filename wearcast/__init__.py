import huggingface_hub

# Global Compatibility Patch: Standardize cached_download for newer huggingface_hub versions
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# This ensures that any module importing wearcast will have the correct patches applied.
