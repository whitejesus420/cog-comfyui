"""Custom node helper: register Civitai-hosted checkpoints at container start.

Adds Civitai-hosted SDXL-family checkpoints not in cog-comfyui's bundled
weights. Vault tavern away-mode picks one of these as the base.

The auth token is baked in at `cog push` time by the
.github/workflows/push-to-replicate.yml sed substitution against the
__CIVITAI_TOKEN__ placeholder, so this file stays token-free in the public
repo. The runtime download is performed by the patched weights_downloader
(non-.tar branch using `pget -f` to a file path).
"""

from custom_node_helper import CustomNodeHelper
from config import config

MODELS_PATH = config["MODELS_PATH"]

# Replaced by .github/workflows/push-to-replicate.yml at build time.
# If you still see the literal placeholder in a deployed image, the sed step
# did not run and weights_map will return {} -> predictions referencing these
# checkpoints will fail at the weights-resolve step rather than hitting
# Civitai with an invalid token.
_CIVITAI_TOKEN = "__CIVITAI_TOKEN__"

# Filename → Civitai modelVersionId. Filename MUST match the safetensors name
# the vault's _COMFY_BASES dict references — that's what ComfyUI's
# CheckpointLoaderSimple looks up under MODELS_PATH/checkpoints/.
#
# Disk-space note: each checkpoint is ~6.8GB; adding more than ~5 here at
# once will bust the GitHub Actions ubuntu-latest runner's free disk during
# the `cog push` build step. Stage additions across multiple workflow runs
# (the next run picks up cached layers, so the marginal disk cost on a
# subsequent run is only the new weights).
_CHECKPOINTS = {
    # Phase 2 — furry/anthro Illustrious bases.
    "novaFurryXL_ilV180A.safetensors":          "2943166",
    "anthroblendIndigo_v30.safetensors":        "3004444",
    # Phase 3 batch 1 — visual-diversity bases.
    "furrytoonmix_xlV3.safetensors":            "2961728",  # FurryToonMix (IL XL-V3)
    "cyberrealisticPony_semiRealV6.safetensors": "3007024",  # CyberRealistic Pony Semi-Real v6
    "BSSEquinoxILSemi_v50.safetensors":         "2973682",  # BSS Equinox IL v5.0
    "reapony_v110.safetensors":                 "3003924",  # ReaPony v11.0
}


class Civitai(CustomNodeHelper):
    @staticmethod
    def models():
        return list(_CHECKPOINTS.keys())

    @staticmethod
    def weights_map(base_url):
        if not _CIVITAI_TOKEN or _CIVITAI_TOKEN == "__CIVITAI_TOKEN__":
            return {}
        ckpt_dest = f"{MODELS_PATH}/checkpoints"
        return {
            filename: {
                "url":  f"https://civitai.com/api/download/models/{version_id}?token={_CIVITAI_TOKEN}",
                "dest": ckpt_dest,
            }
            for filename, version_id in _CHECKPOINTS.items()
        }
