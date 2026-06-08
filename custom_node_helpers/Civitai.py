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
# IMPORTANT: only ONE occurrence of the placeholder string should appear in
# this file. The CI sed step is global, so any other literal
# `__CIVITAI_TOKEN__` (e.g. inside a comparison) would also be substituted,
# producing a tautological `_CIVITAI_TOKEN == _CIVITAI_TOKEN` check that
# silently returns {} and makes every Civitai-hosted checkpoint look
# "unavailable" at predict time.
_CIVITAI_TOKEN = "__CIVITAI_TOKEN__"

# Module-load diagnostic. Use a length heuristic instead of a literal
# placeholder compare so a future sed mishap doesn't false-positive the
# OK branch. The placeholder is 17 chars; a real Civitai token is 32 hex.
print(
    "[Civitai helper] token state: "
    + ("EMPTY" if not _CIVITAI_TOKEN
       else "PLACEHOLDER_LIKE" if len(_CIVITAI_TOKEN) < 20
       else f"OK (tail=...{_CIVITAI_TOKEN[-4:]})"),
    flush=True,
)

# Filename → Civitai modelVersionId. Filename MUST match the safetensors name
# the vault's _COMFY_BASES dict references — that's what ComfyUI's
# CheckpointLoaderSimple looks up under MODELS_PATH/checkpoints/.
#
# Weights are downloaded at PREDICTION time by weights_downloader, not at
# build time, so the image stays small regardless of how many entries are
# listed here. The marginal cost of a new entry is one ~30s-2min cold
# download on the first prediction that references it (subsequent
# predictions on the same warm container reuse the cached file).
_CHECKPOINTS = {
    # Phase 2 — furry/anthro Illustrious bases.
    "novaFurryXL_ilV180A.safetensors":            "2943166",
    "anthroblendIndigo_v30.safetensors":          "3004444",
    # Phase 3 — visual-diversity bases.
    "furrytoonmix_xlV3.safetensors":              "2961728",  # FurryToonMix (IL XL-V3)
    # cyberrealisticPony_semiRealV6 (v3007024) and reapony_v110 (v3003924) were
    # the original picks for the pony_semireal / pony_real_v11 slots, but their
    # Civitai creator flipped the "require sign-in to download" gate after
    # we'd already wired them in — the API token alone returns 401 with
    # {"error":"Unauthorized","message":"creator ... requires you to be
    # logged in"}. There's no public substitute that's an exact match for
    # "semi-real Pony", so we fall back to the canonical realistic-Pony
    # checkpoints that ARE public and R2-hosted.
    "cyberrealisticPony_v180Coreshift.safetensors": "2884631",  # CyberRealistic Pony v1.8.0 Coreshift (pony_semireal slot)
    "BSSEquinoxILSemi_v50.safetensors":             "2973682",  # BSS Equinox IL v5.0
    "ponyRealism_V22.safetensors":                  "914390",   # Pony Realism V22 (pony_real_v11 slot)
    "miaomiaoRealskin_epsV14.safetensors":        "2602600",  # MiaoMiao RealSkin (IL EPS v1.4)
    "icerealisticAnima_v21Noob.safetensors":      "59253",    # IceRealistic (NoobAI v2.1)
    "kodorail_v190plus.safetensors":              "3011012",  # KodoraIL v1.9.0+
    "mopMix_waiZynthra.safetensors":              "2422675",  # MoP Mix (Illustrious)
}


class Civitai(CustomNodeHelper):
    @staticmethod
    def models():
        return list(_CHECKPOINTS.keys())

    @staticmethod
    def weights_map(base_url):
        # Length heuristic instead of literal placeholder compare; see note
        # at top of file about why sed-substituted compares break.
        if not _CIVITAI_TOKEN or len(_CIVITAI_TOKEN) < 20:
            return {}
        ckpt_dest = f"{MODELS_PATH}/checkpoints"
        return {
            filename: {
                "url":  f"https://civitai.com/api/download/models/{version_id}?token={_CIVITAI_TOKEN}",
                "dest": ckpt_dest,
            }
            for filename, version_id in _CHECKPOINTS.items()
        }
