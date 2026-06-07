import subprocess
import time
import os
from weights_manifest import WeightsManifest


class WeightsDownloader:
    supported_filetypes = [
        ".ckpt",
        ".safetensors",
        ".sft",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".torchscript",
        ".engine",
        ".patch",
    ]

    def __init__(self):
        self.weights_manifest = WeightsManifest()
        self.weights_map = self.weights_manifest.weights_map

    def get_canonical_weight_str(self, weight_str):
        return self.weights_manifest.get_canonical_weight_str(weight_str)

    def get_weights_by_type(self, type):
        return self.weights_manifest.get_weights_by_type(type)

    def download_weights(self, weight_str):
        if weight_str in self.weights_map:
            if self.weights_manifest.is_non_commercial_only(weight_str):
                print(
                    f"⚠️  {weight_str} is for non-commercial use only. Unless you have obtained a commercial license.\nDetails: https://github.com/replicate/cog-comfyui/blob/main/weights_licenses.md"
                )

            if isinstance(self.weights_map[weight_str], list):
                for weight in self.weights_map[weight_str]:
                    self.download_if_not_exists(
                        weight_str, weight["url"], weight["dest"]
                    )
            else:
                self.download_if_not_exists(
                    weight_str,
                    self.weights_map[weight_str]["url"],
                    self.weights_map[weight_str]["dest"],
                )
        else:
            # Diagnostic: surface what's actually in weights_map so we can tell
            # whether Civitai.weights_map() ran (and with what state) vs being
            # silently empty due to a token-substitution failure.
            civitai_keys = sorted(
                k for k in self.weights_map
                if k.endswith(".safetensors") and (
                    "kodorail" in k or "nova" in k.lower()
                    or "anthro" in k.lower() or "furrytoon" in k.lower()
                    or "cyberrealistic" in k.lower() or "equinox" in k.lower()
                    or "reapony" in k.lower() or "miaomiao" in k.lower()
                    or "icerealistic" in k.lower() or "mopmix" in k.lower()
                )
            )
            raise ValueError(
                f"{weight_str} unavailable. "
                f"[diag] weights_map total={len(self.weights_map)}, "
                f"civitai-style keys present={civitai_keys}. "
                "View the list of available weights: https://github.com/replicate/cog-comfyui/blob/main/supported_weights.md"
            )

    def check_if_file_exists(self, weight_str, dest):
        if dest.endswith(weight_str):
            path_string = dest
        else:
            path_string = os.path.join(dest, weight_str)
        return os.path.exists(path_string)

    def download_if_not_exists(self, weight_str, url, dest):
        if self.check_if_file_exists(weight_str, dest):
            print(f"✅ {weight_str} exists in {dest}")
            return
        WeightsDownloader.download(weight_str, url, dest)

    @staticmethod
    def download(weight_str, url, dest):
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
            os.makedirs(dest, exist_ok=True)

        print(f"⏳ Downloading {weight_str} to {dest}")
        start = time.time()

        # Replicate CDN weights are tar archives extracted into `dest` via
        # `pget -xf`. Raw files (e.g. Civitai's /api/download/models/{id}?token=
        # endpoint) have no .tar extension and must be fetched straight to a
        # file path with `pget -f`. The querystring is stripped before the
        # extension check so `?token=...` doesn't fool the tar branch.
        url_no_qs = url.split("?", 1)[0]
        if url_no_qs.endswith(".tar"):
            cmd = ["pget", "--log-level", "warn", "-xf", url, dest]
        else:
            os.makedirs(dest, exist_ok=True)
            file_dest = os.path.join(dest, os.path.basename(weight_str))
            cmd = ["pget", "--log-level", "warn", "-f", url, file_dest]
        subprocess.check_call(cmd, close_fds=False)

        elapsed_time = time.time() - start
        try:
            file_size_bytes = os.path.getsize(
                os.path.join(dest, os.path.basename(weight_str))
            )
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        except FileNotFoundError:
            print(f"✅ {weight_str} downloaded to {dest} in {elapsed_time:.2f}s")

    def delete_weights(self, weight_str):
        if weight_str in self.weights_map:
            weight_path = os.path.join(self.weights_map[weight_str]["dest"], weight_str)
            if os.path.exists(weight_path):
                os.remove(weight_path)
                print(f"Deleted {weight_path}")
