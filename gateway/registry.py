"""Model registry — scans /data/models and merges with models.yaml overrides."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# Model types that vLLM can serve (exclude TTS, multimodal, custom arches)
VLLM_MODEL_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "phi3"}

# Directories to skip when scanning (HF cache dirs, non-model dirs)
SKIP_PREFIXES = ("models--", "huggingface", "whisper", "crv_")


@dataclass
class ModelInfo:
    model_id: str  # e.g. "Qwen/Qwen2.5-14B-Instruct-AWQ"
    name: str  # short name e.g. "qwen2.5-14b-instruct-awq"
    path: str  # absolute path on disk
    architecture: str  # model_type from config.json
    quant_method: str  # "awq", "gptq", "none"
    vram_mb: int  # estimated VRAM for weights
    kv_reserve_mb: int = 4096  # extra VRAM for KV cache
    tool_call_parser: str = "hermes"
    thinking_suppression: bool = False
    dtype: str = "half"
    quantization: str | None = None  # vLLM --quantization flag
    max_model_len: int = 32768
    pinned: bool = False
    aliases: list[str] = field(default_factory=list)


def _infer_tool_call_parser(model_type: str) -> str:
    if model_type in ("qwen2", "qwen3"):
        return "hermes"
    if model_type == "llama":
        return "llama3_json"
    if model_type == "mistral":
        return "mistral"
    return "hermes"


def _infer_thinking_suppression(model_type: str) -> bool:
    return model_type == "qwen3"


def _estimate_vram(model_dir: str, quant: str) -> int:
    """Estimate weight VRAM in MB from total safetensors/bin size on disk."""
    total = 0
    for f in os.listdir(model_dir):
        if f.endswith((".safetensors", ".bin")):
            total += os.path.getsize(os.path.join(model_dir, f))
    mb = total / (1024 * 1024)
    if quant in ("awq", "gptq", "awq_marlin"):
        return int(mb * 1.1)
    return int(mb * 1.05)


def _infer_quantization_flag(quant: str) -> str | None:
    if quant == "awq":
        return "awq_marlin"
    if quant == "gptq":
        return "gptq_marlin"
    return None


def _infer_dtype(quant: str) -> str:
    if quant in ("awq", "gptq", "awq_marlin"):
        return "half"
    return "auto"


def _make_short_name(model_id: str) -> str:
    """Turn 'Qwen/Qwen2.5-14B-Instruct-AWQ' → 'qwen2.5-14b-instruct-awq'."""
    return model_id.split("/")[-1].lower()


def _scan_model_dir(model_dir: str) -> ModelInfo | None:
    """Read config.json from a model directory and build ModelInfo."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return None

    with open(config_path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "")
    if model_type not in VLLM_MODEL_TYPES:
        return None

    quant = cfg.get("quantization_config", {}).get("quant_method", "none")

    # Derive model_id from path structure (org/model)
    parts = Path(model_dir).parts
    # Try to find org/model pattern
    models_idx = None
    for i, p in enumerate(parts):
        if p == "models":
            models_idx = i
            break
    if models_idx is not None and len(parts) > models_idx + 2:
        model_id = "/".join(parts[models_idx + 1 :])
    else:
        model_id = parts[-1]

    return ModelInfo(
        model_id=model_id,
        name=_make_short_name(model_id),
        path=model_dir,
        architecture=model_type,
        quant_method=quant,
        vram_mb=_estimate_vram(model_dir, quant),
        tool_call_parser=_infer_tool_call_parser(model_type),
        thinking_suppression=_infer_thinking_suppression(model_type),
        dtype=_infer_dtype(quant),
        quantization=_infer_quantization_flag(quant),
    )


def _walk_models_dir(models_dir: str) -> list[ModelInfo]:
    """Recursively find model directories (those containing config.json)."""
    results = []
    base = Path(models_dir)
    if not base.is_dir():
        log.warning("Models directory %s does not exist", models_dir)
        return results

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        if any(entry.name.startswith(p) for p in SKIP_PREFIXES):
            continue
        # Check direct config.json
        if (entry / "config.json").is_file():
            info = _scan_model_dir(str(entry))
            if info:
                results.append(info)
        else:
            # One level deeper (org/model pattern)
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and (sub / "config.json").is_file():
                    info = _scan_model_dir(str(sub))
                    if info:
                        results.append(info)
    return results


def _load_yaml_overrides(yaml_path: str) -> dict:
    """Load models.yaml, return dict keyed by model_id."""
    if not yaml_path or not os.path.isfile(yaml_path):
        return {}
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    return {m["model_id"]: m for m in data.get("models", []) if "model_id" in m}


def load_registry(
    models_dir: str, config_path: str | None = None
) -> dict[str, ModelInfo]:
    """Scan models_dir, merge yaml overrides, return {name: ModelInfo} with alias entries."""
    infos = _walk_models_dir(models_dir)
    overrides = _load_yaml_overrides(config_path)

    registry: dict[str, ModelInfo] = {}

    for info in infos:
        ov = overrides.get(info.model_id, {})
        if "vram_mb" in ov:
            info.vram_mb = ov["vram_mb"]
        if "kv_reserve_mb" in ov:
            info.kv_reserve_mb = ov["kv_reserve_mb"]
        if "tool_call_parser" in ov:
            info.tool_call_parser = ov["tool_call_parser"]
        if "thinking_suppression" in ov:
            info.thinking_suppression = ov["thinking_suppression"]
        if "aliases" in ov:
            info.aliases = ov["aliases"]
        if "pinned" in ov:
            info.pinned = ov["pinned"]
        if "max_model_len" in ov:
            info.max_model_len = ov["max_model_len"]
        if "dtype" in ov:
            info.dtype = ov["dtype"]
        if "quantization" in ov:
            info.quantization = ov["quantization"]

        # Register by short name
        registry[info.name] = info
        # Register by model_id
        registry[info.model_id] = info
        # Register aliases
        for alias in info.aliases:
            registry[alias] = info

    log.info("Registry loaded: %d models, %d entries (with aliases)", len(infos), len(registry))
    return registry
