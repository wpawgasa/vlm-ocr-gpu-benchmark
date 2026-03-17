"""YAML configuration loading with hierarchical merge and Pydantic validation."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import structlog
import yaml

from vlm_ocr_bench.config.schema import ExperimentConfig

logger = structlog.get_logger()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = path.resolve()
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"Expected YAML mapping at top level, got {type(data).__name__} in {path}"
        raise TypeError(msg)
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. Lists are replaced, not appended. Override wins."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _find_config_root(experiment_path: Path) -> Path:
    """Walk up from experiment config to find the configs/ root directory."""
    path = experiment_path.resolve().parent
    while path != path.parent:
        if path.name == "configs":
            return path
        path = path.parent
    return experiment_path.resolve().parent.parent


def load_experiment_config(
    path: Path,
    cli_overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load and validate a full experiment config with hierarchical merge.

    Merge order: defaults → hardware → model → phase → experiment → CLI overrides.
    """
    config_root = _find_config_root(path)
    merged: dict[str, Any] = {}

    # 1. Load defaults
    defaults_path = config_root / "defaults.yaml"
    if defaults_path.exists():
        defaults = load_yaml(defaults_path)
        merged = deep_merge(merged, defaults)
        logger.debug("loaded_defaults", path=str(defaults_path))

    # 2. Load experiment manifest
    experiment = load_yaml(path)

    # 3. Merge hardware configs referenced by experiment
    gpus = experiment.get("gpus", [])
    gpu_configs: dict[str, Any] = {}
    for gpu_name in gpus:
        hw_path = config_root / "hardware" / f"{gpu_name}.yaml"
        if hw_path.exists():
            hw_data = load_yaml(hw_path)
            gpu_configs[gpu_name] = hw_data
            logger.debug("loaded_hardware_config", gpu=gpu_name, path=str(hw_path))
    if gpu_configs:
        experiment.setdefault("gpu_configs", {})
        for gpu_name, hw_data in gpu_configs.items():
            experiment["gpu_configs"][gpu_name] = deep_merge(
                experiment["gpu_configs"].get(gpu_name, {}), hw_data
            )

    # 4. Merge model configs referenced by experiment
    models = experiment.get("models", [])
    model_configs: dict[str, Any] = {}
    for model_name in models:
        model_path = config_root / "models" / f"{model_name}.yaml"
        if model_path.exists():
            model_data = load_yaml(model_path)
            model_configs[model_name] = model_data
            logger.debug("loaded_model_config", model=model_name, path=str(model_path))
    if model_configs:
        experiment.setdefault("model_configs", {})
        for model_name, model_data in model_configs.items():
            experiment["model_configs"][model_name] = deep_merge(
                experiment["model_configs"].get(model_name, {}), model_data
            )

    # 5. Merge phase configs
    for phase_name in ("inference", "training", "quality"):
        phase_path = config_root / "phases" / f"{phase_name}.yaml"
        if phase_path.exists() and phase_name not in experiment:
            phase_data = load_yaml(phase_path)
            experiment[phase_name] = phase_data
            logger.debug("loaded_phase_config", phase=phase_name, path=str(phase_path))

    # 6. Deep merge: defaults + experiment
    merged = deep_merge(merged, experiment)

    # 7. Apply CLI overrides
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)
        logger.debug("applied_cli_overrides", overrides=cli_overrides)

    # 8. Validate via Pydantic
    config = ExperimentConfig.model_validate(merged)
    logger.info("config_loaded", experiment=config.name, models=len(config.models))
    return config


def save_config_snapshot(config: ExperimentConfig, output_dir: Path) -> Path:
    """Save a resolved config snapshot to the output directory for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.json"
    with open(snapshot_path, "w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, default=str)
    logger.info("config_snapshot_saved", path=str(snapshot_path))
    return snapshot_path


def save_config_yaml_copy(source_path: Path, output_dir: Path) -> Path:
    """Copy the original YAML config file to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / source_path.name
    shutil.copy2(source_path, dest)
    return dest
