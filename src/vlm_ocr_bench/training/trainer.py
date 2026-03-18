"""Training benchmark runner — orchestrates the full Phase B sweep."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    ModelConfig,
    PrecisionMode,
    TrainingConfig,
)
from vlm_ocr_bench.training.callbacks import MemoryCallback, PowerCallback, ThroughputCallback
from vlm_ocr_bench.training.deepspeed_config import build_deepspeed_config
from vlm_ocr_bench.training.lora import build_peft_config, get_trainable_param_summary
from vlm_ocr_bench.training.metrics import TrainingMetrics, compute_training_metrics

logger = structlog.get_logger()


# ─── Result Dataclasses ───


@dataclass
class TrainingConfigResult:
    """Result from a single (micro_batch_size, precision, run_id) configuration."""

    micro_batch_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    precision: str
    run_id: int
    metrics: TrainingMetrics
    steps_completed: int = 0
    wall_time_s: float = 0.0
    oom: bool = False
    error: str | None = None


@dataclass
class ConvergenceResult:
    """Result from a full convergence training run."""

    precision: str
    micro_batch_size: int
    epochs_completed: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    best_val_epoch: int = 0
    total_training_time_hours: float = 0.0
    total_energy_kwh: float = 0.0
    total_cost_estimate: float = 0.0
    loss_curve: list[tuple[int, float]] = field(default_factory=list)
    val_metrics: list[tuple[int, dict[str, float]]] = field(default_factory=list)


@dataclass
class TrainingBenchmarkResult:
    """Complete result from a Phase B training benchmark."""

    model_name: str
    gpu_type: str
    trainable_param_summary: dict[str, Any] = field(default_factory=dict)
    configs: list[TrainingConfigResult] = field(default_factory=list)
    convergence_run: ConvergenceResult | None = None
    max_batch_sizes: dict[str, int] = field(default_factory=dict)
    total_wall_time_s: float = 0.0


# ─── Runner ───


class TrainingBenchmarkRunner:
    """Orchestrates Phase B: LoRA fine-tuning benchmark sweep.

    For each (micro_batch_size, precision) combo:
    1. Load base model + apply LoRA
    2. Configure HF Trainer with metric callbacks
    3. Warmup steps (discarded)
    4. Run measurement steps (multiple runs, resetting LoRA weights)
    5. Record throughput, loss, power
    6. Binary search for max feasible batch size (OOM boundary)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
        training_config: TrainingConfig,
    ) -> None:
        self._model_config = model_config
        self._gpu_config = gpu_config
        self._training_config = training_config

    def run(self) -> TrainingBenchmarkResult:
        """Execute the full training benchmark sweep."""
        total_start = time.monotonic()

        result = TrainingBenchmarkResult(
            model_name=self._model_config.name,
            gpu_type=self._gpu_config.gpu_type.value,
        )

        for precision in self._training_config.precision_modes:
            try:
                self._run_precision_sweep(precision, result)
            except Exception:
                logger.error(
                    "precision_sweep_failed",
                    precision=precision.value,
                    model=self._model_config.name,
                    exc_info=True,
                )

        result.total_wall_time_s = time.monotonic() - total_start
        logger.info(
            "training_benchmark_complete",
            model=self._model_config.name,
            num_configs=len(result.configs),
            total_time_s=round(result.total_wall_time_s, 1),
        )
        return result

    def _run_precision_sweep(
        self,
        precision: PrecisionMode,
        result: TrainingBenchmarkResult,
    ) -> None:
        """Run all micro_batch_size combos for one precision."""
        logger.info(
            "starting_precision_sweep",
            precision=precision.value,
            model=self._model_config.name,
        )

        # Load model + LoRA
        model, tokenizer = self._load_model_with_lora(precision)

        if model is None:
            logger.error(
                "model_load_failed",
                precision=precision.value,
                model=self._model_config.name,
            )
            return

        # Record trainable parameter summary (once per precision)
        if not result.trainable_param_summary:
            result.trainable_param_summary = get_trainable_param_summary(model)

        # Save initial LoRA weights for reset between runs
        initial_lora_state = self._get_lora_state_dict(model)

        try:
            # Find max batch size via binary search
            max_bs = self._find_max_batch_size(model, tokenizer, precision)
            result.max_batch_sizes[precision.value] = max_bs

            # Run sweep for each configured micro_batch_size
            for micro_bs in self._training_config.micro_batch_sizes:
                if micro_bs > max_bs:
                    logger.info(
                        "skipping_batch_size_exceeds_max",
                        micro_batch_size=micro_bs,
                        max_batch_size=max_bs,
                        precision=precision.value,
                    )
                    continue

                for run_id in range(self._training_config.runs_per_config):
                    # Reset LoRA weights for fair comparison
                    if initial_lora_state is not None:
                        self._reset_lora_weights(model, initial_lora_state)

                    config_result = self._run_single_config(
                        model=model,
                        tokenizer=tokenizer,
                        micro_batch_size=micro_bs,
                        precision=precision,
                        run_id=run_id,
                    )
                    result.configs.append(config_result)

        finally:
            self._cleanup_model(model)

    def _run_single_config(
        self,
        model: Any,
        tokenizer: Any,
        micro_batch_size: int,
        precision: PrecisionMode,
        run_id: int,
    ) -> TrainingConfigResult:
        """Run a single training configuration: warmup + measurement."""
        gradient_accumulation_steps = max(
            1, self._training_config.effective_batch_size // micro_batch_size
        )

        logger.info(
            "running_training_config",
            micro_batch_size=micro_batch_size,
            precision=precision.value,
            run_id=run_id,
            grad_accum=gradient_accumulation_steps,
        )

        try:
            return self._execute_training_config(
                model=model,
                tokenizer=tokenizer,
                micro_batch_size=micro_batch_size,
                precision=precision,
                run_id=run_id,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        except Exception as exc:
            oom = "out of memory" in str(exc).lower()
            logger.warning(
                "training_config_failed",
                micro_batch_size=micro_batch_size,
                precision=precision.value,
                error=str(exc),
                oom=oom,
            )
            return TrainingConfigResult(
                micro_batch_size=micro_batch_size,
                effective_batch_size=self._training_config.effective_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                precision=precision.value,
                run_id=run_id,
                metrics=TrainingMetrics(),
                oom=oom,
                error=str(exc),
            )

    def _execute_training_config(
        self,
        model: Any,
        tokenizer: Any,
        micro_batch_size: int,
        precision: PrecisionMode,
        run_id: int,
        gradient_accumulation_steps: int,
    ) -> TrainingConfigResult:
        """Execute warmup + measurement for a single training config."""
        from transformers import Trainer

        # Create callbacks
        throughput_cb = ThroughputCallback()
        memory_cb = MemoryCallback()
        power_cb = PowerCallback()

        total_steps = self._training_config.warmup_steps + self._training_config.measurement_steps

        # Build training arguments
        training_args = self._build_training_args(
            micro_batch_size=micro_batch_size,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            total_steps=total_steps,
        )

        # Build dataset
        train_dataset = self._build_dummy_dataset(
            tokenizer=tokenizer,
            num_samples=total_steps * micro_batch_size,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self._build_data_collator(tokenizer),
            callbacks=[throughput_cb, memory_cb, power_cb],
        )

        # Train
        wall_start = time.monotonic()
        train_result = trainer.train()
        wall_time_s = time.monotonic() - wall_start

        # Collect metrics
        steps_completed = (
            train_result.global_step if hasattr(train_result, "global_step") else total_steps
        )
        loss_values = [
            log.get("loss", 0.0) for log in (trainer.state.log_history or []) if "loss" in log
        ]

        metrics = compute_training_metrics(
            num_samples=steps_completed * micro_batch_size,
            num_tokens=0,  # Will be populated if tokenizer tracking is available
            wall_time_s=wall_time_s,
            peak_gpu_memory_gb=memory_cb.get_peak_memory_gb(),
            mean_power_watts=power_cb.get_mean_power_watts(),
            peak_power_watts=power_cb.get_peak_power_watts(),
            loss_values=loss_values,
        )

        return TrainingConfigResult(
            micro_batch_size=micro_batch_size,
            effective_batch_size=self._training_config.effective_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            precision=precision.value,
            run_id=run_id,
            metrics=metrics,
            steps_completed=steps_completed,
            wall_time_s=wall_time_s,
        )

    def _build_training_args(
        self,
        micro_batch_size: int,
        precision: PrecisionMode,
        gradient_accumulation_steps: int,
        total_steps: int,
    ) -> Any:
        """Build HF TrainingArguments for a benchmark run."""
        from transformers import TrainingArguments

        args_kwargs: dict[str, Any] = {
            "output_dir": "/tmp/vlm_ocr_bench_training",
            "per_device_train_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": self._training_config.learning_rate,
            "lr_scheduler_type": self._training_config.lr_scheduler,
            "warmup_ratio": self._training_config.warmup_ratio,
            "max_steps": total_steps,
            "logging_steps": 10,
            "save_strategy": "no",
            "remove_unused_columns": False,
            "dataloader_pin_memory": True,
            "optim": self._training_config.optimizer,
            "report_to": "none",
        }

        # Precision settings
        if precision == PrecisionMode.BF16:
            args_kwargs["bf16"] = True
        elif precision == PrecisionMode.FP16:
            args_kwargs["fp16"] = True
        elif precision in (PrecisionMode.FP8,):
            args_kwargs["bf16"] = True  # FP8 training uses bf16 with FP8 compute

        # Gradient checkpointing for 7B+ models
        if self._training_config.gradient_checkpointing:
            args_kwargs["gradient_checkpointing"] = True
            args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        # DeepSpeed for 7B+ models
        if self._training_config.deepspeed_stage is not None:
            args_kwargs["deepspeed"] = build_deepspeed_config(
                self._training_config,
                self._gpu_config,
                micro_batch_size,
                precision,
            )

        return TrainingArguments(**args_kwargs)

    def _load_model_with_lora(
        self,
        precision: PrecisionMode,
    ) -> tuple[Any, Any]:
        """Load base model from HuggingFace and apply LoRA config.

        Returns (model, tokenizer) or (None, None) on failure.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(
                "loading_model",
                model=self._model_config.hf_model_id,
                precision=precision.value,
            )

            # Determine torch dtype
            is_bf16 = precision in (PrecisionMode.BF16, PrecisionMode.FP8)
            dtype = torch.bfloat16 if is_bf16 else torch.float16

            tokenizer = AutoTokenizer.from_pretrained(
                self._model_config.hf_model_id,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self._model_config.hf_model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto",
            )

            # Apply LoRA
            from peft import LoraConfig, get_peft_model

            peft_kwargs = build_peft_config(self._model_config, self._training_config)
            lora_config = LoraConfig(**peft_kwargs)
            model = get_peft_model(model, lora_config)

            # Enable gradient checkpointing if configured and model is large enough
            if (
                self._training_config.gradient_checkpointing
                and self._model_config.params_billion >= 7.0
            ):
                model.enable_input_require_grads()

            logger.info(
                "model_loaded_with_lora",
                model=self._model_config.name,
                precision=precision.value,
            )
            return model, tokenizer

        except Exception:
            logger.error("model_load_with_lora_failed", exc_info=True)
            return None, None

    def _find_max_batch_size(
        self,
        model: Any,
        tokenizer: Any,
        precision: PrecisionMode,
    ) -> int:
        """Binary search for the maximum micro_batch_size before OOM.

        Returns the largest batch size that doesn't cause OOM.
        """
        candidates = sorted(self._training_config.micro_batch_sizes)
        if not candidates:
            return 1

        low, high = 0, len(candidates) - 1
        max_feasible = candidates[0]  # At minimum, try the smallest

        while low <= high:
            mid = (low + high) // 2
            batch_size = candidates[mid]

            if self._try_batch_size(model, tokenizer, batch_size, precision):
                max_feasible = batch_size
                low = mid + 1
            else:
                high = mid - 1

        logger.info(
            "max_batch_size_found",
            max_batch_size=max_feasible,
            precision=precision.value,
            model=self._model_config.name,
        )
        return max_feasible

    def _try_batch_size(
        self,
        model: Any,
        tokenizer: Any,
        batch_size: int,
        precision: PrecisionMode,
    ) -> bool:
        """Try a training step with the given batch size. Returns True if no OOM."""
        try:
            import torch

            # Create a small dummy batch
            dummy = self._build_dummy_batch(tokenizer, batch_size)

            # Move to GPU
            device = next(model.parameters()).device
            dummy = {k: v.to(device) if hasattr(v, "to") else v for k, v in dummy.items()}

            # Forward + backward
            model.train()
            is_bf16 = precision in (PrecisionMode.BF16, PrecisionMode.FP8)
            amp_dtype = torch.bfloat16 if is_bf16 else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                outputs = model(**dummy)
                loss = outputs.loss
                if loss is not None:
                    loss.backward()

            # Clean up
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            return True

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                import torch

                torch.cuda.empty_cache()
                return False
            raise

    def _build_dummy_batch(self, tokenizer: Any, batch_size: int) -> dict[str, Any]:
        """Create a minimal dummy batch for OOM probing."""
        import torch

        seq_len = 512
        vocab_size = tokenizer.vocab_size or 32000
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _build_dummy_dataset(self, tokenizer: Any, num_samples: int) -> Any:
        """Build a dummy training dataset for benchmark measurement."""
        import torch
        from torch.utils.data import Dataset

        class DummyVLMDataset(Dataset):  # type: ignore[misc]
            def __init__(self, size: int, seq_len: int, vocab_size: int) -> None:
                self._size = size
                self._seq_len = seq_len
                self._vocab_size = vocab_size

            def __len__(self) -> int:
                return self._size

            def __getitem__(self, idx: int) -> dict[str, Any]:
                input_ids = torch.randint(0, self._vocab_size, (self._seq_len,)).tolist()
                return {
                    "input_ids": input_ids,
                    "attention_mask": [1] * self._seq_len,
                    "labels": input_ids[:],
                }

        vocab_size = tokenizer.vocab_size or 32000
        return DummyVLMDataset(size=num_samples, seq_len=512, vocab_size=vocab_size)

    def _build_data_collator(self, tokenizer: Any) -> Any:
        """Create a VLMDataCollator for the trainer."""
        from vlm_ocr_bench.training.data_collator import VLMDataCollator

        return VLMDataCollator(
            tokenizer=tokenizer,
            max_length=2048,
            max_image_resolution=self._training_config.max_image_resolution,
        )

    @staticmethod
    def _get_lora_state_dict(model: Any) -> dict[str, Any] | None:
        """Save a copy of the current LoRA adapter weights."""
        try:
            lora_state = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    lora_state[name] = param.data.clone()
            return lora_state if lora_state else None
        except Exception:
            logger.warning("lora_state_dict_save_failed", exc_info=True)
            return None

    @staticmethod
    def _reset_lora_weights(model: Any, state_dict: dict[str, Any]) -> None:
        """Reset LoRA adapter weights to initial values for fair comparison."""
        try:
            for name, param in model.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name])
            logger.debug("lora_weights_reset")
        except Exception:
            logger.warning("lora_weights_reset_failed", exc_info=True)

    @staticmethod
    def _cleanup_model(model: Any) -> None:
        """Release GPU memory held by the model."""
        try:
            import torch

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
