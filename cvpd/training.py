"""Phase 2: contrastive on-policy self-distillation over curated blind spots."""

from __future__ import annotations

import gc
import json
import random
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import set_seed

from .core import (
    ANSWER_PROMPT,
    TrainableVisionLanguageModel,
    build_crop_context,
    build_ghost_context,
    resize_square,
)
from .objective import compute_loss

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"

    blindspots_jsonl: str = "./data/blindspots.jsonl"
    image_resize: int = 448
    output_dir: str = "./runs"
    run_name: str = "cvpd_8b"
    checkpoint_root: str = "./checkpoints"
    log_dir: str = "./logs"

    total_steps: int = -1
    save_every: int = 200
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum: int = 4
    seed: int = 42

    max_answer_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    use_cached_answer: bool = True

    top_k_logits: int = 100
    lambda_rank: float = 0.5
    margin: float = 0.10
    beta_ref: float = 1e-3
    kl_target: float = 0.030
    kl_adapt_rate: float = 0.10
    ema_alpha: float = 0.05

    crop_pad_ratio: float = 0.20
    ghost_blur_sigma: float = 25.0
    ghost_method: str = "blur"

    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS

    freeze_vision: bool = True
    clear_cache_every: int = 10
    load_adapter: Optional[str] = None
    start_step: int = 0
    max_checkpoints: int = 10


class BlindSpotPool:
    """Load passing records from the discovery JSONL."""

    required_keys = {"image_path", "question", "region", "image_w", "image_h"}

    def __init__(self, jsonl_path: str, seed: int) -> None:
        path = Path(jsonl_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Blind-spot JSONL not found: {path}")

        entries: list[dict[str, Any]] = []
        skipped = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "skipped" in entry:
                    skipped += 1
                    continue
                if not self.required_keys.issubset(entry):
                    continue
                if not Path(entry["image_path"]).is_file():
                    continue
                entries.append(entry)
        if not entries:
            raise RuntimeError(f"No usable entries in {path}")
        random.Random(seed).shuffle(entries)
        self.entries = entries
        print(
            f"[BlindSpotPool] {len(entries)} usable entries "
            f"(skipped {skipped}) from {path}"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self, step: int) -> dict[str, Any]:
        return self.entries[(max(1, step) - 1) % len(self.entries)]


def _assistant_logits(
    logits: torch.Tensor,
    prompt_length: int,
    completion_length: int,
) -> torch.Tensor:
    start = max(0, prompt_length - 1)
    end = min(logits.shape[1], start + completion_length)
    return logits[0, start:end]


class CVPDTrainer:
    """Train the full-image student from crop and ghost EMA teachers."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        set_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        self.run_dir = Path(config.output_dir).expanduser() / config.run_name
        self.checkpoint_dir = (
            Path(config.checkpoint_root).expanduser() / config.run_name
        )
        self.log_path = (
            Path(config.log_dir).expanduser() / f"{config.run_name}.jsonl"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.pool = BlindSpotPool(config.blindspots_jsonl, config.seed)
        self.total_steps = (
            len(self.pool) if config.total_steps <= 0 else config.total_steps
        )
        self.model = TrainableVisionLanguageModel(
            model_name=config.model_name,
            device=config.device,
            dtype=config.dtype,
            freeze_vision=config.freeze_vision,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_targets=config.lora_targets,
            load_adapter=config.load_adapter,
        )
        trainable = [
            parameter
            for parameter in self.model.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("The model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.beta_ref = config.beta_ref
        self.skip_counts = {
            "empty_answer": 0,
            "forward_error": 0,
            "empty_tokens": 0,
        }
        self.trained_samples = 0

    def _write(self, metrics: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")

    def _skip(self, step: int, reason: str, image_path: str) -> dict[str, Any]:
        metrics = {
            "step": step,
            "image_path": image_path,
            "skipped": reason,
            "skip_counts": dict(self.skip_counts),
            "trained_samples": self.trained_samples,
        }
        self._write(metrics)
        print(f"[step {step}] SKIP {reason}")
        return metrics

    def _optimizer_step(self) -> None:
        parameters = [
            parameter
            for parameter in self.model.model.parameters()
            if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.update_ema(self.config.ema_alpha)

    def train_step(self, step: int) -> dict[str, Any]:
        config = self.config
        entry = self.pool.sample(step)
        image_path = entry["image_path"]
        try:
            original = Image.open(image_path).convert("RGB")
        except Exception as error:
            self.skip_counts["forward_error"] += 1
            return self._skip(step, f"open_fail:{error}", image_path)
        image = resize_square(original, config.image_resize)

        original_width = max(1, int(entry["image_w"]))
        original_height = max(1, int(entry["image_h"]))
        width, height = image.size
        x1, y1, x2, y2 = entry["region"]
        region = (
            int(x1 * width / original_width),
            int(y1 * height / original_height),
            int(x2 * width / original_width),
            int(y2 * height / original_height),
        )
        question = entry["question"]
        prompt = ANSWER_PROMPT.format(question=question)

        cached_answer = (entry.get("full_answer") or "").strip()
        if config.use_cached_answer and cached_answer:
            rollout = cached_answer
            rollout_source = "cached"
        else:
            rollout = self.model.generate(
                image,
                prompt,
                max_new_tokens=config.max_answer_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            ).strip()
            rollout_source = "student"
        if not rollout:
            self.skip_counts["empty_answer"] += 1
            return self._skip(step, "empty_answer", image_path)
        if self.model.tokenizer is None:
            raise RuntimeError("The model processor did not expose a tokenizer")
        completion_length = len(
            self.model.tokenizer(rollout, add_special_tokens=False)["input_ids"]
        )
        if completion_length < 1:
            self.skip_counts["empty_tokens"] += 1
            return self._skip(step, "empty_tokens", image_path)

        positive_context = build_crop_context(
            image,
            region,
            config.crop_pad_ratio,
        )
        negative_context = build_ghost_context(
            image,
            region,
            config.ghost_method,
            config.ghost_blur_sigma,
        )
        try:
            with self.model.use_ema_teacher():
                positive_logits, positive_prompt_length = self.model.forward_logits(
                    positive_context,
                    prompt,
                    rollout,
                    requires_grad=False,
                )
                negative_logits, negative_prompt_length = self.model.forward_logits(
                    negative_context,
                    prompt,
                    rollout,
                    requires_grad=False,
                )
            with self.model.use_reference_policy():
                reference_logits, reference_prompt_length = self.model.forward_logits(
                    image,
                    prompt,
                    rollout,
                    requires_grad=False,
                )
            student_logits, student_prompt_length = self.model.forward_logits(
                image,
                prompt,
                rollout,
                requires_grad=True,
            )
        except torch.cuda.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.skip_counts["forward_error"] += 1
            return self._skip(step, f"oom:{error}", image_path)
        except Exception as error:
            self.skip_counts["forward_error"] += 1
            return self._skip(step, f"forward_error:{error}", image_path)

        streams = [
            _assistant_logits(logits, prompt_length, completion_length)
            for logits, prompt_length in (
                (student_logits, student_prompt_length),
                (positive_logits, positive_prompt_length),
                (negative_logits, negative_prompt_length),
                (reference_logits, reference_prompt_length),
            )
        ]
        token_count = min(stream.shape[0] for stream in streams)
        if token_count < 1:
            self.skip_counts["empty_tokens"] += 1
            return self._skip(step, "empty_tokens", image_path)
        student, positive, negative, reference = [
            stream[:token_count].float() for stream in streams
        ]

        losses = compute_loss(
            student,
            positive,
            negative,
            reference,
            config,
        )
        total_loss = (
            losses["loss_pos"]
            + config.lambda_rank * losses["loss_rank"]
            + self.beta_ref * losses["kl_ref"]
        )
        total_loss_value = float(total_loss.item())
        (total_loss / max(1, config.grad_accum)).backward()
        self.trained_samples += 1

        kl_value = float(losses["kl_ref"].item())
        adjustment = (
            1.0 + config.kl_adapt_rate
            if kl_value > config.kl_target
            else 1.0 - config.kl_adapt_rate
        )
        self.beta_ref = min(5.0, max(1e-6, self.beta_ref * adjustment))

        if self.trained_samples % max(1, config.grad_accum) == 0:
            self._optimizer_step()
        if (
            config.clear_cache_every > 0
            and step % config.clear_cache_every == 0
        ):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        metrics = {
            "step": step,
            "image_path": image_path,
            "question": question[:120],
            "candidate_name": entry.get("candidate_name", "?"),
            "rollout_source": rollout_source,
            "region": list(region),
            "n_rollout_tokens": completion_length,
            "n_trained_tokens": int(losses["n_tokens"].item()),
            "loss_total": total_loss_value,
            "loss_pos": float(losses["loss_pos"].item()),
            "loss_rank": float(losses["loss_rank"].item()),
            "kl_ref": kl_value,
            "d_pos": float(losses["d_pos"].item()),
            "d_neg": float(losses["d_neg"].item()),
            "d_pos_neg": float(losses["d_pos_neg"].item()),
            "beta_ref": self.beta_ref,
            "trained_samples": self.trained_samples,
            "skip_counts": dict(self.skip_counts),
        }
        self._write(metrics)
        print(
            f"[step {step}] L={metrics['loss_total']:.4f} "
            f"L_pos={metrics['loss_pos']:.4f} "
            f"L_rank={metrics['loss_rank']:.4f} KL={kl_value:.4f} "
            f"D_pn={metrics['d_pos_neg']:.4f} "
            f"n_tok={metrics['n_trained_tokens']} beta={self.beta_ref:.3f}"
        )
        return metrics

    def save_checkpoint(self, step: int) -> None:
        destination = self.checkpoint_dir / f"step_{step:06d}"
        temporary = destination.with_name(f"{destination.name}.tmp")
        if temporary.is_dir():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            self.model.model.save_pretrained(temporary)
            if self.model.tokenizer is not None:
                self.model.tokenizer.save_pretrained(temporary)
            self.model.processor.save_pretrained(temporary)
            torch.save(
                {
                    name: value.detach().cpu()
                    for name, value in self.model.ema_state.items()
                },
                temporary / "ema_state.pt",
            )
            with (temporary / "checkpoint_meta.json").open(
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    {
                        "step": step,
                        "model_name": self.config.model_name,
                        "beta_ref": self.beta_ref,
                        "trained_samples": self.trained_samples,
                        "time": int(time.time()),
                    },
                    stream,
                    indent=2,
                )
            (temporary / "SAVE_OK").write_text("ok\n", encoding="utf-8")
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(f"[Checkpoint] Saved {destination}")
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            match = re.fullmatch(r"step_(\d+)", path.name)
            if match and (path / "SAVE_OK").is_file():
                checkpoints.append((int(match.group(1)), path))
        checkpoints.sort()
        excess = len(checkpoints) - self.config.max_checkpoints
        for _, path in checkpoints[: max(0, excess)]:
            shutil.rmtree(path)
            print(f"[Checkpoint] Pruned {path}")

    def train(self) -> None:
        config = self.config
        print("=" * 72)
        print(f"CVPD training: {config.run_name}")
        print(f"  data        = {config.blindspots_jsonl}")
        print(f"  steps       = {self.total_steps}")
        print(f"  lr/accum    = {config.lr} / {config.grad_accum}")
        print(
            f"  top_k       = {config.top_k_logits} "
            f"margin={config.margin} rank_weight={config.lambda_rank}"
        )
        print(
            f"  kl_target   = {config.kl_target} beta0={config.beta_ref}"
        )
        print(f"  ema_alpha   = {config.ema_alpha}")
        print(f"  lora        = r{config.lora_r}/alpha{config.lora_alpha}")
        print("=" * 72)

        for step in range(config.start_step + 1, self.total_steps + 1):
            try:
                self.train_step(step)
            except torch.cuda.OutOfMemoryError as error:
                print(f"[step {step}] OOM: {error}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                self.optimizer.zero_grad(set_to_none=True)
                continue
            except Exception as error:
                print(f"[step {step}] ERROR: {error}")
                traceback.print_exc()
                self.optimizer.zero_grad(set_to_none=True)
                continue
            if config.save_every > 0 and step % config.save_every == 0:
                self.save_checkpoint(step)

        if (
            self.trained_samples % max(1, config.grad_accum) != 0
            and self.trained_samples > 0
        ):
            pending = self.trained_samples % max(1, config.grad_accum)
            correction = max(1, config.grad_accum) / pending
            for parameter in self.model.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            self._optimizer_step()
        if (
            config.save_every <= 0
            or self.total_steps % config.save_every != 0
        ):
            self.save_checkpoint(self.total_steps)
        print(
            f"Done: trained={self.trained_samples} "
            f"skips={self.skip_counts} beta_ref={self.beta_ref}"
        )
