"""Shared image and model utilities for both CVPD phases."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

Box = tuple[int, int, int, int]

ANSWER_PROMPT = """Answer the following question about the image concisely (just the answer, no explanation):
{question}"""


def safe_dtype(name: str) -> torch.dtype:
    """Resolve a requested precision, falling back when the device lacks support."""
    if name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name == "float16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def strip_tags(text: str, tag: str) -> Optional[str]:
    """Extract text enclosed by a simple XML-style tag."""
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else None


def parse_box(text: str) -> Optional[tuple[float, float, float, float]]:
    """Parse four box coordinates from tagged or plain model output."""
    candidate = strip_tags(text, "box") or text
    values = re.findall(r"-?\d+(?:\.\d+)?", candidate)
    if len(values) < 4:
        return None
    try:
        return tuple(float(value) for value in values[:4])  # type: ignore[return-value]
    except ValueError:
        return None


def denormalize_box(
    box: Sequence[float],
    image_width: int,
    image_height: int,
    scale: int = 1000,
) -> Box:
    """Convert a box from ``[0, scale]`` coordinates to image pixels."""
    x1, y1, x2, y2 = box
    return (
        int(x1 / scale * image_width),
        int(y1 / scale * image_height),
        int(x2 / scale * image_width),
        int(y2 / scale * image_height),
    )


def validate_box(
    box: Sequence[int],
    image_width: int,
    image_height: int,
    min_area_fraction: float,
    max_area_fraction: float,
) -> bool:
    """Check box ordering, bounds, and relative area."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return False
    area_fraction = (x2 - x1) * (y2 - y1) / max(1, image_width * image_height)
    return min_area_fraction <= area_fraction <= max_area_fraction


def resize_square(image: Image.Image, size: int) -> Image.Image:
    """Resize every visual context to the same square resolution."""
    return image.resize((size, size), Image.Resampling.BILINEAR)


def build_crop_context(image: Image.Image, box: Box, pad_ratio: float) -> Image.Image:
    """Crop a padded region and resize it to the full-view resolution."""
    x1, y1, x2, y2 = box
    width, height = image.size
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    padded_box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    return image.crop(padded_box).resize(image.size, Image.Resampling.BILINEAR)


def build_ghost_context(
    image: Image.Image,
    box: Box,
    method: str = "blur",
    sigma: float = 25.0,
) -> Image.Image:
    """Suppress evidence inside a region by blurring or mean filling it."""
    x1, y1, x2, y2 = box
    width, height = image.size
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return image.copy()

    output = image.copy()
    region = output.crop((x1, y1, x2, y2))
    if method == "blur":
        replacement = region.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif method == "mean":
        mean_color = tuple(
            np.asarray(region, dtype=np.float32).mean(axis=(0, 1)).astype(np.uint8)
        )
        replacement = Image.new("RGB", region.size, mean_color)
    else:
        raise ValueError(f"Unsupported ghosting method: {method}")
    output.paste(replacement, (x1, y1))
    return output


class VisionLanguageModel:
    """Qwen-compatible visual language model wrapper used in discovery."""

    def __init__(self, model_name: str, device: str, dtype: str) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = safe_dtype(dtype)
        print(f"[Model] {model_name} dtype={self.dtype} device={self.device}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device if self.device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.model.eval()

    def _build_inputs(
        self,
        image: Image.Image,
        prompt: str,
        completion: Optional[str] = None,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if completion is not None:
            text += completion
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    @contextlib.contextmanager
    def evaluation_mode(self) -> Iterator[None]:
        """Temporarily disable training-only layers such as dropout."""
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        inputs = self._build_inputs(image, prompt)
        generation_args: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_args.update(temperature=temperature, top_p=top_p)
        with self.evaluation_mode():
            output_ids = self.model.generate(**inputs, **generation_args)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            [generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def forward_logits(
        self,
        image: Image.Image,
        prompt: str,
        completion: str,
        *,
        requires_grad: bool,
    ) -> tuple[torch.Tensor, int]:
        """Return sequence logits and the tokenized prompt length."""
        full_inputs = self._build_inputs(image, prompt, completion)
        prompt_inputs = self._build_inputs(image, prompt)
        prompt_length = prompt_inputs["input_ids"].shape[1]
        context = contextlib.nullcontext() if requires_grad else torch.no_grad()
        with context:
            output = self.model(**full_inputs)
        return output.logits, prompt_length

    def completion_logits(
        self,
        image: Image.Image,
        prompt: str,
        completion: str,
    ) -> torch.Tensor:
        """Return logits aligned with completion tokens, shaped ``[T, V]``."""
        with self.evaluation_mode():
            logits, prompt_length = self.forward_logits(
                image,
                prompt,
                completion,
                requires_grad=False,
            )
        completion_length = logits.shape[1] - prompt_length
        if completion_length <= 0:
            vocabulary_size = getattr(self.model.config, "vocab_size", 0)
            return torch.zeros(0, vocabulary_size, device=self.device)
        start = prompt_length - 1
        return logits[0, start:start + completion_length]


class TrainableVisionLanguageModel(VisionLanguageModel):
    """LoRA student with EMA teachers and an adapter-disabled reference."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dtype: str,
        freeze_vision: bool,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_targets: Sequence[str],
        load_adapter: Optional[str],
    ) -> None:
        super().__init__(model_name, device, dtype)

        if freeze_vision:
            frozen = 0
            for name, parameter in self.model.named_parameters():
                if any(
                    part in name.lower()
                    for part in ("vision", "visual", "vit", "image_encoder")
                ):
                    parameter.requires_grad_(False)
                    frozen += 1
            print(f"[Model] Froze {frozen} vision parameters")

        if use_lora:
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model

            adapter_path = Path(load_adapter).expanduser() if load_adapter else None
            if adapter_path and not adapter_path.is_dir():
                raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
            if adapter_path and adapter_path.is_dir():
                self.model = PeftModel.from_pretrained(
                    self.model,
                    str(adapter_path),
                    is_trainable=True,
                )
                print(f"[LoRA] Loaded {adapter_path}")
            else:
                config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=list(lora_targets),
                    bias="none",
                )
                self.model = get_peft_model(self.model, config)
                print(f"[LoRA] r={lora_r} alpha={lora_alpha}")
            self.model.print_trainable_parameters()

        self.trainable_parameter_names = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if not self.trainable_parameter_names:
            self.trainable_parameter_names = [
                name
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            print("[Warning] No LoRA parameters found; EMA will track all trainable parameters")

        parameters = dict(self.model.named_parameters())
        self.ema_state = {
            name: parameters[name].detach().clone()
            for name in self.trainable_parameter_names
        }
        if load_adapter:
            self._restore_ema(Path(load_adapter).expanduser() / "ema_state.pt")
        self.model.train()

    def _restore_ema(self, path: Path) -> None:
        if not path.is_file():
            return
        try:
            saved = torch.load(path, map_location="cpu")
            restored = 0
            for name, value in saved.items():
                if name in self.ema_state:
                    self.ema_state[name].copy_(value.to(self.ema_state[name].device))
                    restored += 1
            print(f"[EMA] Restored {restored}/{len(self.ema_state)} tensors")
        except Exception as error:
            print(f"[EMA] Restore failed: {error}")

    @contextlib.contextmanager
    def use_ema_teacher(self) -> Iterator[None]:
        """Swap trainable weights to their EMA values for teacher forwards."""
        parameters = dict(self.model.named_parameters())
        backup = {
            name: parameters[name].detach().clone()
            for name in self.trainable_parameter_names
        }
        was_training = self.model.training
        self.model.eval()
        for name in self.trainable_parameter_names:
            parameters[name].data.copy_(self.ema_state[name])
        try:
            yield
        finally:
            for name in self.trainable_parameter_names:
                parameters[name].data.copy_(backup[name])
            self.model.train(was_training)

    @contextlib.contextmanager
    def use_reference_policy(self) -> Iterator[None]:
        """Evaluate the frozen base policy by disabling the active adapter."""
        was_training = self.model.training
        self.model.eval()
        try:
            if hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    yield
            else:
                yield
        finally:
            self.model.train(was_training)

    def update_ema(self, alpha: float) -> None:
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name in self.trainable_parameter_names:
                self.ema_state[name].mul_(1.0 - alpha).add_(
                    parameters[name].detach(),
                    alpha=alpha,
                )
