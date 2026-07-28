"""Phase 1: discover counterfactual visual blind spots from raw images."""

from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import set_seed

from .core import (
    ANSWER_PROMPT,
    Box,
    VisionLanguageModel,
    build_crop_context,
    build_ghost_context,
    denormalize_box,
    parse_box,
    resize_square,
    strip_tags,
    validate_box,
)

SELF_QUESTION_PROMPT = """You are a Visual Reasoning Assistant.
Look at the image and produce ONE concrete question about a SPECIFIC SMALL DETAIL: an object's color, the count of small objects, the value of a number/label, a textual element, or a precise attribute of a small element. The question should require careful inspection of a particular region - NOT a holistic summary of the scene. The question must be answerable in a few words.

Output ONLY: <question>...</question>"""

GROUNDING_PROMPT = """You are a Visual Grounding Model.
Locate the SMALLEST region of the image that contains the information needed to answer the question below.

Question: {question}

Output ONLY the bounding box as: <box>x1,y1,x2,y2</box>
Coordinates are normalised in [0, {scale}] (top-left and bottom-right).
If no such region exists, output <box>0,0,0,0</box>."""


@dataclass
class DiscoveryConfig:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"

    data_dir: str = "./data/images"
    output_path: str = "./data/blindspots.jsonl"
    diagnostics_path: Optional[str] = None
    image_resize: int = 448
    max_images: int = -1
    seed: int = 42
    shuffle: bool = True

    max_question_tokens: int = 64
    question_temperature: float = 0.7
    question_top_p: float = 0.9
    probe_answer_tokens: int = 24
    probe_answer_temperature: float = 0.0

    use_grounding: bool = True
    max_grounding_tokens: int = 64
    coordinate_scale: int = 1000
    use_grid_3x3: bool = True
    use_grid_2x2: bool = True

    box_min_area_frac: float = 0.01
    box_max_area_frac: float = 0.50
    crop_pad_ratio: float = 0.20
    ghost_blur_sigma: float = 25.0
    ghost_method: str = "blur"

    top_k_logits: int = 100
    tau_crop_disagree: float = 0.05
    tau_ghost_agree: float = 0.05
    require_crop_more_confident: bool = True
    max_kept_per_image: int = 1
    log_every: int = 25


class ImagePool:
    """Recursively index an unlabeled image directory."""

    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(self, data_dir: str, seed: int, shuffle: bool) -> None:
        root = Path(data_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {root}")
        self.paths = sorted(
            str(path)
            for path in root.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in self.extensions
        )
        if not self.paths:
            raise RuntimeError(f"No images found under: {root}")
        if shuffle:
            random.Random(seed).shuffle(self.paths)
        print(f"[ImagePool] {len(self.paths)} images under {root}")


def grid_candidates(width: int, height: int, size: int) -> list[Box]:
    """Partition an image into a regular grid of candidate regions."""
    return [
        (
            column * width // size,
            row * height // size,
            (column + 1) * width // size,
            (row + 1) * height // size,
        )
        for row in range(size)
        for column in range(size)
    ]


def _jsd(probabilities_a: torch.Tensor, probabilities_b: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (probabilities_a + probabilities_b)
    epsilon = 1e-10
    return 0.5 * (
        (
            probabilities_a
            * (probabilities_a.add(epsilon).log() - midpoint.add(epsilon).log())
        ).sum()
        + (
            probabilities_b
            * (probabilities_b.add(epsilon).log() - midpoint.add(epsilon).log())
        ).sum()
    )


def counterfactual_jsd(
    crop_logits: torch.Tensor,
    full_logits: torch.Tensor,
    ghost_logits: torch.Tensor,
    top_k: int,
) -> tuple[float, float]:
    """Average crop/full and ghost/full JSD over completion tokens."""
    def pairwise_jsd(
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
    ) -> float:
        token_divergences: list[torch.Tensor] = []
        for row_a, row_b in zip(logits_a, logits_b):
            k = min(top_k, row_a.numel(), row_b.numel())
            if k < 1:
                raise ValueError("top_k must be positive")
            union = torch.unique(
                torch.cat(
                    [
                        torch.topk(row_a, k).indices,
                        torch.topk(row_b, k).indices,
                    ]
                )
            )
            probability_a = F.softmax(row_a[union].float(), dim=-1)
            probability_b = F.softmax(row_b[union].float(), dim=-1)
            token_divergences.append(_jsd(probability_a, probability_b))
        return torch.stack(token_divergences).mean().item()

    return (
        pairwise_jsd(crop_logits, full_logits),
        pairwise_jsd(ghost_logits, full_logits),
    )


def entropy(logits: torch.Tensor) -> float:
    """Compute entropy over a single next-token distribution."""
    probabilities = F.softmax(logits.float(), dim=-1)
    epsilon = 1e-10
    return float(
        -(probabilities * probabilities.add(epsilon).log()).sum().item()
    )


class BlindSpotDiscoverer:
    """Generate candidates and apply CVPD's three-gate criterion."""

    def __init__(self, config: DiscoveryConfig) -> None:
        self.config = config
        self.model = VisionLanguageModel(
            config.model_name,
            config.device,
            config.dtype,
        )

    def _candidates(
        self,
        image: Image.Image,
        question: str,
    ) -> list[tuple[str, Box]]:
        config = self.config
        width, height = image.size
        candidates: list[tuple[str, Box]] = []
        if config.use_grounding:
            try:
                output = self.model.generate(
                    image,
                    GROUNDING_PROMPT.format(
                        question=question,
                        scale=config.coordinate_scale,
                    ),
                    max_new_tokens=config.max_grounding_tokens,
                    temperature=0.0,
                )
                normalized_box = parse_box(output)
                if normalized_box and normalized_box != (0.0, 0.0, 0.0, 0.0):
                    candidates.append(
                        (
                            "grounding",
                            denormalize_box(
                                normalized_box,
                                width,
                                height,
                                config.coordinate_scale,
                            ),
                        )
                    )
            except Exception:
                pass
        if config.use_grid_3x3:
            candidates.extend(("grid3", box) for box in grid_candidates(width, height, 3))
        if config.use_grid_2x2:
            candidates.extend(("grid2", box) for box in grid_candidates(width, height, 2))
        return [
            (source, box)
            for source, box in candidates
            if validate_box(
                box,
                width,
                height,
                config.box_min_area_frac,
                config.box_max_area_frac,
            )
        ]

    def discover(
        self,
        original_image: Image.Image,
    ) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
        """Return selected records, status, and all candidate diagnostics."""
        config = self.config
        image = resize_square(original_image, config.image_resize)
        width, height = image.size

        question_output = self.model.generate(
            image,
            SELF_QUESTION_PROMPT,
            max_new_tokens=config.max_question_tokens,
            temperature=config.question_temperature,
            top_p=config.question_top_p,
        )
        question = strip_tags(question_output, "question")
        if not question or len(question) < 5:
            return [], "parse_fail_question", []

        answer_prompt = ANSWER_PROMPT.format(question=question)
        full_answer = self.model.generate(
            image,
            answer_prompt,
            max_new_tokens=config.probe_answer_tokens,
            temperature=config.probe_answer_temperature,
        ).strip()
        if not full_answer:
            return [], "empty_full_answer", []

        try:
            full_logits = self.model.completion_logits(image, answer_prompt, full_answer)
        except Exception as error:
            return [], f"full_fwd_err:{error}", []
        if full_logits.shape[0] < 1:
            return [], "empty_full_logits", []

        candidates = self._candidates(image, question)
        if not candidates:
            return [], "no_valid_candidates", []

        diagnostics: list[dict[str, Any]] = []
        passing: list[dict[str, Any]] = []
        full_entropy = entropy(full_logits[0])
        for source, box in candidates:
            crop = build_crop_context(image, box, config.crop_pad_ratio)
            ghost = build_ghost_context(
                image,
                box,
                config.ghost_method,
                config.ghost_blur_sigma,
            )
            try:
                crop_logits = self.model.completion_logits(
                    crop,
                    answer_prompt,
                    full_answer,
                )
                ghost_logits = self.model.completion_logits(
                    ghost,
                    answer_prompt,
                    full_answer,
                )
            except Exception:
                continue

            token_count = min(
                full_logits.shape[0],
                crop_logits.shape[0],
                ghost_logits.shape[0],
            )
            if token_count < 1:
                continue
            crop_full_jsd, ghost_full_jsd = counterfactual_jsd(
                crop_logits[:token_count],
                full_logits[:token_count],
                ghost_logits[:token_count],
                config.top_k_logits,
            )
            crop_entropy = entropy(crop_logits[0])
            gate_crop = crop_full_jsd >= config.tau_crop_disagree
            gate_ghost = ghost_full_jsd <= config.tau_ghost_agree
            gate_confidence = crop_entropy < full_entropy
            diagnostic = {
                "name": source,
                "box": list(box),
                "jsd_crop_full": crop_full_jsd,
                "jsd_ghost_full": ghost_full_jsd,
                "h_full": full_entropy,
                "h_crop": crop_entropy,
                "gate_crop_disagree": gate_crop,
                "gate_ghost_agree": gate_ghost,
                "gate_confidence": gate_confidence,
            }
            diagnostics.append(diagnostic)

            if not gate_crop or not gate_ghost:
                continue
            if config.require_crop_more_confident and not gate_confidence:
                continue
            score = crop_full_jsd - ghost_full_jsd + full_entropy - crop_entropy
            passing.append({**diagnostic, "score": score})

        for diagnostic in diagnostics:
            diagnostic["question"] = question
        if not passing:
            return [], f"no_blindspot_gate (cand={len(diagnostics)})", diagnostics

        passing.sort(key=lambda candidate: candidate["score"], reverse=True)
        records = [
            {
                "question": question,
                "full_answer": full_answer,
                "region": candidate["box"],
                "candidate_name": candidate["name"],
                "score": candidate["score"],
                "jsd_crop_full": candidate["jsd_crop_full"],
                "jsd_ghost_full": candidate["jsd_ghost_full"],
                "h_full": candidate["h_full"],
                "h_crop": candidate["h_crop"],
                "image_w": width,
                "image_h": height,
                "n_candidates_eval": len(diagnostics),
            }
            for candidate in passing[: max(1, config.max_kept_per_image)]
        ]
        return records, "ok", diagnostics


def _processed_images(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    processed = set()
    with output_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                processed.add(json.loads(line)["image_path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def run_discovery(config: DiscoveryConfig) -> None:
    """Run resumable offline blind-spot discovery."""
    set_seed(config.seed)
    random.seed(config.seed)

    output_path = Path(config.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path = (
        Path(config.diagnostics_path).expanduser().resolve()
        if config.diagnostics_path
        else None
    )
    if diagnostics_path:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    pool = ImagePool(config.data_dir, config.seed, config.shuffle)
    paths = pool.paths if config.max_images <= 0 else pool.paths[: config.max_images]
    processed = _processed_images(output_path)
    if processed:
        print(f"[Resume] {len(processed)} images already recorded in {output_path}")

    discoverer = BlindSpotDiscoverer(config)
    started = time.monotonic()
    completed = 0
    kept = 0
    skip_reasons: dict[str, int] = {}

    diagnostics_stream = (
        diagnostics_path.open("a", encoding="utf-8") if diagnostics_path else None
    )
    try:
        with output_path.open("a", encoding="utf-8") as output_stream:
            for image_path in paths:
                if image_path in processed:
                    continue
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as error:
                    output_stream.write(
                        json.dumps(
                            {"image_path": image_path, "skipped": f"open_fail:{error}"}
                        )
                        + "\n"
                    )
                    output_stream.flush()
                    continue

                try:
                    records, status, diagnostics = discoverer.discover(image)
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    records, status, diagnostics = [], "oom", []
                except Exception as error:
                    records, status, diagnostics = [], f"error:{error}", []

                if records:
                    for record in records:
                        record["image_path"] = image_path
                        output_stream.write(json.dumps(record) + "\n")
                    kept += len(records)
                else:
                    output_stream.write(
                        json.dumps({"image_path": image_path, "skipped": status}) + "\n"
                    )
                    reason = status.split()[0]
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                output_stream.flush()

                if diagnostics_stream:
                    for diagnostic in diagnostics:
                        diagnostics_stream.write(
                            json.dumps(
                                {
                                    **diagnostic,
                                    "image_path": image_path,
                                }
                            )
                            + "\n"
                        )
                    diagnostics_stream.flush()

                completed += 1
                if config.log_every > 0 and completed % config.log_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    elapsed = time.monotonic() - started
                    rate = completed / max(elapsed, 1e-9)
                    remaining = (len(paths) - len(processed) - completed) / max(
                        rate,
                        1e-9,
                    )
                    print(
                        f"[{completed}/{len(paths) - len(processed)}] "
                        f"kept={kept} ({100 * kept / completed:.1f}%) "
                        f"{rate:.2f} img/s eta={remaining / 60:.0f}min "
                        f"skips={skip_reasons}"
                    )
    finally:
        if diagnostics_stream:
            diagnostics_stream.close()

    print(
        f"Done: processed={completed} kept={kept} "
        f"({100 * kept / max(1, completed):.1f}%) output={output_path}"
    )
    print(f"Skip reasons: {skip_reasons}")
