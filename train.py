"""Command-line entry point for CVPD contrastive self-distillation."""

import argparse

from cvpd.training import DEFAULT_LORA_TARGETS, CVPDTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CVPD on a curated blind-spot JSONL."
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blindspots_jsonl", default="./data/blindspots.jsonl")
    parser.add_argument("--image_resize", type=int, default=448)
    parser.add_argument("--output_dir", default="./runs")
    parser.add_argument("--run_name", default="cvpd_8b")
    parser.add_argument("--checkpoint_root", default="./checkpoints")
    parser.add_argument("--log_dir", default="./logs")

    parser.add_argument(
        "--total_steps",
        type=int,
        default=-1,
        help="Training steps; values <= 0 run one pass over the curated pool.",
    )
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_answer_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--use_cached_answer",
        dest="use_cached_answer",
        action="store_true",
    )
    parser.add_argument(
        "--no_cached_answer",
        dest="use_cached_answer",
        action="store_false",
        help="Generate a fresh student rollout instead of using the discovery probe.",
    )
    parser.set_defaults(use_cached_answer=True)

    parser.add_argument("--top_k_logits", type=int, default=100)
    parser.add_argument("--lambda_rank", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--beta_ref", type=float, default=1e-3)
    parser.add_argument("--kl_target", type=float, default=0.030)
    parser.add_argument("--kl_adapt_rate", type=float, default=0.10)
    parser.add_argument("--ema_alpha", type=float, default=0.05)

    parser.add_argument("--crop_pad_ratio", type=float, default=0.20)
    parser.add_argument("--ghost_blur_sigma", type=float, default=25.0)
    parser.add_argument(
        "--ghost_method",
        choices=("blur", "mean"),
        default="blur",
    )

    parser.add_argument("--use_lora", dest="use_lora", action="store_true")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        default=",".join(DEFAULT_LORA_TARGETS),
    )
    parser.add_argument(
        "--freeze_vision",
        dest="freeze_vision",
        action="store_true",
    )
    parser.add_argument(
        "--no_freeze_vision",
        dest="freeze_vision",
        action="store_false",
    )
    parser.set_defaults(freeze_vision=True)

    parser.add_argument("--clear_cache_every", type=int, default=10)
    parser.add_argument("--load_adapter")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_checkpoints", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
        blindspots_jsonl=args.blindspots_jsonl,
        image_resize=args.image_resize,
        output_dir=args.output_dir,
        run_name=args.run_name,
        checkpoint_root=args.checkpoint_root,
        log_dir=args.log_dir,
        total_steps=args.total_steps,
        save_every=args.save_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        grad_accum=args.grad_accum,
        seed=args.seed,
        max_answer_tokens=args.max_answer_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_cached_answer=args.use_cached_answer,
        top_k_logits=args.top_k_logits,
        lambda_rank=args.lambda_rank,
        margin=args.margin,
        beta_ref=args.beta_ref,
        kl_target=args.kl_target,
        kl_adapt_rate=args.kl_adapt_rate,
        ema_alpha=args.ema_alpha,
        crop_pad_ratio=args.crop_pad_ratio,
        ghost_blur_sigma=args.ghost_blur_sigma,
        ghost_method=args.ghost_method,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=tuple(
            target.strip()
            for target in args.lora_targets.split(",")
            if target.strip()
        ),
        freeze_vision=args.freeze_vision,
        clear_cache_every=args.clear_cache_every,
        load_adapter=args.load_adapter,
        start_step=args.start_step,
        max_checkpoints=args.max_checkpoints,
    )
    CVPDTrainer(config).train()


if __name__ == "__main__":
    main()
