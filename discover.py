"""Command-line entry point for CVPD blind-spot discovery."""

import argparse

from cvpd.discovery import DiscoveryConfig, run_discovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover counterfactual visual blind spots from raw images."
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="./data/images")
    parser.add_argument("--output_path", default="./data/blindspots.jsonl")
    parser.add_argument("--diagnostics_path")
    parser.add_argument("--image_resize", type=int, default=448)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true")

    parser.add_argument("--max_question_tokens", type=int, default=64)
    parser.add_argument("--question_temperature", type=float, default=0.7)
    parser.add_argument("--question_top_p", type=float, default=0.9)
    parser.add_argument("--probe_answer_tokens", type=int, default=24)
    parser.add_argument("--probe_answer_temperature", type=float, default=0.0)

    parser.add_argument("--no_grounding", action="store_true")
    parser.add_argument("--max_grounding_tokens", type=int, default=64)
    parser.add_argument("--coordinate_scale", type=int, default=1000)
    parser.add_argument("--no_grid_3x3", action="store_true")
    parser.add_argument("--no_grid_2x2", action="store_true")

    parser.add_argument("--box_min_area_frac", type=float, default=0.01)
    parser.add_argument("--box_max_area_frac", type=float, default=0.50)
    parser.add_argument("--crop_pad_ratio", type=float, default=0.20)
    parser.add_argument("--ghost_blur_sigma", type=float, default=25.0)
    parser.add_argument(
        "--ghost_method",
        choices=("blur", "mean"),
        default="blur",
    )

    parser.add_argument("--top_k_logits", type=int, default=100)
    parser.add_argument("--tau_crop_disagree", type=float, default=0.05)
    parser.add_argument("--tau_ghost_agree", type=float, default=0.05)
    parser.add_argument("--no_confidence_check", action="store_true")
    parser.add_argument("--max_kept_per_image", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DiscoveryConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
        data_dir=args.data_dir,
        output_path=args.output_path,
        diagnostics_path=args.diagnostics_path,
        image_resize=args.image_resize,
        max_images=args.max_images,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        max_question_tokens=args.max_question_tokens,
        question_temperature=args.question_temperature,
        question_top_p=args.question_top_p,
        probe_answer_tokens=args.probe_answer_tokens,
        probe_answer_temperature=args.probe_answer_temperature,
        use_grounding=not args.no_grounding,
        max_grounding_tokens=args.max_grounding_tokens,
        coordinate_scale=args.coordinate_scale,
        use_grid_3x3=not args.no_grid_3x3,
        use_grid_2x2=not args.no_grid_2x2,
        box_min_area_frac=args.box_min_area_frac,
        box_max_area_frac=args.box_max_area_frac,
        crop_pad_ratio=args.crop_pad_ratio,
        ghost_blur_sigma=args.ghost_blur_sigma,
        ghost_method=args.ghost_method,
        top_k_logits=args.top_k_logits,
        tau_crop_disagree=args.tau_crop_disagree,
        tau_ghost_agree=args.tau_ghost_agree,
        require_crop_more_confident=not args.no_confidence_check,
        max_kept_per_image=args.max_kept_per_image,
        log_every=args.log_every,
    )
    run_discovery(config)


if __name__ == "__main__":
    main()
