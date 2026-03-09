from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .image import ImageWorkflowConfig, compose_founder_images, generate_images, run_image_qa
from .tts import DEFAULT_TTS_MODEL, DEFAULT_VOICE, TTSConfig, run_tts


DEFAULT_TTS_INPUT_DIR = Path("input")
DEFAULT_TTS_OUTPUT_DIR = Path("output")
DEFAULT_IMAGE_SPEC = Path("examples/image/storyboard_spec.json")
DEFAULT_IMAGE_OUTPUT_DIR = Path("artifacts/image/outputs")
DEFAULT_IMAGE_REVIEW_DIR = Path("artifacts/image/review")
DEFAULT_IMAGE_COMPOSITE_DIR = Path("artifacts/image/composites")
DEFAULT_IMAGE_PROMPT_DIR = Path("artifacts/image/prompts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts-gpt",
        description="Generate speech and storyboard imagery from local inputs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    tts_parser = subparsers.add_parser("tts", help="Run the OpenAI TTS workflow.")
    tts_parser.add_argument("--input-dir", type=Path, default=DEFAULT_TTS_INPUT_DIR)
    tts_parser.add_argument("--output-dir", type=Path, default=DEFAULT_TTS_OUTPUT_DIR)
    tts_parser.add_argument("--model", default=DEFAULT_TTS_MODEL)
    tts_parser.add_argument("--voice", default=DEFAULT_VOICE)
    tts_parser.add_argument("--instructions", type=Path, default=None)
    tts_parser.add_argument("--keep-markdown", action="store_true")
    tts_parser.set_defaults(handler=handle_tts)

    image_parser = subparsers.add_parser("image", help="Run storyboard image workflows.")
    image_subparsers = image_parser.add_subparsers(dest="image_command")

    generate_parser = image_subparsers.add_parser("generate", help="Generate image candidates.")
    add_image_common_args(generate_parser)
    generate_parser.add_argument("--only-assets", nargs="*", default=None)
    generate_parser.add_argument("--candidates", type=int, default=None)
    generate_parser.add_argument("--timeout", type=int, default=300)
    generate_parser.set_defaults(handler=handle_image_generate)

    qa_parser = image_subparsers.add_parser("qa", help="Run deterministic QA on generated images.")
    add_image_common_args(qa_parser)
    qa_parser.add_argument("--only-assets", nargs="*", default=None)
    qa_parser.set_defaults(handler=handle_image_qa)

    compose_parser = image_subparsers.add_parser("compose", help="Composite subject images onto generated plates.")
    add_image_common_args(compose_parser)
    compose_parser.add_argument("--asset-id", action="append", dest="asset_ids", default=None)
    compose_parser.add_argument("--candidate", type=int, default=1)
    compose_parser.add_argument("--image-index", type=int, default=1)
    compose_parser.set_defaults(handler=handle_image_compose)

    return parser


def add_image_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, default=DEFAULT_IMAGE_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_IMAGE_OUTPUT_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_IMAGE_REVIEW_DIR)
    parser.add_argument("--composite-dir", type=Path, default=DEFAULT_IMAGE_COMPOSITE_DIR)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_IMAGE_PROMPT_DIR)
    parser.add_argument("--model", default=None)


def handle_tts(args: argparse.Namespace) -> int:
    outputs = run_tts(
        TTSConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=args.model,
            voice=args.voice,
            instructions_path=args.instructions,
            keep_markdown=args.keep_markdown,
        )
    )
    for output in outputs:
        print(f"[ok] wrote {output}")
    return 0


def handle_image_generate(args: argparse.Namespace) -> int:
    manifest = generate_images(
        ImageWorkflowConfig(
            spec_path=args.spec,
            output_dir=args.output_dir,
            review_dir=args.review_dir,
            composite_dir=args.composite_dir,
            prompt_dir=args.prompt_dir,
            model=args.model,
            only_assets=args.only_assets,
            candidates=args.candidates,
            timeout=args.timeout,
        )
    )
    print(f"[ok] generation manifest written to {manifest}")
    return 0


def handle_image_qa(args: argparse.Namespace) -> int:
    manifest = run_image_qa(
        ImageWorkflowConfig(
            spec_path=args.spec,
            output_dir=args.output_dir,
            review_dir=args.review_dir,
            composite_dir=args.composite_dir,
            prompt_dir=args.prompt_dir,
            model=args.model,
            only_assets=args.only_assets,
        )
    )
    print(f"[ok] review manifest written to {manifest}")
    return 0


def handle_image_compose(args: argparse.Namespace) -> int:
    outputs = compose_founder_images(
        ImageWorkflowConfig(
            spec_path=args.spec,
            output_dir=args.output_dir,
            review_dir=args.review_dir,
            composite_dir=args.composite_dir,
            prompt_dir=args.prompt_dir,
            model=args.model,
        ),
        asset_ids=args.asset_ids,
        candidate=args.candidate,
        image_index=args.image_index,
    )
    for output in outputs:
        print(f"[ok] composited {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return handler(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
