from __future__ import annotations

from pathlib import Path

from ttsgpt.cli import build_parser
from ttsgpt.image import ImageWorkflowConfig, load_spec, render_prompt, resolved_project_path
from ttsgpt.tts import strip_markdown, validate_transcript_coverage


def test_accepts_copilot_spacing_difference() -> None:
    expected = "Now that the canvas is established, the optional Copilot makes sense."
    transcript = "Now that the canvas is established, the optional co-pilot makes sense."

    is_valid, similarity = validate_transcript_coverage(expected, transcript)

    assert is_valid
    assert similarity >= 0.9


def test_accepts_number_and_hyphenation_variants() -> None:
    expected = (
        "This is a real working prototype built over roughly fourteen months, "
        "and it still has limitations. Right now it supports English only and is "
        "designed to run on CPU. But the architecture can grow into more languages, "
        "SaaS deployment, and custom server or on-prem GPU setups."
    )
    transcript = (
        "This is a real working prototype, built over roughly 14 months, and it still "
        "has limitations. Right now it supports English only and is designed to run on "
        "CPU, but the architecture can grow into more languages, SaaS deployment, and "
        "custom server or on-prem GPU setups."
    )

    is_valid, similarity = validate_transcript_coverage(expected, transcript)

    assert is_valid
    assert similarity >= 0.9


def test_strip_markdown_flattens_common_syntax() -> None:
    source = "# Title\n\n- **Bold** [link](https://example.com)\n\n`code`"
    assert strip_markdown(source) == "Title\nBold link\n\ncode"


def test_cli_smoke_parses_image_generate() -> None:
    parser = build_parser()
    args = parser.parse_args(["image", "generate", "--spec", "examples/image/storyboard_spec.json"])
    assert args.command == "image"
    assert args.image_command == "generate"
    assert args.spec == Path("examples/image/storyboard_spec.json")


def test_image_model_override_takes_precedence() -> None:
    spec_path = Path("examples/image/storyboard_spec.json")
    config = ImageWorkflowConfig(spec_path=spec_path, model="custom/model")
    spec = load_spec(spec_path)
    assert config.model == "custom/model"
    assert spec["project"]["model"] != config.model


def test_render_prompt_includes_constraints() -> None:
    spec = load_spec(Path("examples/image/storyboard_spec.json"))
    asset = spec["assets"][0]
    prompt = render_prompt(spec, asset)
    assert "Hard constraints:" in prompt
    assert "Avoid:" in prompt


def test_resolved_project_path_uses_override() -> None:
    spec = load_spec(Path("examples/image/storyboard_spec.json"))
    override = Path("tmp/output")
    resolved = resolved_project_path(spec, Path("examples/image/storyboard_spec.json"), "output_dir", override)
    assert resolved == override
