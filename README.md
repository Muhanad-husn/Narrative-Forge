# tts-gpt

`tts-gpt` is a small Python toolkit with two CLI workflows:

- OpenAI text-to-speech for Markdown inputs
- OpenRouter image generation for storyboard-style specs

Generated media is intentionally excluded from version control. The repository ships tiny safe examples instead of project-specific outputs.

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
Copy-Item .env.example .env
```

Set the environment variables you need in `.env` or your shell. `OPENAI_API_KEY` is required for `tts`. `OPENROUTER_API_KEY` is required for `image generate`.

## Quickstart

TTS example:

```powershell
tts-gpt tts --input-dir examples/tts --instructions examples/tts/instructions.md --output-dir artifacts/audio
```

Image example:

```powershell
tts-gpt image generate --spec examples/image/storyboard_spec.json --only-assets IMG-EX-01
tts-gpt image qa --spec examples/image/storyboard_spec.json --only-assets IMG-EX-01
```

## CLI

```powershell
tts-gpt tts --help
tts-gpt image generate --help
tts-gpt image qa --help
tts-gpt image compose --help
```

## Layout

- `src/ttsgpt/`: package code and unified CLI
- `examples/`: tiny sample inputs and image specs
- `docs/`: focused notes for TTS customization and image model overrides
- `artifacts/`: default generated output root

## Notes

- TTS supports `--input-dir`, `--output-dir`, `--model`, `--voice`, `--instructions`, and `--keep-markdown`.
- Image commands support `--spec`, `--output-dir`, `--review-dir`, `--composite-dir`, `--prompt-dir`, `--model`, and asset selection flags.
- CLI `--model` overrides the image model declared inside the spec.
- Paths written to manifests are stable relative paths under the configured output roots.
