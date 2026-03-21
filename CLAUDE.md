# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

tts-gpt is a Python CLI toolkit with two workflows:
- **TTS**: Converts Markdown files to normalized WAV audio via OpenAI's TTS API, with transcript validation and retry logic.
- **Image**: Generates storyboard imagery via OpenRouter (Gemini backend), with QA checks and founder-photo compositing.

Python ≥3.11, MIT licensed.

## Commands

```bash
# Install (editable + dev deps)
pip install -e .[dev]

# Run tests
pytest tests/

# TTS
tts-gpt tts --input-dir examples/tts --instructions examples/tts/instructions.md --output-dir artifacts/audio

# Image generation / QA / compositing
tts-gpt image generate --spec examples/image/storyboard_spec.json --only-assets IMG-EX-01
tts-gpt image qa --spec examples/image/storyboard_spec.json
tts-gpt image compose --spec examples/image/storyboard_spec.json --asset-id IMG-EX-01
```

## Environment Variables

Requires `OPENAI_API_KEY` and `OPENROUTER_API_KEY` in environment or `.env` file. See `.env.example`.

## Architecture

### Entry point
`src/ttsgpt/cli.py` — single CLI using argparse subparsers (`tts`, `image generate`, `image qa`, `image compose`). Each subcommand has a handler that builds a frozen dataclass config and calls the corresponding workflow function.

### TTS workflow (`src/ttsgpt/tts.py`)
1. Discover `*.md` files in input dir
2. Strip markdown formatting (unless `--keep-markdown`)
3. Chunk text respecting a 2000-token budget (paragraph → sentence → word fallback)
4. Call OpenAI TTS API per chunk, concatenate audio
5. Post-process: mono conversion, resample to 48kHz, LUFS loudness normalization (-18 LUFS), peak limiting (-3 dBFS), lead-in/tail silence
6. Save as 24-bit PCM WAV
7. Validate by transcribing output with `gpt-4o-mini-transcribe` and checking similarity (≥0.9 threshold); retry once on failure

### Image workflow (`src/ttsgpt/image.py`)
- **generate**: Loads a JSON spec file defining assets, global style, and constraints. Renders prompts (style preamble + body + hard constraints + negative prompt) and calls OpenRouter API. Saves PNGs + per-candidate metadata + run manifest.
- **qa**: Validates each generated image (file existence, exact dimensions, aspect ratio match). Writes review JSONs.
- **compose**: Alpha-composites founder photos onto background plates using bounding box coordinates from the spec.

### Spec format
The image spec JSON (`examples/image/storyboard_spec.json`) defines project config (model, resolution, aspect ratio), global style, color rules, and an ordered asset list with per-asset prompts, constraints, and optional compositing zones.

### Key design patterns
- **Frozen dataclasses** for all config objects (immutable)
- **`--model` CLI flag** overrides the spec's default model
- **Relative paths** in manifests for portability
- **All generated artifacts** (audio, images, manifests) are gitignored under `artifacts/`, `output/`, `input/`
- **`batch_tts.py`** at repo root is a backwards-compatibility wrapper only
