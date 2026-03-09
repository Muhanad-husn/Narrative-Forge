# Refactor `tts-gpt` Into a Reproducible Public Toolkit

## Summary
Refactor the repo into one lightweight Python toolkit with two first-class workflows: OpenAI TTS and OpenRouter image generation. Keep the UX as a simple CLI, remove project-specific defaults from the public path, exclude generated/media artifacts and secrets, and make the repo immediately cloneable, installable, and runnable from GitHub with tiny safe sample inputs.

## Key Changes
- Restructure the code into a small package with a unified CLI, keeping two clear command groups such as `tts` and `image`.
- Preserve current capabilities, but move the current `batch_tts.py` logic and `image/scripts/*.py` logic behind stable package entrypoints instead of top-level ad hoc scripts.
- Keep TTS provider support OpenAI-only for now, but separate config loading, input preparation, synthesis, validation, and output writing so later provider expansion is straightforward.
- Keep OpenRouter as the image backend, but change image model selection from hard-coded spec default to a user-overridable CLI/config value with the current model as the documented default.
- Move the G-Lap-specific voice contract and storyboard assets out of the default product path into `examples/` so the public tool behaves neutrally unless users opt into sample/example assets.
- Replace repo-local secrets usage with `.env.example`, documented required variables, and startup validation that clearly reports missing env vars without assuming a local `.env`.
- Add reproducibility-oriented project metadata:
  - `pyproject.toml` with package metadata, console script entrypoint, Python version floor, and dependencies.
  - `LICENSE` using MIT.
  - expanded `.gitignore` that excludes `.env`, caches, generated audio/images/manifests, and local composite assets.
- Clean the repo contents before publication:
  - remove committed/generated outputs under `output/`, `image/outputs/`, `image/review/`, rendered prompt logs, and `__pycache__`.
  - remove real API keys from tracked files and ensure no secrets remain in docs, examples, or manifests.
  - replace absolute-path README links with relative docs links.
- Standardize user-facing config and I/O:
  - TTS: input directory, output directory, model, voice, optional instructions file, markdown handling mode.
  - Image: spec path, output directory, review directory, composite directory, model override, asset selection, candidate count, timeout.
- Add minimal safe examples:
  - a tiny TTS example input and optional example instructions file.
  - a tiny image example spec or reduced sample asset set that demonstrates the workflow without shipping private/generated outputs.
- Rewrite documentation around the public product rather than the original project:
  - repo README with install, env setup, quickstart for both workflows, sample commands, repo layout, and output expectations.
  - short docs for how to customize voices/instructions and how to swap OpenRouter image models.
  - note that large generated media is intentionally excluded from version control.

## Public Interfaces
- Add one console entrypoint for the package, exposing a single CLI with subcommands instead of separate loose scripts.
- Replace the current implicit TTS contract default with an optional `--instructions` style input.
- Keep image spec support, but allow `--model` to override the spec’s default model at runtime.
- Ensure all file paths stored in manifests are repo-relative or output-root-relative in a way that remains stable regardless of the current working directory.
- Keep current QA/compositing behavior conceptually intact, but make their commands part of the same CLI surface.

## Test Plan
- CLI smoke tests for help output and argument parsing for `tts`, `image generate`, `image qa`, and `image compose`.
- Unit tests for markdown stripping, token/chunk splitting, section/tone fallback behavior, transcript coverage validation, and path/config resolution.
- Unit tests for image prompt rendering, asset ordering, spec validation, model override precedence, and output path generation.
- Fixture-based tests that verify manifests and review files are written with expected schema using mocked API responses instead of live network calls.
- Packaging/install verification:
  - fresh environment install from `pyproject.toml`
  - console script runs successfully
  - docs quickstart commands match actual CLI behavior
- Publication check before GitHub push:
  - no tracked secrets
  - no generated artifacts committed
  - no absolute local paths in docs
  - clean clone can run sample commands after dependency install and env setup

## Assumptions
- Repo shape: one toolkit repo with two command groups, not two separate repos.
- UX level: simple CLI, not a heavier packaged application beyond standard Python packaging.
- TTS scope: OpenAI only in this refactor.
- Image scope: OpenRouter backend with runtime-selectable model ids.
- Current G-Lap-specific content becomes optional example material, not default product behavior.
- Public examples should be tiny and safe; generated outputs should not ship.
- License: MIT.
