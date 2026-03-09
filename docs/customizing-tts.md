# Customizing TTS

Pass an instructions file with `--instructions` to shape delivery without baking project-specific defaults into the toolkit.

Example:

```powershell
tts-gpt tts --input-dir examples/tts --instructions examples/tts/instructions.md
```

Use `--keep-markdown` if your source formatting should be spoken literally instead of flattened first.
