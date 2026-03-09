# Swapping OpenRouter Models

The image workflow reads the model from the spec and lets you override it at runtime with `--model`.

Example:

```powershell
tts-gpt image generate --spec examples/image/storyboard_spec.json --model google/gemini-3.1-flash-image-preview
```

CLI `--model` takes precedence over the spec value. This keeps example specs reproducible while still allowing experiments.
