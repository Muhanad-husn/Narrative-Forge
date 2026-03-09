# Image Workflow

The supported public interface is the package CLI:

```powershell
tts-gpt image generate --help
tts-gpt image qa --help
tts-gpt image compose --help
```

`image/scripts/*.py` remain as thin compatibility wrappers. For a tiny safe sample spec, use [`examples/image/storyboard_spec.json`](../examples/image/storyboard_spec.json).

Legacy project-specific assets can still live under `image/`, but generated outputs and prompt logs should stay untracked.
