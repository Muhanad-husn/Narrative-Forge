from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from PIL import Image, ImageOps


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_IMAGE_MODEL = "google/gemini-3.1-flash-image-preview"
DEFAULT_REFERER = "https://localhost/tts-gpt"
DEFAULT_TITLE = "tts-gpt"


@dataclass(frozen=True)
class ImageWorkflowConfig:
    spec_path: Path
    output_dir: Path | None = None
    review_dir: Path | None = None
    composite_dir: Path | None = None
    prompt_dir: Path | None = None
    model: str | None = None
    only_assets: list[str] | None = None
    candidates: int | None = None
    timeout: int = 300


@dataclass
class SavedImage:
    path: Path
    mime: str
    original_size: tuple[int, int]
    final_size: tuple[int, int]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_spec(spec_path: Path) -> dict[str, Any]:
    return read_json(spec_path)


def get_asset_map(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {asset["asset_id"]: asset for asset in spec["assets"]}


def parse_target_resolution(spec: dict[str, Any]) -> tuple[int, int]:
    width, height = spec["project"]["target_resolution"].split("x", maxsplit=1)
    return int(width), int(height)


def normalize_asset_order(spec: dict[str, Any], only_assets: list[str] | None = None) -> list[dict[str, Any]]:
    asset_map = get_asset_map(spec)
    order = only_assets or spec["batch_order"]
    missing = [asset_id for asset_id in order if asset_id not in asset_map]
    if missing:
        raise ValueError(f"Unknown asset ids in selection: {', '.join(missing)}")
    return [asset_map[asset_id] for asset_id in order]


def validate_spec(spec: dict[str, Any]) -> None:
    required_top_level = {"project", "global_style", "colors", "rules", "assets", "batch_order"}
    missing = sorted(required_top_level - spec.keys())
    if missing:
        raise ValueError(f"Spec missing required top-level keys: {', '.join(missing)}")
    asset_ids = [asset["asset_id"] for asset in spec["assets"]]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Spec contains duplicate asset_id values.")
    batch_order = spec["batch_order"]
    unknown = [asset_id for asset_id in batch_order if asset_id not in asset_ids]
    if unknown:
        raise ValueError(f"batch_order references unknown asset ids: {', '.join(unknown)}")
    if len(batch_order) != len(set(batch_order)):
        raise ValueError("batch_order contains duplicate asset ids.")
    if set(batch_order) != set(asset_ids):
        raise ValueError("batch_order must include every asset exactly once.")


def resolved_project_path(spec: dict[str, Any], spec_path: Path, key: str, override: Path | None = None) -> Path:
    if override is not None:
        return override
    return (spec_path.parent / spec["project"][key]).resolve()


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def build_hard_constraints(spec: dict[str, Any], asset: dict[str, Any]) -> list[str]:
    constraints = [
        "Dark graphite background only",
        "Only specified accent colors",
        "No readable real text baked into the image",
        "No glowing AI brains, neural networks, or particle effects",
        "Flat matte surfaces, not glossy 3D renders",
        "Editorial documentary tone, not startup ad, not sci-fi, not SaaS marketing",
        "Respect accent semantics exactly",
        f"{spec['project']['aspect_ratio']} aspect ratio composition",
        "AI is never shown as autonomous magic; keep boundaries, evidence, or review UI visible where relevant",
        "Founder images must feel candid, composed, and never heroic",
    ]
    if asset["asset_id"] not in spec["rules"]["crimson_usage"]["allowed_assets"]:
        constraints.append("Do not use crimson or red risk accents in this asset")
    return constraints


def render_prompt(spec: dict[str, Any], asset: dict[str, Any]) -> str:
    style = spec["global_style"]["style_preamble"].strip()
    body = asset["prompt_body"].strip()
    constraints = build_hard_constraints(spec, asset)
    negative = asset.get("negative_prompt", "").strip()
    note_lines = [f"- {note}" for note in asset.get("notes", [])]
    parts = [
        style,
        body,
        "Hard constraints:\n- " + "\n- ".join(constraints),
        f"Avoid:\n{negative}",
    ]
    if note_lines:
        parts.append("Production notes:\n" + "\n".join(note_lines))
    return "\n\n".join(parts)


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    if "," not in data_url:
        raise ValueError("Invalid data URL: missing separator")
    header, b64_data = data_url.split(",", maxsplit=1)
    if not header.startswith("data:"):
        raise ValueError("Invalid data URL: unsupported header")
    mime = header[5:].split(";", maxsplit=1)[0] or "image/png"
    try:
        return base64.b64decode(b64_data), mime
    except Exception as exc:
        raise ValueError("Invalid data URL: base64 decode failed") from exc


def extension_for_mime(mime: str) -> str:
    return ".jpg" if mime == "image/jpeg" else ".png"


def normalize_image_bytes(image_bytes: bytes, target_size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        converted = image.convert("RGBA")
        original_size = converted.size
        if converted.size != target_size:
            converted = ImageOps.fit(converted, target_size, method=Image.Resampling.LANCZOS)
        return converted, original_size


def save_data_url_image(data_url: str, output_path_no_ext: Path, target_size: tuple[int, int]) -> SavedImage:
    image_bytes, mime = decode_data_url(data_url)
    image, original_size = normalize_image_bytes(image_bytes, target_size)
    output_path = output_path_no_ext.with_suffix(extension_for_mime(mime))
    ensure_dir(output_path.parent)
    save_format = "PNG" if output_path.suffix.lower() == ".png" else "JPEG"
    save_kwargs: dict[str, Any] = {}
    if save_format == "JPEG":
        image = image.convert("RGB")
        save_kwargs["quality"] = 95
    image.save(output_path, format=save_format, **save_kwargs)
    return SavedImage(path=output_path, mime=mime, original_size=original_size, final_size=image.size)


def load_openrouter_headers() -> dict[str, str]:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Set it in your environment or a local .env file."
        )
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", DEFAULT_REFERER).strip() or DEFAULT_REFERER
    title = os.environ.get("OPENROUTER_APP_TITLE", DEFAULT_TITLE).strip() or DEFAULT_TITLE
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": title,
    }


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
            elif item:
                chunks.append(str(item))
        return "\n".join(chunks).strip()
    if isinstance(content, dict):
        text = content.get("text")
        return str(text).strip() if text else json.dumps(content)
    return "" if content is None else str(content)


def raise_openrouter_error(response: requests.Response) -> None:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = None
    error_message = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            error_message = str(error.get("message", "")).strip()
    suffix = f" API message: {error_message}" if error_message else ""
    raise RuntimeError(f"OpenRouter request failed with {response.status_code} {response.reason}.{suffix}")


def generate_images(config: ImageWorkflowConfig) -> Path:
    spec = load_spec(config.spec_path)
    validate_spec(spec)
    headers = load_openrouter_headers()
    assets = normalize_asset_order(spec, config.only_assets)
    output_dir = ensure_dir(resolved_project_path(spec, config.spec_path, "output_dir", config.output_dir))
    prompt_dir = ensure_dir(resolved_project_path(spec, config.spec_path, "prompt_log_dir", config.prompt_dir))
    model = config.model or spec["project"].get("model", DEFAULT_IMAGE_MODEL)
    target_size = parse_target_resolution(spec)
    candidates = config.candidates or spec["project"].get("draft_candidates_per_asset", 1)

    run_manifest = {
        "started_at": utc_now_iso(),
        "model": model,
        "aspect_ratio": spec["project"]["aspect_ratio"],
        "target_resolution": spec["project"]["target_resolution"],
        "assets": [],
    }

    for asset in assets:
        prompt = render_prompt(spec, asset)
        asset_output_dir = ensure_dir(output_dir / asset["asset_id"])
        for candidate_index in range(1, candidates + 1):
            prompt_path = prompt_dir / f"{asset['asset_id']}_cand{candidate_index}.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")

            payload = {
                "model": model,
                "modalities": ["image", "text"],
                "messages": [{"role": "user", "content": prompt}],
                "image_config": {"aspect_ratio": spec["project"]["aspect_ratio"]},
            }
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=config.timeout,
            )
            if not response.ok:
                raise_openrouter_error(response)
            data = response.json()
            choice = data["choices"][0]["message"]
            images = choice.get("images", [])
            text_response = flatten_message_content(choice.get("content"))
            if not images:
                raise RuntimeError(f"No image returned for {asset['asset_id']} candidate {candidate_index}.")

            saved_images = []
            for image_index, image_obj in enumerate(images, start=1):
                data_url = image_obj["image_url"]["url"]
                output_stem = asset_output_dir / f"{asset['asset_id']}_cand{candidate_index}_{image_index}"
                saved = save_data_url_image(data_url, output_stem, target_size)
                saved_images.append(
                    {
                        "file": relative_to_root(saved.path, output_dir),
                        "mime": saved.mime,
                        "original_size": list(saved.original_size),
                        "final_size": list(saved.final_size),
                    }
                )

            metadata_path = asset_output_dir / f"{asset['asset_id']}_cand{candidate_index}_meta.json"
            metadata = {
                "asset_id": asset["asset_id"],
                "candidate_index": candidate_index,
                "timestamp": utc_now_iso(),
                "response_id": data.get("id"),
                "model": model,
                "generation_mode": asset.get("generation_mode", "direct"),
                "rendered_prompt_file": relative_to_root(prompt_path, prompt_dir),
                "text_response": text_response,
                "saved_images": saved_images,
                "metadata_file": relative_to_root(metadata_path, output_dir),
            }
            write_json(metadata_path, metadata)
            run_manifest["assets"].append(metadata)

    manifest_path = output_dir / "run_manifest.json"
    run_manifest["completed_at"] = utc_now_iso()
    write_json(manifest_path, run_manifest)
    return manifest_path


def collect_candidate_metadata(asset_output_dir: Path, asset_id: str) -> list[Path]:
    return sorted(asset_output_dir.glob(f"{asset_id}_cand*_meta.json"))


def check_image_dimensions(image_path: Path, target_size: tuple[int, int]) -> bool:
    with Image.open(image_path) as image:
        return image.size == target_size


def check_image_aspect_ratio(image_path: Path, target_size: tuple[int, int]) -> bool:
    with Image.open(image_path) as image:
        width, height = image.size
    target_width, target_height = target_size
    return width * target_height == height * target_width


def review_status(checks: dict[str, bool], manual_review_required: bool) -> str:
    required_checks = {key: value for key, value in checks.items() if key != "manual_review_required"}
    if not all(required_checks.values()):
        return "failed"
    return "needs_review" if manual_review_required else "passed"


def summarize_asset_reviews(asset_id: str, reviews: list[dict[str, Any]], manual_review_required: bool) -> dict[str, Any]:
    statuses = {review["status"] for review in reviews}
    if "failed" in statuses:
        status = "failed"
    elif "needs_review" in statuses:
        status = "needs_review"
    else:
        status = "passed"
    notes: list[str] = []
    for review in reviews:
        for note in review.get("notes", []):
            if note not in notes:
                notes.append(note)
    return {
        "asset_id": asset_id,
        "candidate": [review["candidate"] for review in reviews],
        "status": status,
        "checks": {
            "metadata_exists": any(review["checks"]["metadata_exists"] for review in reviews),
            "prompt_logged": any(review["checks"]["prompt_logged"] for review in reviews),
            "output_exists": any(review["checks"]["output_exists"] for review in reviews),
            "aspect_ratio_ok": any(review["checks"]["aspect_ratio_ok"] for review in reviews),
            "resolution_ok": any(review["checks"]["resolution_ok"] for review in reviews),
            "manual_review_required": manual_review_required,
        },
        "notes": notes,
    }


def run_image_qa(config: ImageWorkflowConfig) -> Path:
    spec = load_spec(config.spec_path)
    validate_spec(spec)
    target_size = parse_target_resolution(spec)
    output_dir = ensure_dir(resolved_project_path(spec, config.spec_path, "output_dir", config.output_dir))
    prompt_dir = ensure_dir(resolved_project_path(spec, config.spec_path, "prompt_log_dir", config.prompt_dir))
    review_dir = ensure_dir(resolved_project_path(spec, config.spec_path, "review_dir", config.review_dir))
    aggregate = {
        "generated_at": utc_now_iso(),
        "target_resolution": list(target_size),
        "reviews": [],
    }

    for asset in normalize_asset_order(spec, config.only_assets):
        asset_output_dir = output_dir / asset["asset_id"]
        metadata_paths = collect_candidate_metadata(asset_output_dir, asset["asset_id"])
        if not metadata_paths:
            review = {
                "asset_id": asset["asset_id"],
                "candidate": None,
                "checks": {
                    "metadata_exists": False,
                    "prompt_logged": False,
                    "output_exists": False,
                    "aspect_ratio_ok": False,
                    "resolution_ok": False,
                    "manual_review_required": asset.get("manual_review_required", False),
                },
                "status": "failed",
                "notes": ["No candidate metadata files found."],
            }
            aggregate["reviews"].append(review)
            write_json(review_dir / f"{asset['asset_id']}_review.json", review)
            continue

        asset_reviews: list[dict[str, Any]] = []
        for metadata_path in metadata_paths:
            metadata = read_json(metadata_path)
            candidate = metadata["candidate_index"]
            prompt_path = prompt_dir / metadata["rendered_prompt_file"]
            saved_images = metadata.get("saved_images", [])
            image_paths = [output_dir / image["file"] for image in saved_images]
            output_exists = bool(saved_images) and all(path.is_file() for path in image_paths)
            checks = {
                "metadata_exists": True,
                "prompt_logged": prompt_path.is_file(),
                "output_exists": output_exists,
                "aspect_ratio_ok": output_exists and all(
                    check_image_aspect_ratio(path, target_size) for path in image_paths
                ),
                "resolution_ok": output_exists and all(
                    check_image_dimensions(path, target_size) for path in image_paths
                ),
                "manual_review_required": asset.get("manual_review_required", False),
            }
            notes: list[str] = []
            if asset.get("manual_review_required"):
                notes.append("Manual review required for composition- or overlay-sensitive assets.")
            review = {
                "asset_id": asset["asset_id"],
                "candidate": candidate,
                "checks": checks,
                "status": review_status(checks, asset.get("manual_review_required", False)),
                "notes": notes,
                "files": {
                    "metadata": relative_to_root(metadata_path, output_dir),
                    "prompt": metadata["rendered_prompt_file"],
                    "images": metadata["saved_images"],
                },
            }
            review_path = review_dir / f"{asset['asset_id']}_cand{candidate}_review.json"
            write_json(review_path, review)
            aggregate["reviews"].append(review)
            asset_reviews.append(review)

        write_json(
            review_dir / f"{asset['asset_id']}_review.json",
            summarize_asset_reviews(asset["asset_id"], asset_reviews, asset.get("manual_review_required", False)),
        )

    manifest_path = review_dir / "review_manifest.json"
    write_json(manifest_path, aggregate)
    return manifest_path


def normalized_box_to_pixels(box: dict[str, float], canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = canvas_size
    x = int(box["x"] * width)
    y = int(box["y"] * height)
    w = int(box["w"] * width)
    h = int(box["h"] * height)
    return x, y, x + w, y + h


def compose_founder_images(
    config: ImageWorkflowConfig,
    asset_ids: list[str] | None = None,
    candidate: int = 1,
    image_index: int = 1,
) -> list[Path]:
    spec = load_spec(config.spec_path)
    validate_spec(spec)
    output_dir = ensure_dir(resolved_project_path(spec, config.spec_path, "output_dir", config.output_dir))
    composite_dir = ensure_dir(resolved_project_path(spec, config.spec_path, "composite_dir", config.composite_dir))
    requested_assets = asset_ids or [
        asset["asset_id"] for asset in spec["assets"] if asset.get("generation_mode") == "background_plate_only"
    ]
    outputs: list[Path] = []

    for asset in normalize_asset_order(spec, requested_assets):
        if asset.get("generation_mode") != "background_plate_only":
            raise ValueError(f"{asset['asset_id']} is not configured for background plate compositing.")
        metadata_path = output_dir / asset["asset_id"] / f"{asset['asset_id']}_cand{candidate}_meta.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata for {asset['asset_id']} candidate {candidate}: {metadata_path}")
        metadata = read_json(metadata_path)
        saved_images = metadata.get("saved_images", [])
        if image_index < 1 or image_index > len(saved_images):
            raise ValueError(f"Image index {image_index} is out of range for {asset['asset_id']}.")

        plate_path = output_dir / saved_images[image_index - 1]["file"]
        founder_path = composite_dir / asset["composite_subject"]
        if not founder_path.is_file():
            raise FileNotFoundError(f"Founder source image not found: {founder_path}")

        with Image.open(plate_path).convert("RGBA") as plate, Image.open(founder_path).convert("RGBA") as founder:
            box = normalized_box_to_pixels(asset["composite_zone"], plate.size)
            target_size = (box[2] - box[0], box[3] - box[1])
            founder_layer = ImageOps.contain(founder, target_size, method=Image.Resampling.LANCZOS)
            paste_x = box[0] + (target_size[0] - founder_layer.size[0]) // 2
            paste_y = box[1] + (target_size[1] - founder_layer.size[1]) // 2
            plate.alpha_composite(founder_layer, dest=(paste_x, paste_y))
            composite_output = plate.convert("RGB")
            composite_path = plate_path.with_name(f"{plate_path.stem}_composited.png")
            composite_output.save(composite_path, format="PNG")

        write_json(
            composite_path.with_suffix(".json"),
            {
                "asset_id": asset["asset_id"],
                "candidate_index": candidate,
                "image_index": image_index,
                "timestamp": utc_now_iso(),
                "plate_file": relative_to_root(plate_path, output_dir),
                "founder_file": relative_to_root(founder_path, composite_dir),
                "output_file": relative_to_root(composite_path, output_dir),
            },
        )
        outputs.append(composite_path)
    return outputs
