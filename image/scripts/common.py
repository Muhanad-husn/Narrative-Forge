from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, ImageOps


IMAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = IMAGE_ROOT.parent
DEFAULT_SPEC_PATH = IMAGE_ROOT / "storyboard_spec.json"
DEFAULT_DOC_PATH = IMAGE_ROOT / "docs" / "ai_image_generation_prompts.md"
TARGET_SIZE = (1920, 1080)
DEFAULT_REFERER = "https://localhost/storyboard-batch-generator"
DEFAULT_TITLE = "Storyboard Batch Generator"


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


def load_spec(spec_path: Path | None = None) -> dict[str, Any]:
    path = spec_path or DEFAULT_SPEC_PATH
    return read_json(path)


def get_asset_map(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {asset["asset_id"]: asset for asset in spec["assets"]}


def project_path(spec: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / spec["project"][key]


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


def build_hard_constraints(spec: dict[str, Any], asset: dict[str, Any]) -> list[str]:
    rules = spec["rules"]
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
        "Founder images must feel candid, composed, and never heroic"
    ]
    if asset["asset_id"] not in rules["crimson_usage"]["allowed_assets"]:
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


@dataclass
class SavedImage:
    path: Path
    mime: str
    original_size: tuple[int, int]
    final_size: tuple[int, int]


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
    if mime == "image/jpeg":
        return ".jpg"
    return ".png"


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
    save_kwargs = {}
    if save_format == "JPEG":
        image = image.convert("RGB")
        save_kwargs["quality"] = 95
    image.save(output_path, format=save_format, **save_kwargs)
    return SavedImage(
        path=output_path,
        mime=mime,
        original_size=original_size,
        final_size=image.size,
    )


def load_openrouter_headers() -> dict[str, str]:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to .env before running the image pipeline.")
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


def validate_spec(spec: dict[str, Any]) -> None:
    required_top_level = {"project", "global_style", "colors", "rules", "assets", "batch_order"}
    missing = sorted(required_top_level - spec.keys())
    if missing:
        raise ValueError(f"Spec missing required top-level keys: {', '.join(missing)}")

    asset_ids = [asset["asset_id"] for asset in spec["assets"]]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Spec contains duplicate asset_id values.")

    total_assets = spec["project"].get("total_assets")
    if total_assets is not None and len(asset_ids) != total_assets:
        raise ValueError(f"Spec asset count mismatch: expected {total_assets}, found {len(asset_ids)}.")

    batch_order = spec["batch_order"]
    unknown = [asset_id for asset_id in batch_order if asset_id not in asset_ids]
    if unknown:
        raise ValueError(f"batch_order references unknown asset ids: {', '.join(unknown)}")
    if len(batch_order) != len(set(batch_order)):
        raise ValueError("batch_order contains duplicate asset ids.")
    if set(batch_order) != set(asset_ids):
        missing = sorted(set(asset_ids) - set(batch_order))
        extra = sorted(set(batch_order) - set(asset_ids))
        details: list[str] = []
        if missing:
            details.append(f"missing assets: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected assets: {', '.join(extra)}")
        raise ValueError("batch_order must include every asset exactly once (" + "; ".join(details) + ").")

    required_founder_assets = {"IMG-S02-A", "IMG-S20-A"}
    founder_assets = {
        asset["asset_id"]
        for asset in spec["assets"]
        if asset.get("generation_mode") == "background_plate_only"
    }
    if not required_founder_assets.issubset(founder_assets):
        raise ValueError("Founder assets must be marked as background_plate_only.")

    crimson_allowed = set(spec["rules"]["crimson_usage"]["allowed_assets"])
    if crimson_allowed != {"IMG-S07-A", "IMG-S10-A"}:
        raise ValueError("Crimson must only be allowed on IMG-S07-A and IMG-S10-A.")

    for asset in spec["assets"]:
        accent_policy = asset.get("accent_policy", {})
        asset_id = asset["asset_id"]
        if accent_policy.get("crimson") and asset_id not in crimson_allowed:
            raise ValueError(f"{asset_id} enables crimson outside the allowed asset list.")
        if asset_id in required_founder_assets:
            if not asset.get("manual_review_required"):
                raise ValueError(f"{asset_id} must require manual review.")
            if not asset.get("composite_subject"):
                raise ValueError(f"{asset_id} must declare composite_subject.")
            if not asset.get("composite_zone"):
                raise ValueError(f"{asset_id} must declare composite_zone.")


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    return parser
