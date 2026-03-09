from __future__ import annotations

import difflib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
from scipy.signal import resample_poly


DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "onyx"
TARGET_SAMPLE_RATE = 48_000
TARGET_PEAK_DBFS = -3.0
TARGET_LUFS = -18.0
LEAD_IN_SECONDS = 0.5
TAIL_SECONDS = 1.0
MAX_TTS_INPUT_TOKENS = 2000
TOKEN_SAFETY_MARGIN = 100
MIN_TRANSCRIPT_SIMILARITY = 0.9
MAX_SYNTHESIS_ATTEMPTS = 2
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
DEFAULT_TTS_TONE = (
    "Narrate with calm confidence, natural pacing, and fully articulated technical terms. "
    "Stay human and grounded, not promotional or exaggerated."
)


@dataclass(frozen=True)
class TTSConfig:
    input_dir: Path
    output_dir: Path
    model: str = DEFAULT_TTS_MODEL
    voice: str = DEFAULT_VOICE
    instructions_path: Path | None = None
    keep_markdown: bool = False


def load_openai_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in your environment or a local .env file."
        )
    return api_key


def read_optional_instructions(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.is_file():
        raise FileNotFoundError(f"Instructions file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    word_based = len(text.split()) * 2
    char_based = max(1, len(text) // 4)
    return max(word_based, char_based)


def split_long_unit(text: str, max_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and estimate_tokens(candidate) > max_tokens:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)

    refined: list[str] = []
    for chunk in chunks:
        if estimate_tokens(chunk) <= max_tokens:
            refined.append(chunk)
            continue
        words = chunk.split()
        current_words: list[str] = []
        for word in words:
            candidate = " ".join([*current_words, word])
            if current_words and estimate_tokens(candidate) > max_tokens:
                refined.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            refined.append(" ".join(current_words))
    return refined


def chunk_model_input(text: str, max_tokens: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and estimate_tokens(candidate) > max_tokens:
            chunks.append(current)
            current = ""
        if estimate_tokens(paragraph) > max_tokens:
            chunks.extend(split_long_unit(paragraph, max_tokens))
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def compose_tts_instructions(extra_instructions: str, attempt: int) -> str:
    retry_note = ""
    if attempt > 0:
        retry_note = (
            "\n\nRetry requirement: read every sentence from the input exactly once, "
            "including the ending. Do not omit or soften the final sentence."
        )
    parts = [DEFAULT_TTS_TONE]
    if extra_instructions:
        parts.append(extra_instructions.strip())
    return "\n\n".join(parts) + retry_note


def max_chunk_tokens_for_request(extra_instructions: str) -> int:
    instruction_tokens = estimate_tokens(compose_tts_instructions(extra_instructions, 0))
    available = MAX_TTS_INPUT_TOKENS - instruction_tokens - TOKEN_SAFETY_MARGIN
    if available <= 0:
        raise ValueError(
            "Instructions are too large for the model limit. Shorten the instructions file."
        )
    return available


def list_markdown_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted(path for path in input_dir.glob("*.md") if path.is_file())


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return samples.mean(axis=1, dtype=np.float32)


def resample_if_needed(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return samples
    return resample_poly(samples, TARGET_SAMPLE_RATE, sample_rate).astype(np.float32, copy=False)


def loudness_normalize(samples: np.ndarray) -> np.ndarray:
    normalized = samples
    try:
        meter = pyln.Meter(TARGET_SAMPLE_RATE)
        measured_loudness = meter.integrated_loudness(samples)
        if np.isfinite(measured_loudness):
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            target_peak = 10 ** (TARGET_PEAK_DBFS / 20.0)
            effective_target_lufs = TARGET_LUFS
            if peak > 0:
                max_gain_db = 20.0 * np.log10(target_peak / peak)
                effective_target_lufs = min(TARGET_LUFS, measured_loudness + max_gain_db)
            normalized = pyln.normalize.loudness(
                samples,
                measured_loudness,
                effective_target_lufs,
            )
    except ValueError:
        normalized = samples

    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    target_peak = 10 ** (TARGET_PEAK_DBFS / 20.0)
    if peak > 0 and peak > target_peak:
        normalized = normalized * (target_peak / peak)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def add_required_silence(samples: np.ndarray) -> np.ndarray:
    lead_in = np.zeros(int(TARGET_SAMPLE_RATE * LEAD_IN_SECONDS), dtype=np.float32)
    tail = np.zeros(int(TARGET_SAMPLE_RATE * TAIL_SECONDS), dtype=np.float32)
    return np.concatenate([lead_in, samples, tail])


def normalize_transcript_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = text.strip().split()
    return " ".join(canonicalize_transcript_tokens(tokens))


def canonicalize_transcript_tokens(tokens: list[str]) -> list[str]:
    canonical: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in NUMBER_WORDS:
            value = NUMBER_WORDS[token]
            if (
                value >= 20
                and value % 10 == 0
                and index + 1 < len(tokens)
                and tokens[index + 1] in NUMBER_WORDS
                and 0 < NUMBER_WORDS[tokens[index + 1]] < 10
            ):
                value += NUMBER_WORDS[tokens[index + 1]]
                index += 1
            canonical.append(str(value))
        else:
            canonical.append(token)
        index += 1
    return canonical


def compact_transcript_text(text: str) -> str:
    return normalize_transcript_text(text).replace(" ", "")


def final_sentence(text: str) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return parts[-1] if parts else text.strip()


def split_for_retry(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) < 2:
        return [text.strip()]
    lead = " ".join(sentences[:-1]).strip()
    tail = sentences[-1]
    return [part for part in [lead, tail] if part]


def transcript_similarity(expected_text: str, transcript_text: str) -> float:
    expected_tokens = normalize_transcript_text(expected_text).split()
    transcript_tokens = normalize_transcript_text(transcript_text).split()
    expected_compact = "".join(expected_tokens)
    transcript_compact = "".join(transcript_tokens)
    if not expected_tokens or not transcript_tokens:
        return 0.0
    token_ratio = difflib.SequenceMatcher(a=expected_tokens, b=transcript_tokens).ratio()
    compact_ratio = difflib.SequenceMatcher(a=expected_compact, b=transcript_compact).ratio()
    return max(token_ratio, compact_ratio)


def transcribe_audio_file(client: OpenAI, audio_path: Path) -> str:
    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )
    return transcript.text.strip()


def validate_transcript_coverage(expected_text: str, transcript_text: str) -> tuple[bool, float]:
    similarity = transcript_similarity(expected_text, transcript_text)
    expected_ending = compact_transcript_text(final_sentence(expected_text))
    transcript_normalized = compact_transcript_text(transcript_text)
    has_expected_ending = bool(expected_ending) and expected_ending in transcript_normalized
    return similarity >= MIN_TRANSCRIPT_SIMILARITY and has_expected_ending, similarity


def post_process_wav_chunk(raw_wav: bytes, add_edge_silence: bool) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(io.BytesIO(raw_wav), dtype="float32", always_2d=False)
    samples = to_mono(samples)
    samples = resample_if_needed(samples, sample_rate)
    samples = loudness_normalize(samples)
    if add_edge_silence:
        samples = add_required_silence(samples)
    return samples, TARGET_SAMPLE_RATE


def synthesize_markdown_file(
    client: OpenAI,
    source_path: Path,
    config: TTSConfig,
    extra_instructions: str,
) -> Path:
    source_text = source_path.read_text(encoding="utf-8").strip()
    if not source_text:
        raise ValueError(f"Input file is empty: {source_path}")

    model_input = source_text if config.keep_markdown else strip_markdown(source_text)
    if not model_input:
        raise ValueError(f"No speakable content remained after Markdown cleanup: {source_path}")

    output_path = config.output_dir / f"{source_path.stem}.wav"
    max_chunk_tokens = max_chunk_tokens_for_request(extra_instructions)
    for attempt in range(MAX_SYNTHESIS_ATTEMPTS):
        instructions = compose_tts_instructions(extra_instructions, attempt)
        model_chunks = chunk_model_input(model_input, max_chunk_tokens)
        if attempt > 0:
            model_chunks = [
                retry_chunk
                for chunk in model_chunks
                for retry_chunk in split_for_retry(chunk)
                if retry_chunk
            ]
        if not model_chunks:
            raise ValueError(f"No speakable content remained after chunking: {source_path}")

        chunk_samples: list[np.ndarray] = []
        sample_rate = TARGET_SAMPLE_RATE
        for model_chunk in model_chunks:
            if estimate_tokens(model_chunk) > MAX_TTS_INPUT_TOKENS:
                raise ValueError(f"Chunk still exceeds the model limit after splitting: {source_path}")
            response = client.audio.speech.create(
                model=config.model,
                voice=config.voice,
                instructions=instructions,
                input=model_chunk,
                response_format="wav",
            )
            samples, sample_rate = post_process_wav_chunk(response.content, add_edge_silence=False)
            chunk_samples.append(samples)

        combined = np.concatenate(chunk_samples) if len(chunk_samples) > 1 else chunk_samples[0]
        sf.write(output_path, add_required_silence(combined), sample_rate, subtype="PCM_24")

        transcript_text = transcribe_audio_file(client, output_path)
        is_valid, similarity = validate_transcript_coverage(model_input, transcript_text)
        if is_valid:
            return output_path
        if attempt + 1 == MAX_SYNTHESIS_ATTEMPTS:
            raise ValueError(
                f"TTS output for {source_path.name} appears incomplete after retry "
                f"(transcript similarity {similarity:.2f})."
            )
    raise RuntimeError(f"Unreachable synthesis state for {source_path}")


def run_tts(config: TTSConfig) -> list[Path]:
    api_key = load_openai_api_key()
    extra_instructions = read_optional_instructions(config.instructions_path)
    input_files = list_markdown_files(config.input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .md files found in {config.input_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)
    outputs: list[Path] = []
    for source_path in input_files:
        outputs.append(
            synthesize_markdown_file(
                client=client,
                source_path=source_path,
                config=config,
                extra_instructions=extra_instructions,
            )
        )
    return outputs
