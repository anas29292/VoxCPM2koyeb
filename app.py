from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from math import gcd
from pathlib import Path
from typing import Annotated, Any, Optional, Literal

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.signal import resample_poly
from starlette.concurrency import run_in_threadpool

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("koyeb-voxcpm2")

MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DATA_DIR / "models" / "VoxCPM2")))
HF_HOME = Path(os.getenv("HF_HOME", str(DATA_DIR / "huggingface")))
MODELSCOPE_CACHE = Path(os.getenv("MODELSCOPE_CACHE", str(DATA_DIR / "modelscope")))
DEVICE = os.getenv("DEVICE", "cuda")
OPTIMIZE = os.getenv("OPTIMIZE", "0").strip().lower() in {"1", "true", "yes", "on"}
LOAD_DENOISER = os.getenv("LOAD_DENOISER", "0").strip().lower() in {"1", "true", "yes", "on"}
API_KEY = os.getenv("API_KEY", "").strip()

MAX_TEXT_CHARS = max(100, int(os.getenv("MAX_TEXT_CHARS", "3000")))
MAX_CONTROL_CHARS = max(20, int(os.getenv("MAX_CONTROL_CHARS", "500")))
MAX_PROMPT_TEXT_CHARS = max(100, int(os.getenv("MAX_PROMPT_TEXT_CHARS", "5000")))
MAX_AUDIO_MB = max(1.0, float(os.getenv("MAX_AUDIO_MB", "20")))
MAX_REFERENCE_SECONDS = max(1.0, float(os.getenv("MAX_REFERENCE_SECONDS", "60")))
MAX_REQUEST_MB = max(MAX_AUDIO_MB * 1.4, float(os.getenv("MAX_REQUEST_MB", "30")))
MAX_INFLIGHT_REQUESTS = max(1, int(os.getenv("MAX_INFLIGHT_REQUESTS", "8")))
QUEUE_WAIT_TIMEOUT_SECONDS = max(1.0, float(os.getenv("QUEUE_WAIT_TIMEOUT_SECONDS", "180")))
SEGMENT_CHARS = max(100, int(os.getenv("SEGMENT_CHARS", "600")))

origins_value = os.getenv("CORS_ORIGINS", "*").strip()
CORS_ORIGINS = [item.strip() for item in origins_value.split(",") if item.strip()]

for directory in (DATA_DIR, MODEL_PATH.parent, HF_HOME, MODELSCOPE_CACHE):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("MODELSCOPE_CACHE", str(MODELSCOPE_CACHE))

# Keep one model per process. The Dockerfile should continue to run one Uvicorn worker.
model: Any = None
model_status = "loading"
model_error: Optional[str] = None
model_loaded_at: Optional[float] = None
model_load_task: Optional[asyncio.Task] = None

# Safe default for one GPU: a single inference at a time, with a bounded queue.
generation_semaphore = asyncio.Semaphore(1)
pending_counter_lock = asyncio.Lock()
pending_requests = 0

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationOptions(StrictModel):
    cfg_value: float = Field(default=2.0, ge=0.5, le=4.0, description="Guidance scale; usual range is 1.0-3.0")
    inference_timesteps: int = Field(default=10, ge=4, le=30, description="More steps may improve detail but take longer")
    normalize: bool = Field(default=False, description="Normalize numbers, dates, and abbreviations")
    retry_badcase: bool = Field(default=True, description="Retry abnormal very short/long generations")
    output_format: Literal["wav"] = "wav"


class TTSRequest(GenerationOptions):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, description="Text to synthesize")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class VoiceDesignRequest(TTSRequest):
    control_instruction: str = Field(
        min_length=3,
        max_length=MAX_CONTROL_CHARS,
        description="Natural-language voice description, e.g. 'warm adult female voice, calm and clear'",
    )

    @field_validator("control_instruction")
    @classmethod
    def control_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("control_instruction must not be blank")
        return value


class VoiceCloningRequest(GenerationOptions):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    reference_audio_base64: str = Field(min_length=8, description="Base64-encoded reference audio")
    control_instruction: Optional[str] = Field(
        default=None,
        max_length=MAX_CONTROL_CHARS,
        description="Optional natural-language style instruction (emotion, pace, delivery)",
    )
    denoise: bool = Field(default=False, description="Denoise the reference audio; requires LOAD_DENOISER=1")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value

    @field_validator("control_instruction")
    @classmethod
    def control_must_not_be_blank_if_set(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if not value:
                return None
        return value


class HiFiCloningRequest(GenerationOptions):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS, description="New text to generate")
    reference_audio_base64: str = Field(min_length=8, description="Reference voice audio, Base64")
    prompt_text: str = Field(
        min_length=1,
        max_length=MAX_PROMPT_TEXT_CHARS,
        description="Exact transcript of the prompt/reference audio",
    )
    prompt_audio_base64: Optional[str] = Field(
        default=None,
        description="Optional separate prompt audio; when omitted, the reference audio is also used as prompt audio",
    )
    denoise: bool = Field(default=False, description="Denoise reference/prompt audio; requires LOAD_DENOISER=1")

    @field_validator("text", "prompt_text")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text fields must not be blank")
        return value


class HealthResponse(BaseModel):
    status: str
    model_status: str
    model_id: str
    device: str
    cuda_available: bool
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
def _download_and_load_model() -> None:
    global model, model_status, model_error, model_loaded_at
    try:
        model_status = "downloading"
        logger.info("Preparing model %s at %s", MODEL_ID, MODEL_PATH)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_PATH),
            cache_dir=str(HF_HOME),
        )
        if not (MODEL_PATH / "config.json").is_file():
            raise RuntimeError(f"Model download completed but {MODEL_PATH / 'config.json'} is missing")
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE is set to CUDA, but torch.cuda.is_available() is False. "
                "Check the Koyeb GPU instance and the container's NVIDIA driver compatibility."
            )

        model_status = "loading"
        logger.info(
            "Loading VoxCPM2 (device=%s, optimize=%s, denoiser=%s)",
            DEVICE,
            OPTIMIZE,
            LOAD_DENOISER,
        )
        from voxcpm import VoxCPM

        loaded = VoxCPM.from_pretrained(
            str(MODEL_PATH),
            load_denoiser=LOAD_DENOISER,
            cache_dir=str(HF_HOME),
            local_files_only=True,
            optimize=OPTIMIZE,
            device=DEVICE,
        )
        model = loaded
        model_loaded_at = time.time()
        model_error = None
        model_status = "ready"
        logger.info("VoxCPM2 is ready; sample_rate=%s", getattr(model.tts_model, "sample_rate", "unknown"))
    except Exception as exc:
        logger.exception("Model initialization failed")
        model_error = f"{type(exc).__name__}: {exc}"[:1000]
        model_status = "error"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_load_task
    logger.info(
        "Starting VoxCPM2 API model_id=%s, device=%s, CUDA=%s",
        MODEL_ID,
        DEVICE,
        torch.cuda.is_available(),
    )
    model_load_task = asyncio.create_task(asyncio.to_thread(_download_and_load_model))
    yield
    # Model teardown is left to process shutdown; CUDA/model objects should not be forcibly unloaded here.


app = FastAPI(
    title="VoxCPM2 Production API",
    version="2.0.0",
    description=(
        "Multilingual TTS, Voice Design, Controllable Voice Cloning, and Hi-Fi Voice Cloning. "
        "All generation endpoints require x-api-key or Authorization: Bearer. Audio is returned as Base64 WAV."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "System", "description": "Health, readiness and service info"},
        {"name": "Text to Speech", "description": "Generate speech from text"},
        {"name": "Voice Design", "description": "Design a new voice from a text description"},
        {"name": "Voice Cloning", "description": "Clone timbre from a reference recording"},
        {"name": "Hi-Fi Cloning", "description": "Transcript-guided, continuation-style voice cloning"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else [],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Retry-After"],
    max_age=600,
)


@app.middleware("http")
async def request_body_limit(request: Request, call_next):
    """Reject oversized JSON bodies when Content-Length is supplied by the client/proxy."""
    if request.url.path in {
        "/tts",
        "/voice-design",
        "/voice-cloning",
        "/voice-cloning/hifi",
    }:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > MAX_REQUEST_MB * 1024 * 1024
            except ValueError:
                too_large = True
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds {MAX_REQUEST_MB:.0f} MB limit"},
                )
    return await call_next(request)


@app.get("/", tags=["System"])
def root() -> dict:
    return {
        "service": "koyeb-voxcpm2",
        "model_id": MODEL_ID,
        "model_status": model_status,
        "health": "/health",
        "readiness": "/ready",
        "docs": "/docs",
        "endpoints": {
            "tts": "POST /tts",
            "voice_design": "POST /voice-design",
            "voice_cloning": "POST /voice-cloning",
            "hifi_cloning": "POST /voice-cloning/hifi",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    # Liveness remains available while multi-GB model files are downloading/loading.
    return HealthResponse(
        status="ok",
        model_status=model_status,
        model_id=MODEL_ID,
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
        error=model_error if model_status == "error" else None,
    )


@app.get("/ready", tags=["System"])
def readiness() -> dict:
    if model_status != "ready" or model is None:
        raise HTTPException(
            status_code=503,
            detail={"model_status": model_status, "error": model_error},
            headers={"Retry-After": "10"},
        )
    return {
        "status": "ready",
        "model_id": MODEL_ID,
        "sample_rate": int(getattr(model.tts_model, "sample_rate", 48000)),
    }


def _authenticate(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API_KEY is not configured on the server")
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    supplied = x_api_key or bearer
    if not supplied or not hmac.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def _decode_base64_audio(value: str, field_name: str) -> bytes:
    payload = value.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    # Check encoded size before allocating the decoded audio bytes.
    if len(payload) * 3 / 4 > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"{field_name} exceeds {MAX_AUDIO_MB:.0f} MB")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid Base64") from exc
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    return raw


def _prepare_wav_16k(raw: bytes, destination: Path, field_name: str) -> float:
    """Convert a decodable reference audio file to mono 16 kHz PCM WAV."""
    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} could not be decoded. Use an audio format supported by SoundFile (WAV/FLAC recommended).",
        ) from exc
    if audio.size == 0 or sample_rate <= 0:
        raise HTTPException(status_code=400, detail=f"{field_name} contains no audio")
    if not np.isfinite(audio).all():
        raise HTTPException(status_code=400, detail=f"{field_name} contains invalid samples")
    mono = audio.mean(axis=1, dtype=np.float32)
    duration = len(mono) / float(sample_rate)
    if duration > MAX_REFERENCE_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} is {duration:.1f}s; maximum is {MAX_REFERENCE_SECONDS:.0f}s",
        )
    if sample_rate != 16000:
        divisor = gcd(int(sample_rate), 16000)
        mono = resample_poly(mono, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    sf.write(str(destination), mono, 16000, subtype="PCM_16", format="WAV")
    return duration


def _clean_instruction(instruction: Optional[str]) -> str:
    # The model's control-instruction format is (instruction)target text.
    cleaned = re.sub(r"[()（）]", "", (instruction or "")).strip()
    return cleaned[:MAX_CONTROL_CHARS]


def _split_text(text: str, max_chars: int = SEGMENT_CHARS) -> list[str]:
    """Split very long text on sentence/space boundaries to reduce unstable generations."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentence_list = re.split(r"(?<=[.!?。！？؟])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentence_list:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            words = sentence.split()
            if len(words) <= 1:
                words = [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]
            pieces: list[str] = []
            piece = ""
            for word in words:
                if not piece:
                    piece = word
                elif len(piece) + 1 + len(word) <= max_chars:
                    piece += " " + word
                else:
                    pieces.append(piece)
                    piece = word
            if piece:
                pieces.append(piece)
            for part in pieces:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(part)
            continue
        candidate = sentence if not current else current + " " + sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [text]


def _to_float_wave(wave: Any) -> np.ndarray:
    arr = np.asarray(wave, dtype=np.float32).squeeze()
    if arr.ndim != 1 or arr.size == 0:
        raise RuntimeError(f"Model returned unexpected audio array shape: {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError("Model returned non-finite audio samples")
    return arr


def _generate_audio_sync(mode: str, payload: GenerationOptions) -> tuple[bytes, int, float, int]:
    """Synchronous model call. The async generation gate serializes this function."""
    if model is None or model_status != "ready":
        raise RuntimeError("VoxCPM2 model is not ready")

    denoise = bool(getattr(payload, "denoise", False))
    if denoise and not LOAD_DENOISER:
        raise ValueError("denoise=true requires LOAD_DENOISER=1 on the server; otherwise use denoise=false")

    reference_value = getattr(payload, "reference_audio_base64", None)
    prompt_audio_value = getattr(payload, "prompt_audio_base64", None)
    reference_transcript = getattr(payload, "prompt_text", None)
    control_instruction = _clean_instruction(getattr(payload, "control_instruction", None))

    with tempfile.TemporaryDirectory(prefix="voxcpm2-", dir="/tmp") as tmp_dir:
        tmp = Path(tmp_dir)
        reference_path: Optional[Path] = None
        prompt_path: Optional[Path] = None

        if reference_value:
            reference_raw = _decode_base64_audio(reference_value, "reference_audio_base64")
            reference_path = tmp / "reference_16k.wav"
            _prepare_wav_16k(reference_raw, reference_path, "reference_audio_base64")

        if prompt_audio_value:
            prompt_raw = _decode_base64_audio(prompt_audio_value, "prompt_audio_base64")
            prompt_path = tmp / "prompt_16k.wav"
            _prepare_wav_16k(prompt_raw, prompt_path, "prompt_audio_base64")

        if mode == "hifi_cloning":
            if reference_path is None:
                raise ValueError("reference_audio_base64 is required for Hi-Fi cloning")
            if not reference_transcript:
                raise ValueError("prompt_text is required for Hi-Fi cloning")
            if prompt_path is None:
                prompt_path = reference_path
        elif mode == "voice_cloning" and reference_path is None:
            raise ValueError("reference_audio_base64 is required for voice cloning")
        elif mode in {"tts", "voice_design"} and reference_path is not None:
            raise ValueError("Reference audio is not accepted for this mode; use /voice-cloning")

        target_chunks = _split_text(payload.text)
        all_waves: list[np.ndarray] = []
        # Short deterministic silence between chunks avoids clicks and helps preserve sentence boundaries.
        sample_rate = int(getattr(model.tts_model, "sample_rate", 48000))
        silence = np.zeros(int(sample_rate * 0.12), dtype=np.float32)

        for index, chunk in enumerate(target_chunks):
            if mode in {"voice_design", "voice_cloning"} and control_instruction:
                final_text = f"({control_instruction}){chunk}"
            else:
                final_text = chunk

            kwargs: dict[str, Any] = {
                "text": final_text,
                "cfg_value": float(payload.cfg_value),
                "inference_timesteps": int(payload.inference_timesteps),
                "normalize": bool(payload.normalize),
                "denoise": denoise,
                "retry_badcase": bool(payload.retry_badcase),
            }
            if reference_path is not None:
                kwargs["reference_wav_path"] = str(reference_path)
            if mode == "hifi_cloning":
                kwargs["prompt_wav_path"] = str(prompt_path)
                kwargs["prompt_text"] = reference_transcript

            # Do not pass seed: the voxcpm==2.0.3 runtime used by this service
            # raises TypeError when seed is passed to VoxCPM._generate().
            wave = _to_float_wave(model.generate(**kwargs))
            all_waves.append(wave)
            if index < len(target_chunks) - 1:
                all_waves.append(silence)

        complete_wave = np.concatenate(all_waves) if len(all_waves) > 1 else all_waves[0]
        duration = float(len(complete_wave) / sample_rate)
        output = io.BytesIO()
        sf.write(output, complete_wave, sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue(), sample_rate, duration, len(target_chunks)


@asynccontextmanager
async def generation_queue():
    """Limit total active+waiting requests and serialize inference for one GPU/model."""
    global pending_requests
    async with pending_counter_lock:
        if pending_requests >= MAX_INFLIGHT_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="The speech generation queue is full. Retry shortly.",
                headers={"Retry-After": "10"},
            )
        pending_requests += 1

    acquired = False
    try:
        try:
            await asyncio.wait_for(generation_semaphore.acquire(), timeout=QUEUE_WAIT_TIMEOUT_SECONDS)
            acquired = True
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=429,
                detail="Timed out waiting for the GPU queue. Retry with shorter text or try again later.",
                headers={"Retry-After": "15"},
            ) from exc
        yield
    finally:
        if acquired:
            generation_semaphore.release()
        async with pending_counter_lock:
            pending_requests = max(0, pending_requests - 1)


async def _execute_generation(
    *,
    mode: str,
    payload: GenerationOptions,
    authorization: Optional[str],
    x_api_key: Optional[str],
    request_id: str,
) -> dict:
    _authenticate(authorization, x_api_key)
    if model_status != "ready" or model is None:
        raise HTTPException(
            status_code=503,
            detail={"message": "Model is not ready yet", "model_status": model_status, "error": model_error},
            headers={"Retry-After": "10"},
        )
    if getattr(payload, "denoise", False) and not LOAD_DENOISER:
        raise HTTPException(status_code=422, detail="denoise=true requires LOAD_DENOISER=1 on the server")

    started = time.perf_counter()
    async with generation_queue():
        try:
            audio_bytes, sample_rate, duration, segment_count = await run_in_threadpool(
                _generate_audio_sync, mode, payload
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Generation failed request_id=%s mode=%s", request_id, mode)
            raise HTTPException(
                status_code=500,
                detail={"message": "Audio generation failed", "request_id": request_id, "error_type": type(exc).__name__},
            ) from exc

    elapsed = time.perf_counter() - started
    logger.info(
        "Generated mode=%s duration=%.2fs elapsed=%.2fs segments=%d request_id=%s",
        mode,
        duration,
        elapsed,
        segment_count,
        request_id,
    )
    return {
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 3),
        "output_format": "wav",
        "model": MODEL_ID,
        "mode": mode,
        "generation_seconds": round(elapsed, 3),
        "segments": segment_count,
        "request_id": request_id,
    }


def _request_id(value: Optional[str]) -> str:
    if value and re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", value):
        return value
    return str(uuid.uuid4())


# -----------------------------------------------------------------------------
# Generation endpoints
# -----------------------------------------------------------------------------
@app.post("/tts", tags=["Text to Speech"], summary="Generate speech from text")
async def tts(
    payload: TTSRequest,
    authorization: Annotated[Optional[str], Header()] = None,
    x_api_key: Annotated[Optional[str], Header()] = None,
    x_request_id: Annotated[Optional[str], Header()] = None,
) -> dict:
    return await _execute_generation(
        mode="tts",
        payload=payload,
        authorization=authorization,
        x_api_key=x_api_key,
        request_id=_request_id(x_request_id),
    )


@app.post("/voice-design", tags=["Voice Design"], summary="Design a new voice from a description")
async def voice_design(
    payload: VoiceDesignRequest,
    authorization: Annotated[Optional[str], Header()] = None,
    x_api_key: Annotated[Optional[str], Header()] = None,
    x_request_id: Annotated[Optional[str], Header()] = None,
) -> dict:
    return await _execute_generation(
        mode="voice_design",
        payload=payload,
        authorization=authorization,
        x_api_key=x_api_key,
        request_id=_request_id(x_request_id),
    )


@app.post("/voice-cloning", tags=["Voice Cloning"], summary="Clone a reference voice with optional style control")
async def voice_cloning(
    payload: VoiceCloningRequest,
    authorization: Annotated[Optional[str], Header()] = None,
    x_api_key: Annotated[Optional[str], Header()] = None,
    x_request_id: Annotated[Optional[str], Header()] = None,
) -> dict:
    return await _execute_generation(
        mode="voice_cloning",
        payload=payload,
        authorization=authorization,
        x_api_key=x_api_key,
        request_id=_request_id(x_request_id),
    )


@app.post(
    "/voice-cloning/hifi",
    tags=["Hi-Fi Cloning"],
    summary="Transcript-guided high-fidelity voice cloning",
)
async def voice_cloning_hifi(
    payload: HiFiCloningRequest,
    authorization: Annotated[Optional[str], Header()] = None,
    x_api_key: Annotated[Optional[str], Header()] = None,
    x_request_id: Annotated[Optional[str], Header()] = None,
) -> dict:
    return await _execute_generation(
        mode="hifi_cloning",
        payload=payload,
        authorization=authorization,
        x_api_key=x_api_key,
        request_id=_request_id(x_request_id),
    )


@app.get("/stats", tags=["System"], summary="Read basic queue and model status (requires API key)")
async def stats(
    authorization: Annotated[Optional[str], Header()] = None,
    x_api_key: Annotated[Optional[str], Header()] = None,
) -> dict:
    _authenticate(authorization, x_api_key)
    async with pending_counter_lock:
        queued = pending_requests
    return {
        "model_status": model_status,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "pending_requests": queued,
        "max_inflight_requests": MAX_INFLIGHT_REQUESTS,
        "queue_wait_timeout_seconds": QUEUE_WAIT_TIMEOUT_SECONDS,
        "model_loaded_at": model_loaded_at,
    }
