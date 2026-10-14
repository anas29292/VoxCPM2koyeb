from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from scipy.signal import resample_poly
from starlette.concurrency import run_in_threadpool

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
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "3000"))
MAX_AUDIO_MB = float(os.getenv("MAX_AUDIO_MB", "20"))
MAX_REFERENCE_SECONDS = float(os.getenv("MAX_REFERENCE_SECONDS", "60"))
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]

for path in (DATA_DIR, MODEL_PATH.parent, HF_HOME, MODELSCOPE_CACHE):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("MODELSCOPE_CACHE", str(MODELSCOPE_CACHE))

model = None
model_status = "loading"
model_error: Optional[str] = None
model_loaded_at: Optional[float] = None
model_load_task: Optional[asyncio.Task] = None
generation_lock = asyncio.Lock()


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    # Both fields expect Base64 containing an audio file, preferably WAV.
    reference_audio_base64: Optional[str] = None
    prompt_audio_base64: Optional[str] = None
    prompt_text: Optional[str] = Field(default=None, max_length=5000)
    cfg_value: float = Field(default=2.0, ge=0.1, le=10.0)
    inference_timesteps: int = Field(default=10, ge=1, le=50)
    normalize: bool = False
    denoise: bool = False
    retry_badcase: bool = True
    seed: Optional[int] = Field(default=None, ge=0, le=2_147_483_647)
    output_format: str = "wav"

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value.strip()

    @field_validator("output_format")
    @classmethod
    def only_wav_is_supported(cls, value: str) -> str:
        value = value.lower().strip()
        if value != "wav":
            raise ValueError("Only output_format='wav' is currently supported")
        return value

    @field_validator("prompt_text")
    @classmethod
    def prompt_text_not_blank_if_set(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("prompt_text must not be blank when provided")
        return value.strip() if value is not None else None


class HealthResponse(BaseModel):
    status: str
    model_status: str
    model_id: str
    device: str
    cuda_available: bool
    error: Optional[str] = None


def _download_and_load_model() -> None:
    """Download model files into the mounted persistent volume, then load once."""
    global model, model_status, model_error, model_loaded_at
    try:
        model_status = "downloading"
        logger.info("Preparing model %s at %s", MODEL_ID, MODEL_PATH)
        from huggingface_hub import snapshot_download

        # local_dir directs the downloaded model into persistent volume storage.
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_PATH),
            cache_dir=str(HF_HOME),
        )
        config_path = MODEL_PATH / "config.json"
        if not config_path.is_file():
            raise RuntimeError(f"Model download completed but {config_path} is missing")

        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE is set to CUDA, but torch.cuda.is_available() is False. "
                "Check that a Koyeb GPU instance is selected and the container has GPU access."
            )

        model_status = "loading"
        logger.info(
            "Loading VoxCPM model (device=%s, optimize=%s, denoiser=%s)",
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
        model_status = "ready"
        model_error = None
        logger.info("VoxCPM2 is ready; sample_rate=%s", getattr(model.tts_model, "sample_rate", "unknown"))
    except Exception as exc:  # keep the service process alive so the status endpoint remains useful
        logger.exception("Model initialization failed")
        model_error = f"{type(exc).__name__}: {exc}"[:1000]
        model_status = "error"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_load_task
    model_status_initial = model_status
    logger.info(
        "Starting Koyeb VoxCPM2 API; model_path=%s; device=%s; CUDA=%s; initial_status=%s",
        MODEL_PATH,
        DEVICE,
        torch.cuda.is_available(),
        model_status_initial,
    )
    model_load_task = asyncio.create_task(asyncio.to_thread(_download_and_load_model))
    yield
    # No attempt to cancel a blocking model download during ordinary shutdown.

app = FastAPI(
    title="VoxCPM2 TTS API for Koyeb",
    version="1.0.0",
    description="FastAPI wrapper for VoxCPM2 multilingual TTS and voice cloning.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else [],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    max_age=600,
)


@app.get("/")
def root() -> dict:
    return {
        "service": "koyeb-voxcpm2",
        "model_id": MODEL_ID,
        "health": "/health",
        "readiness": "/ready",
        "tts": "/tts",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Liveness endpoint remains 200 while a multi-GB model is downloading/loading.
    return HealthResponse(
        status="ok",
        model_status=model_status,
        model_id=MODEL_ID,
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
        error=model_error if model_status == "error" else None,
    )


@app.get("/ready")
def readiness() -> dict:
    if model_status != "ready" or model is None:
        raise HTTPException(
            status_code=503,
            detail={"model_status": model_status, "error": model_error},
        )
    return {"status": "ready", "model_id": MODEL_ID}


def _authenticate(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API_KEY is not configured on the server. Set it as a Koyeb secret/environment variable.",
        )
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
    # Reject obviously oversized payloads before allocating decoded bytes.
    estimated_size = len(payload) * 3 / 4
    if estimated_size > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"{field_name} exceeds MAX_AUDIO_MB={MAX_AUDIO_MB}")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid Base64") from exc
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    return raw


def _prepare_wav_16k(raw: bytes, destination: Path, field_name: str) -> float:
    """Decode WAV-like audio bytes, convert to mono 16 kHz PCM WAV for the model."""
    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} could not be decoded. Send a valid WAV/PCM audio file.",
        ) from exc
    if audio.size == 0 or sample_rate <= 0:
        raise HTTPException(status_code=400, detail=f"{field_name} contains no audio")
    mono = audio.mean(axis=1, dtype=np.float32)
    duration = len(mono) / float(sample_rate)
    if duration > MAX_REFERENCE_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} is {duration:.1f}s; maximum is {MAX_REFERENCE_SECONDS:.0f}s",
        )
    if sample_rate != 16000:
        # Reduce the ratio to keep resample_poly efficient and accurate.
        from math import gcd
        divisor = gcd(int(sample_rate), 16000)
        mono = resample_poly(mono, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    sf.write(str(destination), mono, 16000, subtype="PCM_16", format="WAV")
    return duration


def _generate_audio(req: TTSRequest) -> tuple[bytes, int, float]:
    if model is None or model_status != "ready":
        raise RuntimeError("VoxCPM2 model is not ready")
    if req.denoise and not LOAD_DENOISER:
        raise ValueError("denoise=true requires LOAD_DENOISER=1 on the server; otherwise use denoise=false")

    with tempfile.TemporaryDirectory(prefix="voxcpm2-", dir="/tmp") as tmp_dir:
        tmp = Path(tmp_dir)
        reference_path: Optional[Path] = None
        prompt_path: Optional[Path] = None

        if req.reference_audio_base64:
            raw_ref = _decode_base64_audio(req.reference_audio_base64, "reference_audio_base64")
            reference_path = tmp / "reference_16k.wav"
            _prepare_wav_16k(raw_ref, reference_path, "reference_audio_base64")

        if req.prompt_audio_base64:
            raw_prompt = _decode_base64_audio(req.prompt_audio_base64, "prompt_audio_base64")
            prompt_path = tmp / "prompt_16k.wav"
            _prepare_wav_16k(raw_prompt, prompt_path, "prompt_audio_base64")
            if not req.prompt_text:
                raise ValueError("prompt_text is required when prompt_audio_base64 is supplied")
        elif req.prompt_text and reference_path:
            # Highest-similarity/ultimate-cloning path: reuse reference clip + its transcript.
            prompt_path = reference_path
        elif req.prompt_text and not reference_path:
            raise ValueError("prompt_text requires reference_audio_base64 or prompt_audio_base64")

        kwargs = {
            "text": req.text,
            "cfg_value": req.cfg_value,
            "inference_timesteps": req.inference_timesteps,
            "normalize": req.normalize,
            "denoise": req.denoise,
            "retry_badcase": req.retry_badcase,
        }
        if reference_path:
            kwargs["reference_wav_path"] = str(reference_path)
        if prompt_path:
            kwargs["prompt_wav_path"] = str(prompt_path)
            kwargs["prompt_text"] = req.prompt_text

        wav = model.generate(**kwargs)
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        if wav.ndim != 1 or wav.size == 0:
            raise RuntimeError(f"Model returned unexpected audio array shape: {wav.shape}")
        sample_rate = int(getattr(model.tts_model, "sample_rate", 48000))
        duration = float(wav.shape[0] / sample_rate)
        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), sample_rate, duration


@app.post("/tts")
async def tts(
    payload: TTSRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    _authenticate(authorization, x_api_key)
    if model_status != "ready" or model is None:
        raise HTTPException(
            status_code=503,
            detail={"message": "Model is not ready yet", "model_status": model_status, "error": model_error},
            headers={"Retry-After": "10"},
        )
    if payload.prompt_audio_base64 and not payload.prompt_text:
        raise HTTPException(status_code=422, detail="prompt_text is required with prompt_audio_base64")
    if payload.prompt_text and not (payload.prompt_audio_base64 or payload.reference_audio_base64):
        raise HTTPException(status_code=422, detail="prompt_text requires a prompt/reference audio")
    if payload.denoise and not LOAD_DENOISER:
        raise HTTPException(status_code=422, detail="Set LOAD_DENOISER=1 to use denoise=true")

    start = time.perf_counter()
    async with generation_lock:
        try:
            audio_bytes, sample_rate, duration = await run_in_threadpool(_generate_audio, payload)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("TTS generation failed")
            raise HTTPException(status_code=500, detail=f"Audio generation failed: {type(exc).__name__}") from exc
    elapsed = time.perf_counter() - start
    logger.info("Generated %.2fs audio in %.2fs; clone=%s", duration, elapsed, bool(payload.reference_audio_base64 or payload.prompt_audio_base64))
    return {
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 3),
        "output_format": "wav",
        "model": MODEL_ID,
        "generation_seconds": round(elapsed, 3),
    }
