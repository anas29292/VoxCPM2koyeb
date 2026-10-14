from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests


def b64_file(path: str | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    return base64.b64encode(file_path.read_bytes()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test VoxCPM2 Koyeb API and save the returned WAV.")
    parser.add_argument("--url", default=os.getenv("API_URL", "").rstrip("/"), help="Service URL, e.g. https://my-app.koyeb.app")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key; can also be set using API_KEY env")
    parser.add_argument("--text", default="Selamat pagi. Ini adalah pengujian suara VoxCPM dua dalam bahasa Indonesia.")
    parser.add_argument("--reference-audio", help="Reference WAV for voice cloning")
    parser.add_argument("--prompt-audio", help="Optional prompt WAV; if used, --prompt-text is required")
    parser.add_argument("--prompt-text", help="Exact transcript of the prompt/reference clip")
    parser.add_argument("--output", default="voxcpm2-output.wav")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--normalize", action="store_true", help="Enable text normalization")
    parser.add_argument("--denoise", action="store_true", help="Requires server LOAD_DENOISER=1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if not args.url:
        parser.error("Set --url or API_URL to your Koyeb service URL")
    if not args.api_key:
        parser.error("Set --api-key or API_KEY")
    if args.prompt_audio and not args.prompt_text:
        parser.error("--prompt-text is required with --prompt-audio")

    payload = {
        "text": args.text,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.steps,
        "normalize": args.normalize,
        "denoise": args.denoise,
        "retry_badcase": True,
        "output_format": "wav",
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.reference_audio:
        payload["reference_audio_base64"] = b64_file(args.reference_audio)
    if args.prompt_audio:
        payload["prompt_audio_base64"] = b64_file(args.prompt_audio)
    if args.prompt_text:
        payload["prompt_text"] = args.prompt_text

    url = args.url + "/tts"
    print(f"POST {url}", flush=True)
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=args.timeout,
    )
    if not response.ok:
        try:
            details = response.json()
        except ValueError:
            details = response.text[:2000]
        print(f"HTTP {response.status_code}: {json.dumps(details, ensure_ascii=False)}", file=sys.stderr)
        return 1

    result = response.json()
    audio_base64 = result.get("audio_base64")
    if not audio_base64:
        print(f"Response did not contain audio_base64: {result}", file=sys.stderr)
        return 1
    output_path = Path(args.output)
    output_path.write_bytes(base64.b64decode(audio_base64))
    print(f"Saved: {output_path.resolve()}")
    print(f"Sample rate: {result.get('sample_rate')} Hz")
    print(f"Duration: {result.get('duration_seconds')} sec")
    print(f"Generation time: {result.get('generation_seconds')} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
