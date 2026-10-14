#!/usr/bin/env bash
set -euo pipefail
: "${API_URL:?Set API_URL, e.g. https://your-app.koyeb.app}"
: "${API_KEY:?Set API_KEY to the Koyeb secret}"

curl --fail-with-body --silent --show-error "${API_URL%/}/health"
printf '\n'
curl --fail-with-body --silent --show-error "${API_URL%/}/ready"
printf '\n'
# TTS sample; uses only text so does not need a reference file.
curl --fail-with-body --silent --show-error \
  -H "Authorization: Bearer ${API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Selamat pagi, ini tes VoxCPM2.","output_format":"wav"}' \
  "${API_URL%/}/tts"
