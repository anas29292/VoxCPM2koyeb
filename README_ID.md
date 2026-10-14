# VoxCPM2 TTS + Voice Cloning — Koyeb GPU

Paket ini membuat HTTP API FastAPI untuk model resmi [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2), dengan:

- `GET /health`: liveness, dapat mengembalikan `model_status=downloading/loading` saat startup.
- `GET /ready`: HTTP 200 hanya ketika model siap; jika belum siap akan mengembalikan HTTP 503.
- `POST /tts`: text-to-speech dan voice cloning, hasil WAV Base64.
- Download bobot model otomatis ke `MODEL_PATH` pada persistent volume.
- `download_model.py` untuk download manual, dan `test_api.py` untuk tes TTS/voice cloning.
- Autentikasi API menggunakan `Authorization: Bearer ...` atau `X-API-Key`.

> Status paket: source dan sintaks Python sudah diperiksa lokal. Image Docker belum dibuild dan inferensi belum diuji pada GPU Koyeb. Deployment perlu diverifikasi dari log Koyeb.

## 1. Hal yang perlu diketahui tentang Koyeb Volume

Dokumentasi Koyeb saat ini menyebut persistent volume masih public preview, kapasitas 1–10 GB, hanya region Washington, D.C. (`was`) dan Frankfurt (`fra`), dan Service dengan volume dibatasi ke satu instance. Periksa dashboard akun Anda untuk ketersediaan GPU + volume di region yang sama. Model VoxCPM2 membutuhkan beberapa GB; kapasitas 10 GB cukup ketat tetapi target model sekitar 4.6 GB. Jangan menyimpan audio output jangka panjang ke volume model. Buat snapshot/backup untuk file penting.

Scale-to-zero tersedia pada GPU Services di public preview dan mensyaratkan minimum instance 0. Namun cold start melibatkan pembuatan instance, download/cache model jika belum ada, dan loading GPU; permintaan pertama bisa jauh lebih lambat dari request berikutnya. Periksa kompatibilitas volume dengan konfigurasi sleep yang ditampilkan di dashboard sebelum mengandalkan pola biaya ini.

## 2. Buat repository

1. Ekstrak ZIP ini.
2. Commit semua file ke repository GitHub baru (jangan commit `.env` atau API key).
3. Koyeb: **Create Web Service → GitHub → pilih repository**. Pilih builder `Dockerfile`/Docker.
4. Pilih GPU instance. RTX 4000 SFF Ada (20 GB VRAM) atau L4 (24 GB) merupakan titik awal yang masuk akal untuk model 2B; ketersediaan/kelayakan akun harus dicek di dashboard.
5. Expose port `8000` sebagai HTTP, atau gunakan nilai `PORT` yang diberikan oleh platform.
6. Set startup grace period health check ke nilai besar (contoh 300–900 detik jika UI mengizinkan) karena model bisa perlu diunduh dan dimuat. Set health check path ke `/health`. `/health` sengaja tetap 200 ketika model sedang disiapkan. Gunakan `/ready` untuk memeriksa kesiapan model sesungguhnya.
7. Set environment variables/secrets seperti di bawah.

## 3. Persistent volume dan environment

Buat volume pada region yang sama dengan service. Lampirkan/mount ke:

```text
/data
```

Isi environment variables berikut (melalui dashboard Koyeb, bukan `.env` yang di-commit):

```text
API_KEY=<secret acak panjang>
MODEL_ID=openbmb/VoxCPM2
DATA_DIR=/data
MODEL_PATH=/data/models/VoxCPM2
HF_HOME=/data/huggingface
MODELSCOPE_CACHE=/data/modelscope
DEVICE=cuda
OPTIMIZE=0
LOAD_DENOISER=0
PORT=8000
MAX_TEXT_CHARS=3000
MAX_AUDIO_MB=20
MAX_REFERENCE_SECONDS=60
CORS_ORIGINS=*
```

`API_KEY` wajib. Jika belum diisi, endpoint `/tts` menolak request. Simpan sebagai Koyeb Secret jika tersedia.

- `OPTIMIZE=0` dipilih untuk startup yang lebih sederhana dan menghindari kompilasi awal. Setelah stabil, Anda dapat menguji `OPTIMIZE=1` lalu membandingkan startup dan latensi.
- `LOAD_DENOISER=0` mengurangi unduhan ekstra dan penggunaan memori. Jika ingin `denoise=true`, set `LOAD_DENOISER=1`, lalu uji kembali startup dan penyimpanan. Audio referensi yang bersih biasanya mempermudah perbandingan.
- `CORS_ORIGINS=*` memudahkan tes browser, tetapi jangan masukkan API key RunPod/Koyeb ke kode browser publik. Untuk produksi, gunakan backend/proxy milik aplikasi Anda dan batasi origin yang diizinkan.

## 4. Download model ke volume

Secara otomatis, saat service mulai, `app.py` menjalankan `snapshot_download()` dari Hugging Face ke `MODEL_PATH`. Endpoint `/health` akan menampilkan status `downloading`, kemudian `loading`, lalu `ready`. Karena file masuk ke volume persisten, deploy berikutnya memakai file yang sudah ada dan mengunduh hanya file yang perlu disegarkan.

Bila Anda ingin menjalankan download dari environment yang sudah memiliki paket dan volume terpasang:

```bash
python download_model.py
```

Model membutuhkan akses jaringan ke Hugging Face saat pertama kali diunduh. Bila deployment berhenti di status `downloading`, periksa log, kapasitas volume, dan koneksi Hugging Face.

## 5. Test kesehatan

Setelah URL Koyeb muncul, buka:

```text
https://NAMA-APLIKASI.koyeb.app/health
https://NAMA-APLIKASI.koyeb.app/ready
```

Poll `/health` sampai `model_status` bernilai `ready`. Jika `model_status=error`, lihat runtime logs. `/health` berstatus 200 ketika sedang loading agar health check platform tidak salah menganggap model 2B yang masih menginisialisasi sebagai proses mati.

## 6. Tes TTS biasa

Di komputer Anda, Python 3.10+ diperlukan untuk script tester; install `requests` bila belum tersedia:

```bash
python -m pip install requests
```

PowerShell:

```powershell
$env:API_URL = "https://NAMA-APLIKASI.koyeb.app"
$env:API_KEY = "SECRET_YANG_SAMA_DENGAN_KOYEB"
python test_api.py --text "Selamat pagi. Ini pengujian VoxCPM2 bahasa Indonesia." --output tts.wav
```

Linux/macOS:

```bash
export API_URL="https://NAMA-APLIKASI.koyeb.app"
export API_KEY="SECRET_YANG_SAMA_DENGAN_KOYEB"
python test_api.py --text "Selamat pagi. Ini pengujian VoxCPM2 bahasa Indonesia." --output tts.wav
```

## 7. Tes voice cloning

Gunakan file WAV yang berisi suara pembicara yang Anda miliki izin untuk gunakan. API menormalkan reference audio menjadi mono WAV 16 kHz, yang cocok dengan panduan VoxCPM2. Audio input saat ini diharapkan bisa dibaca `soundfile`, jadi WAV PCM paling aman. Potongan referensi bersih sekitar 4–30 detik sering lebih praktis untuk uji awal; batas server default 60 detik.

### A. Cloning menggunakan audio referensi saja

```bash
python test_api.py \
  --reference-audio reference.wav \
  --text "Selamat pagi, ini suara hasil pengujian kloning." \
  --output clone.wav
```

### B. Cloning dengan transkrip referensi (disarankan untuk uji kemiripan)

`--prompt-text` harus merupakan transkrip yang sesuai dengan isi file reference, bukan teks target yang akan diucapkan.

```bash
python test_api.py \
  --reference-audio reference.wav \
  --prompt-text "Tuliskan di sini kalimat yang benar-benar diucapkan dalam reference.wav." \
  --text "Hari ini kita menguji suara hasil kloning dengan VoxCPM2." \
  --output clone-ultimate.wav
```

Bila audio prompt berbeda dengan audio referensi, gunakan `--prompt-audio prompt.wav` bersama `--prompt-text`, dan sertakan `--reference-audio reference.wav` bila ingin memberi keduanya ke model.

## 8. Bentuk API

Request JSON untuk TTS dasar:

```json
{
  "text": "Selamat pagi, ini pengujian suara.",
  "output_format": "wav",
  "cfg_value": 2.0,
  "inference_timesteps": 10,
  "normalize": false,
  "retry_badcase": true
}
```

Request voice cloning:

```json
{
  "text": "Kalimat baru yang harus dibacakan.",
  "reference_audio_base64": "BASE64_DARI_REFERENCE_WAV",
  "prompt_text": "Transkrip tepat yang diucapkan dalam audio referensi.",
  "output_format": "wav",
  "cfg_value": 2.0,
  "inference_timesteps": 10
}
```

Headers:

```text
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

Response mencakup `audio_base64`, `sample_rate`, `duration_seconds`, `output_format`, `model`, dan `generation_seconds`. Sampel WAV dapat diubah menjadi Blob atau ditulis ke file setelah Base64 di-decode.

Saat ini hanya WAV yang diterima sebagai output. Input reference sebaiknya WAV PCM; file MP3/WebM bisa gagal dibaca oleh `soundfile`. Payload audio Base64 berukuran maksimum default 20 MB.

## 9. Konfigurasi Bolt.new

Atur server API base URL menjadi `https://NAMA-APLIKASI.koyeb.app`; endpoint adalah `/tts` dan health check `/health`. Format payload menggunakan `reference_audio_base64`, bukan path file lokal. Respons audio terdapat pada `audio_base64`.

Jangan meletakkan `API_KEY` Koyeb di frontend Bolt.new yang dipublikasikan. Tambahkan backend/proxy agar secret disimpan di server dan browser memanggil proxy tersebut. Jika panggilan langsung dari browser diperlukan selama development, gunakan secret sementara dan jangan publikasikan aplikasi itu.

## 10. Troubleshooting

- **`model_status=error` / CUDA unavailable:** pastikan memilih GPU instance, bukan CPU. Baca runtime logs.
- **Out of memory:** pastikan satu instance, satu worker Uvicorn (`--workers 1`), dan gunakan GPU dengan VRAM memadai. Jangan menaikkan concurrency sebelum diuji.
- **Terlalu lama loading:** unduhan pertama melibatkan beberapa GB. Periksa volume, region, jaringan, log, dan startup grace period. Volume Koyeb saat ini punya batas 10 GB.
- **401:** pastikan `API_KEY` pada Koyeb sama persis dengan `API_KEY` di script test.
- **503 saat `/tts`:** model belum `ready`, atau `API_KEY` belum dikonfigurasi. Poll `/health`.
- **422 `denoise=true`:** set `LOAD_DENOISER=1` pada server atau kirim `denoise=false`.
- **400 audio tidak bisa dibaca:** sediakan WAV PCM yang valid.
- **Timeout:** generasi dapat memerlukan waktu; script test default memberi timeout 900 detik. Aplikasi web/proxy juga harus punya timeout yang cukup besar.
- **Numeral/angka bahasa Indonesia:** coba `normalize=true` di request lalu bandingkan hasilnya; text normalization tetap perlu diuji untuk kasus bahasa Indonesia Anda.

## Sumber resmi

- VoxCPM2 README dan API: https://github.com/OpenBMB/VoxCPM
- Model weights: https://huggingface.co/openbmb/VoxCPM2
- Koyeb volumes: https://www.koyeb.com/docs/reference/volumes
- Koyeb GPU instances: https://www.koyeb.com/docs/reference/instances
- Koyeb health checks: https://www.koyeb.com/docs/run-and-scale/health-checks
- Koyeb scale-to-zero: https://www.koyeb.com/docs/run-and-scale/scale-to-zero
