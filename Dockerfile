FROM vllm/vllm-omni:nightly

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PORT=8000 \
    GPU_MEMORY_UTILIZATION=0.70 \
    MAX_NUM_SEQS=1

RUN uv pip install \
    --python "$(python3 -c 'import sys; print(sys.executable)')" \
    --no-cache-dir \
    "voxcpm>=2.0" soundfile httpx ninja

EXPOSE 8000

CMD ["sh", "-lc", "exec vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port ${PORT:-8000} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.70} --max-num-seqs ${MAX_NUM_SEQS:-1} --api-key \"$API_KEY\""]
