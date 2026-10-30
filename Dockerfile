# Baymax web server: Whisper (speech to text) + Kokoro (voice) + the safety layers, talking to an Ollama server.
# CPU image: runs anywhere Docker runs. The big model (Ollama) is a separate container; see docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    KOKORO_DIR=/opt/kokoro \
    BAYMAX_AUDIO_CACHE=/opt/audio_cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libsndfile1 espeak-ng curl openssl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. dependencies first (this layer is cached until requirements change)
COPY requirements-server.txt .
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-server.txt

# 2. the code
COPY *.py voice.json LICENSE ./
COPY web ./web
COPY docker/entrypoint.sh /entrypoint.sh

# 3. models that never change, baked in so the first start needs no downloads (about 1 GB)
ARG BAYMAX_WHISPER_MODEL=distil-medium.en
ENV BAYMAX_WHISPER_MODEL=$BAYMAX_WHISPER_MODEL
RUN python -c "from speak import ensure_model_files; ensure_model_files('/opt/kokoro')" \
 && python -c "from faster_whisper import WhisperModel; WhisperModel('$BAYMAX_WHISPER_MODEL', device='cpu', compute_type='int8')" \
 && python -c "from silero_vad import load_silero_vad; load_silero_vad()"

# (the default helpline wording is generic; docker-compose.yml passes the same default at run time, so the cache matches)
ENV BAYMAX_CRISIS_HELPLINE=""
# 4. pre-render the fixed lines (safety replies, greetings, remedies) so they play instantly.
#    Not fatal if it fails: the voice then just synthesizes them live.
RUN python prerender.py --voice-backend kokoro --folder /opt/audio_cache --model-dir /opt/kokoro || echo "prerender skipped"

# 5. run as a normal user; /data holds the memory file (mount a volume there)
RUN useradd --create-home --uid 10001 baymax && mkdir -p /data && chown -R baymax /data /opt \
 && chmod +x /entrypoint.sh
USER baymax
ENV BAYMAX_MEMORY=/data/baymax_memory.db \
    BAYMAX_HOST=0.0.0.0 \
    BAYMAX_PORT=8000
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || curl -fsSk https://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["/entrypoint.sh"]
