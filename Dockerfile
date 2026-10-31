FROM python:3.12-slim

# system libraries the Python packages need
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libsndfile1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Python packages (change rarely, so they come first)
COPY requirements-server.txt .
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-server.txt

# 2. Models (huge, never change, so they come before your code)
RUN python -c "import urllib.request,os; os.makedirs('/opt/kokoro',exist_ok=True); b='https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'; [urllib.request.urlretrieve(b+f,'/opt/kokoro/'+f) for f in ('kokoro-v1.0.onnx','voices-v1.0.bin')]"
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('distil-medium.en', device='cpu', compute_type='int8')"

# 3. Your code (changes constantly, so it comes last)
COPY *.py voice.json ./
COPY web ./web

ENV KOKORO_DIR=/opt/kokoro BAYMAX_HOST=0.0.0.0 BAYMAX_STT_DEVICE=cpu
EXPOSE 8000
CMD ["python", "server.py"]
