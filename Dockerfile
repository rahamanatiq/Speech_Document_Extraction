FROM python:3.12-slim

WORKDIR /app

# CPU-only torch, installed explicitly and BEFORE requirements.txt.
# easyocr depends on torch/torchvision but doesn't pin a CPU build itself --
# left to default resolution, pip pulls torch's standard PyPI wheel, which
# bundles full CUDA support and ~2GB of nvidia_* libraries that are never
# used here: both adapters force CPU explicitly (device="cpu" in
# whisper_adapter.py, gpu=False in easyocr_adapter.py). Installing the CPU
# wheel first means it's already satisfied by the time easyocr's install
# runs, so the CUDA build is never fetched.
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY adapters/ ./adapters/
COPY services/ ./services/
COPY api/ ./api/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]