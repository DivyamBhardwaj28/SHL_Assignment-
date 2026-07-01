FROM python:3.11-slim

WORKDIR /app

# faiss / torch need a C++ toolchain to install cleanly on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheel first -- the default PyPI wheel pulls in ~2GB of CUDA
# libraries you'll never use on a CPU-only host. This alone can be the
# difference between fitting comfortably in a free-tier image size limit
# and not.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the BGE embedding model into the image at BUILD time. Without this,
# every cold start on Render re-downloads/re-resolves the model from the HF
# Hub before your agent can even start serving -- exactly the unauthenticated-
# request warning seen in local runs, except now also on the evaluator's
# clock. Baking it in means runtime start never touches the network for this.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Now copy the actual app code + your prebuilt FAISS index / metadata.
# IMPORTANT: shl_catalog.index and shl_metadata.json must exist in this
# build context (not gitignored) -- the container has no other way to get
# them, there's no build_index.py run step here.
COPY . .

EXPOSE 8000

# Render sets $PORT dynamically; must bind to it, not a hardcoded port.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
