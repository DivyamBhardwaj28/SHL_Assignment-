FROM python:3.11-slim

# HF Spaces runs containers as a non-root user by convention (UID 1000).
# Doing this correctly up front avoids permission errors writing to
# /home/user or the HF cache at runtime.
RUN useradd -m -u 1000 user

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the BGE embedding model into the image at BUILD time, so runtime
# cold-start never touches the network / HF Hub. Also set HF_HUB_OFFLINE so
# sentence-transformers doesn't even attempt an update-check network call at
# startup (that's the "unauthenticated requests" warning source) -- fully
# offline once built.
ENV HF_HUB_OFFLINE=1
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

COPY --chown=user . .

USER user
ENV HOME=/home/user

# HF Spaces routes traffic to this specific port by convention -- must match
# app_port in README.md's YAML frontmatter.
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
