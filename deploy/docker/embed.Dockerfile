# The embedding service. This is the image that carries torch.
#
# The CPU-only torch wheel is requested explicitly: the default wheel bundles CUDA
# runtime libraries that add several GB and are pure waste on a CPU node pool.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/venv/bin/pip install ".[serve]" sentence-transformers


FROM python:3.12-slim

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Weights land here. Baked into the image at build time so a pod restart never
    # depends on huggingface.co being reachable -- an external dependency in the
    # startup path is an external dependency in your availability number.
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV KVRAG_EMBED_MODEL=${EMBED_MODEL}

RUN mkdir -p /opt/models \
    && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBED_MODEL}')" \
    && chown -R app:app /opt/models

WORKDIR /app
COPY --chown=app:app src ./src
ENV PYTHONPATH=/app/src

USER app
EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8081/healthz', timeout=2).status==200 else 1)"

# One worker: the model is held in process memory, so N workers means N copies of the
# weights on one pod. Scale with replicas, not workers.
CMD ["uvicorn", "kubeversion_rag.serving.embed_service:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]
