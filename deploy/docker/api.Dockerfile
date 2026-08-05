# The API image deliberately does NOT install torch.
#
# Embeddings come from the embedding service over HTTP, so the only ML dependency here
# would be dead weight -- and it is a lot of weight: with torch this image is ~4 GB and
# an HPA scale-up spends minutes pulling before it can serve. Without it the image is
# a few hundred MB and a new replica is ready in seconds, which is exactly when you
# need it.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[serve]"


FROM python:3.12-slim

# Non-root by default. The API needs no filesystem writes at runtime, so it also runs
# comfortably with a read-only root filesystem (see the Helm securityContext).
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app
COPY --chown=app:app src ./src
ENV PYTHONPATH=/app/src

# Deprecation facts drive implicit version inference ("you mentioned policy/v1beta1, so
# you're on 1.24 or earlier"). Baked in rather than mounted: it is a small, versioned
# artifact of the same build, and mounting it would let the image and the facts drift.
COPY --chown=app:app data/interim/deprecation_facts.jsonl /app/data/interim/deprecation_facts.jsonl
ENV KVRAG_DATA_DIR=/app/data

USER app
EXPOSE 8080

# Liveness only -- readiness additionally checks the Qdrant collection, which belongs
# in the Kubernetes probe rather than here.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "kubeversion_rag.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
